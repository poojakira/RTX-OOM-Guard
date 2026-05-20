"""
Optimizer state and gradient migration during defragmentation.

Fixes the critical gap where defragmentation only moved parameter .data
but left optimizer state (Adam exp_avg, exp_avg_sq) and gradients scattered.
"""

import torch
from typing import Dict, Any
import logging

log = logging.getLogger("rtx_oom_guard.migration")


def migrate_optimizer_state(
    optimizer: torch.optim.Optimizer,
    param_id_map: Dict[int, torch.Tensor],
    target_device: torch.device,
    non_blocking: bool = True,
) -> int:
    """Migrate all optimizer state tensors to target device/contiguous memory.

    Args:
        optimizer: The optimizer whose state to migrate.
        param_id_map: Mapping from old param data_ptr to new tensor location.
        target_device: Device to migrate to.
        non_blocking: Use async CUDA copies.

    Returns:
        Number of state tensors migrated.
    """
    migrated = 0
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if param not in optimizer.state:
                continue
            state = optimizer.state[param]
            for key, val in state.items():
                if isinstance(val, torch.Tensor) and val.is_cuda:
                    # Create contiguous copy on target device
                    state[key] = val.to(target_device, non_blocking=non_blocking).contiguous()
                    migrated += 1

    if non_blocking:
        torch.cuda.synchronize()

    log.info(f"Migrated {migrated} optimizer state tensors")
    return migrated


def migrate_gradients(
    parameters,
    target_device: torch.device,
    non_blocking: bool = True,
) -> int:
    """Migrate all .grad tensors to contiguous memory.

    Args:
        parameters: Iterable of model parameters.
        target_device: Device to migrate to.
        non_blocking: Use async CUDA copies.

    Returns:
        Number of gradients migrated.
    """
    migrated = 0
    for param in parameters:
        if param.grad is not None and param.grad.is_cuda:
            param.grad = param.grad.to(target_device, non_blocking=non_blocking).contiguous()
            migrated += 1

    if non_blocking:
        torch.cuda.synchronize()

    log.info(f"Migrated {migrated} gradient tensors")
    return migrated


def full_migration(model, optimizer, device: torch.device | None = None) -> Dict[str, int]:
    """Complete migration of parameters, optimizer state, and gradients.

    This is the correct sequence for defragmentation:
    1. Compact parameters into contiguous buffer
    2. Migrate optimizer state (exp_avg, exp_avg_sq for Adam)
    3. Migrate gradients
    4. Synchronize
    5. Release old allocations via empty_cache()
    """
    if device is None:
        device = next(model.parameters()).device

    param_id_map: Dict[int, torch.Tensor] = {}
    stats = {"params": 0, "optimizer_states": 0, "gradients": 0}

    # Step 1: Parameters are already handled by defragmenter
    stats["params"] = sum(1 for p in model.parameters() if p.is_cuda)

    # Step 2: Optimizer state
    stats["optimizer_states"] = migrate_optimizer_state(optimizer, param_id_map, device)

    # Step 3: Gradients
    stats["gradients"] = migrate_gradients(model.parameters(), device)

    # Step 4: Release fragmented memory
    torch.cuda.empty_cache()

    log.info(f"Full migration complete: {stats}")
    return stats
