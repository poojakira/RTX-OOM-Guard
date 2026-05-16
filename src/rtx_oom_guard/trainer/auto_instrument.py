"""
rtx_oom_guard.trainer.auto_instrument — Zero-Code-Change PyTorch Injector.

This module provides enterprise-level autoinstrumentation for any PyTorch
model and optimizer. It dynamically intercepts forward passes, backward passes,
and optimizer steps using PyTorch hooks, abstracting the `TrainingHook` entirely
away from the user's workload code.

Usage:
    from rtx_oom_guard import auto_instrument
    model, optimizer = auto_instrument(model, optimizer)
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from rtx_oom_guard.defrag_engine.defragmenter import GPUMemoryDefragmenter
from rtx_oom_guard.trainer.training_hook import TrainingHook
from rtx_oom_guard.defrag_engine.policy import MitigationPolicy
from rtx_oom_guard.scheduler.risk_model import OOMRiskModel
from rtx_oom_guard.trainer.ddp import DDPSyncManager

log = logging.getLogger("rtx_oom_guard.auto_instrument")


class _InstrumentedModel(nn.Module):
    """
    Thin wrapper over the user's root model that triggers telemetry
    on every forward pass and intercepts backward hooks on the 
    root gradient edges.
    """

    def __init__(self, model: nn.Module, hook: TrainingHook):
        super().__init__()
        self.module = model
        self.hook = hook

        # Register forward hook directly on the root module
        self.module.register_forward_pre_hook(self._forward_pre_hook)
        self.module.register_forward_hook(self._forward_post_hook)

    def _forward_pre_hook(self, module, inputs):
        self.hook.on_forward_begin()

    def _forward_post_hook(self, module, inputs, output):
        self.hook.on_forward_end()
        # The backward pass starts right after the loss is derived from this output.
        # We assume the user creates loss and calls backward.
        self.hook.on_backward_begin()

        # We can loosely register a backward hook on the output tensor if it requires grad
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.register_hook(self._backward_done_hook)
        elif isinstance(output, (list, tuple)):
            for o in output:
                if isinstance(o, torch.Tensor) and o.requires_grad:
                    o.register_hook(self._backward_done_hook)

    def _backward_done_hook(self, grad):
        self.hook.on_backward_end()
        return grad

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _InstrumentedOptimizer:
    """
    Wrapper around the PyTorch optimizer to intercept `.step()`.
    """
    def __init__(self, optimizer: Optimizer, hook: TrainingHook, policy: MitigationPolicy, model: nn.Module, ddp_manager: DDPSyncManager):
        self.optimizer = optimizer
        self.hook = hook
        self.policy = policy
        self.model = model
        self.ddp_manager = ddp_manager

    @property
    def __class__(self):
        """Spoof class to pass isinstance(opt, Optimizer) checks in LR Schedulers."""
        return self.optimizer.__class__

    def __getattr__(self, name):
        """Transparently forward all undefined attributes to the real optimizer."""
        return getattr(self.optimizer, name)

    def step(self, closure=None):
        self.hook.on_optimizer_step()
        result = self.optimizer.step(closure)

        # Extract batch size heuristically if possible, default to 1
        batch_size = 1
        risk = self.hook.on_step_complete(batch_size=batch_size)

        # Determine strict DDP global consistency BEFORE evaluating local policy
        local_pending = risk >= self.policy.act_threshold
        global_act = self.ddp_manager.check_global_compaction(local_pending)

        # Dispatch the mitigation policy transparently
        self.policy.evaluate(
            risk_score=risk,
            current_batch_size=batch_size,
            tensors_to_defragment=self.model.parameters(),
            force_act=global_act
        )
        return result

    def zero_grad(self, set_to_none=False):
        return self.optimizer.zero_grad(set_to_none=set_to_none)


def auto_instrument(
    model: nn.Module,
    optimizer: Optimizer,
    risk_threshold: float = 0.8,
    use_triton: bool = True
) -> Tuple[nn.Module, Optimizer]:
    """
    Zero-code-change instrumentation for PyTorch workloads.
    
    Transforms standard models and optimizers into rtx_oom_guard-aware 
    components that automatically report structural memory diagnostics 
    and invoke custom Triton defragmentation kernels immediately prior 
    to encountering Out-Of-Memory exceptions.
    
    Args:
        model: PyTorch nn.Module instance
        optimizer: PyTorch Optimizer instance
        risk_threshold: The utilization threshold before triggering compaction
        use_triton: Activate native Triton zero-copy compaction kernels
        
    Returns:
        (instrumented_model, instrumented_optimizer)
    """
    log.info("Applying Zero-Code-Change generic Auto-Instrumentation...")

    # Initialize the entire intelligence stack silently
    risk_model = OOMRiskModel(mode="rule")
    hook = TrainingHook(risk_model=risk_model)

    engine = GPUMemoryDefragmenter(use_triton=use_triton)
    policy = MitigationPolicy(act_threshold=risk_threshold, engine=engine)
    ddp_manager = DDPSyncManager()

    # Force an initial heartbeat to create results/live_telemetry.json immediately
    try:
        import torch
        if torch.cuda.is_available():
            engine._persist_telemetry(torch.cuda.memory_allocated() / 1024**2, torch.cuda.memory_reserved() / 1024**2)
    except Exception:
        pass

    # Wrap user objects
    wrapped_model = _InstrumentedModel(model, hook)
    wrapped_optimizer = _InstrumentedOptimizer(optimizer, hook, policy, wrapped_model.module, ddp_manager)

    return wrapped_model, wrapped_optimizer
