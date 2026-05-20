"""
DDP-safe coordination for distributed defragmentation.

Fixes the deadlock issue where calling barrier() from a daemon thread
would hang NCCL. Instead, uses a flag-based approach where the monitor
sets a pending_compaction flag, and the training loop checks it at a
safe synchronization point.
"""

import torch
import torch.distributed as dist
import logging
from typing import Optional

log = logging.getLogger("rtx_oom_guard.ddp_coord")


class DDPCoordinator:
    """Coordinates defragmentation across DDP ranks without daemon-thread barriers.

    Instead of calling barrier() from the monitor thread (which deadlocks NCCL),
    this uses an all_reduce(MAX) called FROM THE TRAINING LOOP where all ranks
    participate synchronously.
    """

    def __init__(self):
        self._pending_compaction = False
        self._rank: int = 0
        self._world_size: int = 1

    @property
    def is_distributed(self) -> bool:
        return dist.is_initialized() and dist.get_world_size() > 1

    def request_compaction(self):
        """Called by monitor thread — sets flag, does NOT call barrier."""
        self._pending_compaction = True
        log.debug("Compaction requested (flag set)")

    def check_and_sync(self) -> bool:
        """Called from training loop — synchronizes compaction decision across ranks.

        Returns True if ALL ranks agree compaction should happen.
        Must be called at the same point in the training loop on all ranks.
        """
        if not self.is_distributed:
            result = self._pending_compaction
            self._pending_compaction = False
            return result

        # Use all_reduce(MAX) so if ANY rank wants compaction, all do it
        flag = torch.tensor([1.0 if self._pending_compaction else 0.0], device="cuda")
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)

        should_compact = flag.item() > 0.5
        self._pending_compaction = False

        if should_compact:
            log.info(f"Rank {dist.get_rank()}: synchronized compaction decision")

        return should_compact
