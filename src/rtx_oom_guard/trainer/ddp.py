"""
rtx_oom_guard.trainer.ddp
=====================

Distributed Data Parallel (DDP) logic for multi-GPU fragmentation tracking.
Handles gradient sync wrappers, batch splitting optimizations, and syncing
defragmentation decisions across the process group.
"""

import torch
import torch.distributed as dist

class DDPSyncManager:
    """
    Handles robust distributed synchronization for memory infrastructure.
    """
    def __init__(self):
        self.is_initialized = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_initialized else 0
        self.world_size = dist.get_world_size() if self.is_initialized else 1

        # Track communication overhead
        self.sync_events = []
        self._has_cuda = torch.cuda.is_available()
        if self._has_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_event = None
            self.end_event = None

    def check_global_compaction(self, local_pending: bool) -> bool:
        """
        Synchronizes whether ANY node in the DDP group needs compaction.
        If Node A hits the fragmentation threshold but Node B hasn't, we MUST
        force both nodes to compact. If they fall out of sync, Node A will stall 
        Node B on the next all_reduce autograd pass, leading to systemic lockup.
        """
        if not self.is_initialized:
            return local_pending

        device = torch.device('cuda', torch.cuda.current_device())

        # 1 means pending, 0 means not pending
        flag_tensor = torch.tensor([1 if local_pending else 0], dtype=torch.int32, device=device)

        # Start timing the infrastructural overhead
        if self._has_cuda:
            self.start_event.record()

        # Logical OR across all processes
        dist.all_reduce(flag_tensor, op=dist.ReduceOp.MAX)

        if self._has_cuda:
            self.end_event.record()
            torch.cuda.synchronize() # Only synchronize local stream for event resolution

            overhead_ms = self.start_event.elapsed_time(self.end_event)
            # Keep only the last 50 events for a smooth moving average
            self.sync_events.append(overhead_ms)
            if len(self.sync_events) > 50:
                self.sync_events.pop(0)  # pragma: no cover


        return flag_tensor.item() > 0

    def get_sync_status(self) -> dict:
        """
        Returns a high-level summary of the DDP orchestration state.
        This provides the AeroGrid dashboard with transparency on multi-GPU overhead.
        """
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "is_distributed": self.is_initialized,
            "avg_sync_overhead_ms": round(self.get_avg_overhead(), 3),
            "total_sync_events": len(self.sync_events)
        }

    def get_avg_overhead(self) -> float:
        if not self.sync_events:
            return 0.0  # pragma: no cover
        return sum(self.sync_events) / len(self.sync_events)
