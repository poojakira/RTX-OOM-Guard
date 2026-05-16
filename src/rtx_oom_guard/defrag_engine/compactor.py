"""
rtx_oom_guard.compactor — Memory compaction engine.

Implements the actual defragmentation logic:
1. Synchronize all CUDA streams
2. Release the CUDA memory cache
3. Optionally force a garbage collection cycle
4. Record compaction metrics for analysis
"""

import gc
import time
import torch
from typing import Dict, List
from rtx_oom_guard.utils import get_logger

log = get_logger("compactor")


class MemoryCompactor:
    """
    GPU memory compaction engine.

    Performs controlled memory cleanup operations when triggered by the
    DefragMonitor. Tracks all compaction events for post-analysis.
    """

    def __init__(self, force_gc: bool = True):
        self.force_gc = force_gc
        self._history: List[Dict] = []
        self._total_compactions = 0

    def compact(self, reason: str = "threshold") -> Dict:
        """
        Execute a memory compaction cycle.

        Steps:
            1. Record pre-compaction memory state
            2. Synchronize all CUDA streams (barrier)
            3. Empty the CUDA memory cache
            4. Optionally force Python garbage collection
            5. Record post-compaction memory state

        Returns:
            Dict with compaction metrics
        """
        if not torch.cuda.is_available():
            return {"skipped": True, "reason": "no_cuda"}

        t0 = time.perf_counter()

        # Pre-state
        pre_allocated = torch.cuda.memory_allocated()
        pre_reserved = torch.cuda.memory_reserved()
        pre_frag = 1.0 - (pre_allocated / pre_reserved) if pre_reserved > 0 else 0.0

        # Compact
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if self.force_gc:
            gc.collect()

        # Post-state
        post_allocated = torch.cuda.memory_allocated()
        post_reserved = torch.cuda.memory_reserved()
        post_frag = 1.0 - (post_allocated / post_reserved) if post_reserved > 0 else 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._total_compactions += 1

        record = {
            "timestamp": time.time(),
            "reason": reason,
            "pre_allocated_mb": pre_allocated / (1024**2),
            "post_allocated_mb": post_allocated / (1024**2),
            "pre_reserved_mb": pre_reserved / (1024**2),
            "post_reserved_mb": post_reserved / (1024**2),
            "freed_mb": (pre_reserved - post_reserved) / (1024**2),
            "pre_frag": pre_frag,
            "post_frag": post_frag,
            "frag_reduction": pre_frag - post_frag,
            "elapsed_ms": elapsed_ms,
            "compaction_id": self._total_compactions,
        }

        self._history.append(record)

        log.info(
            "Compaction #%d: freed %.1f MB (%.1f%% → %.1f%% frag) in %.1fms [%s]",
            self._total_compactions,
            record["freed_mb"],
            pre_frag * 100,
            post_frag * 100,
            elapsed_ms,
            reason,
        )

        return record

    @property
    def history(self) -> List[Dict]:
        return self._history.copy()

    @property
    def total_compactions(self) -> int:
        return self._total_compactions

    @property
    def total_freed_mb(self) -> float:
        return sum(r["freed_mb"] for r in self._history)
