"""
rtx_oom_guard.profiler.allocator_logger — Per-step GPU memory state logger.

Records allocated, reserved, free_estimate, fragmentation_ratio,
step_time, and batch_size at each training step. Works on CPU (zeros
GPU fields) so tests run without hardware.

Usage::

    logger = AllocatorLogger()
    logger.begin_step(batch_size=32)
    # ... training step ...
    logger.end_step()
    logger.to_json("results/run_1.json")
"""

import csv
import json
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

log = logging.getLogger("rtx_oom_guard.allocator_logger")

# GPU helpers (lazy-import torch so module is importable without CUDA)

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _mem_stats() -> Dict[str, float]:
    """Return current GPU memory stats or zeros on CPU."""
    if not _cuda_available():
        return {"allocated": 0.0, "reserved": 0.0}
    import torch
    return {
        "allocated": torch.cuda.memory_allocated() / (1024 ** 2),   # MB
        "reserved": torch.cuda.memory_reserved() / (1024 ** 2),     # MB
    }


# Data structures

@dataclass
class StepRecord:
    """Single training-step memory snapshot."""
    step: int
    allocated_mb: float
    reserved_mb: float
    free_estimate_mb: float
    fragmentation_ratio: float
    step_time_s: float
    batch_size: int
    phase: str = "step"                    # forward / backward / optimizer / step
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Logger

class AllocatorLogger:
    """
    Collects per-step GPU memory telemetry.

    Attributes
    ----------
    records : list[StepRecord]
        All recorded snapshots.
    """

    def __init__(self) -> None:
        self.records: List[StepRecord] = []
        self._step_idx: int = 0
        self._step_start: float = 0.0
        self._current_batch_size: int = 0

    # -- Step lifecycle ----------------------------------------------------

    def begin_step(self, batch_size: int = 0) -> None:
        """Mark the start of a training step."""
        self._step_start = time.perf_counter()
        self._current_batch_size = batch_size

    def end_step(
        self,
        allocated_mb: Optional[float] = None,
        reserved_mb: Optional[float] = None,
    ) -> StepRecord:
        """Mark the end of a training step and record memory state."""
        elapsed = time.perf_counter() - self._step_start
        record = self.snapshot(
            phase="step",
            step_time_s=elapsed,
            batch_size=self._current_batch_size,
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
        )
        self._step_idx += 1
        return record

    # -- Snapshot at arbitrary points --------------------------------------

    def snapshot(
        self,
        phase: str = "manual",
        step_time_s: float = 0.0,
        batch_size: int = 0,
        allocated_mb: Optional[float] = None,
        reserved_mb: Optional[float] = None,
    ) -> StepRecord:
        """Capture a single memory snapshot and append to records."""
        if allocated_mb is not None and reserved_mb is not None:
            allocated = allocated_mb
            reserved = reserved_mb
        else:
            stats = _mem_stats()
            allocated = stats["allocated"]
            reserved = stats["reserved"]

        free_est = reserved - allocated
        frag = 1.0 - (allocated / reserved) if reserved > 0 else 0.0

        rec = StepRecord(
            step=self._step_idx,
            allocated_mb=round(allocated, 3),
            reserved_mb=round(reserved, 3),
            free_estimate_mb=round(free_est, 3),
            fragmentation_ratio=round(frag, 6),
            step_time_s=round(step_time_s, 6),
            batch_size=batch_size or self._current_batch_size,
            phase=phase,
            timestamp=round(time.time(), 3),
        )
        self.records.append(rec)
        return rec

    # -- Export ------------------------------------------------------------

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Return all records as a list of plain dicts."""
        return [r.to_dict() for r in self.records]

    def to_json(self, path: str) -> None:
        """Write records to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dicts(), f, indent=2)
        log.info("Saved %d records → %s", len(self.records), path)

    def to_csv(self, path: str) -> None:
        """Write records to a CSV file."""
        if not self.records:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fields = list(asdict(self.records[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rec in self.records:
                writer.writerow(rec.to_dict())
        log.info("Saved %d records → %s", len(self.records), path)

    # -- Summaries ---------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Compute aggregate statistics over all step-phase records."""
        steps = [r for r in self.records if r.phase == "step"]
        if not steps:
            return {}
        n = len(steps)
        return {
            "total_steps": n,
            "avg_allocated_mb": round(sum(r.allocated_mb for r in steps) / n, 2),
            "peak_reserved_mb": round(max(r.reserved_mb for r in steps), 2),
            "avg_fragmentation": round(sum(r.fragmentation_ratio for r in steps) / n, 6),
            "avg_step_time_s": round(sum(r.step_time_s for r in steps) / n, 6),
        }

    def clear(self) -> None:
        """Reset all records."""
        self.records.clear()
        self._step_idx = 0
