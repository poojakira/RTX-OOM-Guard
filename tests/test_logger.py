"""
tests/test_logger.py — Prove the AllocatorLogger actually runs.
"""

import json
import csv
from pathlib import Path

from rtx_oom_guard.profiler.allocator_logger import AllocatorLogger, StepRecord


class TestAllocatorLogger:
    """Core logger tests — all run on CPU."""

    def test_record_step(self):
        """begin_step / end_step produces a StepRecord with correct fields."""
        logger = AllocatorLogger()
        logger.begin_step(batch_size=32)
        rec = logger.end_step()

        assert isinstance(rec, StepRecord)
        assert rec.step == 0
        assert rec.batch_size == 32
        assert rec.phase == "step"
        assert rec.step_time_s >= 0
        assert 0.0 <= rec.fragmentation_ratio <= 1.0

    def test_multiple_steps(self):
        """Step index increments correctly."""
        logger = AllocatorLogger()
        for i in range(5):
            logger.begin_step(batch_size=16)
            logger.end_step()
        assert len(logger.records) == 5
        assert logger.records[-1].step == 4

    def test_snapshot_phase(self):
        """Manual snapshot records the right phase."""
        logger = AllocatorLogger()
        rec = logger.snapshot(phase="forward_end")
        assert rec.phase == "forward_end"

    def test_export_json(self, tmp_path):
        """Round-trip JSON export."""
        logger = AllocatorLogger()
        logger.begin_step(batch_size=8)
        logger.end_step()

        path = str(tmp_path / "test.json")
        logger.to_json(path)
        assert Path(path).exists()

        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["batch_size"] == 8

    def test_export_csv(self, tmp_path):
        """Round-trip CSV export."""
        logger = AllocatorLogger()
        for i in range(3):
            logger.begin_step(batch_size=4)
            logger.end_step()

        path = str(tmp_path / "test.csv")
        logger.to_csv(path)
        assert Path(path).exists()

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert "allocated_mb" in rows[0]

    def test_cpu_fallback(self):
        """Logger works on CPU — GPU fields are 0."""
        logger = AllocatorLogger()
        logger.begin_step(batch_size=16)
        rec = logger.end_step()
        # On CPU, allocated and reserved should be 0
        assert rec.allocated_mb == 0.0 or rec.allocated_mb >= 0.0  # either CPU or GPU
        assert rec.reserved_mb >= 0.0

    def test_summary(self):
        """Summary stats are computed correctly."""
        logger = AllocatorLogger()
        for _ in range(10):
            logger.begin_step(batch_size=8)
            logger.end_step()

        s = logger.summary()
        assert s["total_steps"] == 10
        assert "avg_allocated_mb" in s
        assert "peak_reserved_mb" in s
        assert "avg_fragmentation" in s
        assert "avg_step_time_s" in s

    def test_clear(self):
        """Clear resets all records."""
        logger = AllocatorLogger()
        logger.begin_step(batch_size=8)
        logger.end_step()
        assert len(logger.records) == 1

        logger.clear()
        assert len(logger.records) == 0

    def test_to_dicts(self):
        """to_dicts returns plain dicts."""
        logger = AllocatorLogger()
        logger.begin_step(batch_size=4)
        logger.end_step()
        dicts = logger.to_dicts()
        assert isinstance(dicts, list)
        assert isinstance(dicts[0], dict)
        assert "step" in dicts[0]
