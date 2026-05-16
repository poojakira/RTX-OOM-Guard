import json
from unittest.mock import patch
from rtx_oom_guard.profiler.allocator_logger import AllocatorLogger, _cuda_available, _mem_stats

def test_allocator_logger_cuda_unavailable():
    """Verify memory stats when CUDA is explicitly unavailable (Line 41-42)."""
    with patch("rtx_oom_guard.profiler.allocator_logger._cuda_available", return_value=False):
        stats = _mem_stats()
        assert stats["allocated"] == 0.0
        assert stats["reserved"] == 0.0

def test_allocator_logger_torch_import_error():
    """Verify fallback when torch cannot be imported (Line 35-36)."""
    with patch.dict("sys.modules", {"torch": None}):
        # In a real environment, this would raise ImportError on 'import torch'
        # but our helper catches it.
        try:
            import torch
            # If we are here, the patch didn't work as expected for the helper
        except ImportError:
            pass
        # The helper specifically catches ImportError
        with patch("builtins.__import__", side_effect=ImportError):
            assert _cuda_available() is False

def test_allocator_logger_explicit_snapshot(tmp_path):
    """Verify snapshot with explicit memory values (Line 127-128)."""
    logger = AllocatorLogger()
    logger.snapshot(allocated_mb=100.0, reserved_mb=200.0)
    assert logger.records[0].allocated_mb == 100.0
    assert logger.records[0].fragmentation_ratio == 0.5

def test_allocator_logger_empty_exports(tmp_path):
    """Verify exports and summary handle empty records (Line 167, 183)."""
    logger = AllocatorLogger()
    
    # 1. to_csv empty (Line 167)
    csv_path = tmp_path / "empty.csv"
    logger.to_csv(str(csv_path))
    assert not csv_path.exists()
    
    # 2. summary empty (Line 183)
    assert logger.summary() == {}

def test_allocator_logger_csv_export(tmp_path):
    """Verify CSV export with data."""
    logger = AllocatorLogger()
    logger.snapshot(phase="step", allocated_mb=10, reserved_mb=20)
    csv_path = tmp_path / "data.csv"
    logger.to_csv(str(csv_path))
    assert csv_path.exists()
    with open(csv_path, "r") as f:
        content = f.read()
        assert "allocated_mb" in content
        assert "10" in content

def test_allocator_logger_json_export(tmp_path):
    """Verify JSON export with data."""
    logger = AllocatorLogger()
    logger.snapshot(phase="step", allocated_mb=10, reserved_mb=20)
    json_path = tmp_path / "data.json"
    logger.to_json(str(json_path))
    assert json_path.exists()
    with open(json_path, "r") as f:
        data = json.load(f)
        assert data[0]["allocated_mb"] == 10.0

def test_allocator_logger_full_summary():
    """Verify summary statistics calculation."""
    logger = AllocatorLogger()
    logger.snapshot(phase="step", allocated_mb=100, reserved_mb=200, step_time_s=1.0)
    logger.snapshot(phase="step", allocated_mb=200, reserved_mb=400, step_time_s=2.0)
    s = logger.summary()
    assert s["total_steps"] == 2
    assert s["avg_allocated_mb"] == 150.0
    assert s["peak_reserved_mb"] == 400.0
    assert s["avg_step_time_s"] == 1.5
