import pytest
import torch
from unittest.mock import patch
from rtx_oom_guard.defrag_engine.defragmenter import GPUMemoryDefragmenter

def test_defragmenter_persistence_io_error(tmp_path):
    """Verify defragmenter handles IO errors during telemetry persistence."""
    # Patch the async thread's mkdir call to fail
    with patch("pathlib.Path.mkdir", side_effect=IOError("Disk Full")):
        engine = GPUMemoryDefragmenter(results_dir=str(tmp_path / "fail"))
        # Should not raise locally, as the thread handles its own exceptions
        engine._persist_telemetry(100, 200, force=True)

def test_defragmenter_no_matching_tensors():
    """Verify defragment_tensors() returns safely when no tensors are provided."""
    engine = GPUMemoryDefragmenter()
    res = engine.defragment_tensors([])
    assert res["skipped"] is True

def test_defragmenter_triton_missing_warning():
    """Verify behavior when Triton kernels are missing."""
    with patch("rtx_oom_guard.defrag_engine.defragmenter.HAS_TRITON", False), \
         patch("rtx_oom_guard.defrag_engine.defragmenter.log") as mock_log:
        engine = GPUMemoryDefragmenter(use_triton=True)
        # triton_compaction_copy should now be the dummy one that raises RuntimeError
        t1 = torch.randn(10)
        engine.defragment_tensors([t1])
        assert engine.use_triton is False

def test_defragmenter_oom_raise():
    """Verify non-OOM RuntimeErrors are raised (Line 128)."""
    engine = GPUMemoryDefragmenter()
    t1 = torch.randn(10)
    # Mock torch.empty to raise a generic RuntimeError (not OOM)
    with patch("torch.empty", side_effect=RuntimeError("Generic Error")):
        with pytest.raises(RuntimeError, match="Generic Error"):
            engine.defragment_tensors([t1])

def test_defragmenter_persist_async_fail():
    """Verify exception handling in async telemetry (Line 257, 262)."""
    engine = GPUMemoryDefragmenter()
    # 1. Hit Line 262: Trigger an exception in the main _persist_telemetry block
    # (e.g. results_dir is a file instead of a dir)
    with patch("pathlib.Path.mkdir", side_effect=Exception("Trigger 262")):
        engine._persist_telemetry(100, 200, force=True)

    # 2. Hit Line 257: Trigger an exception inside the async_write thread
    # We'll patch tempfile.mkstemp to fail
    with patch("tempfile.mkstemp", side_effect=Exception("Trigger 257")):
        # We need to ensure the thread runs at least part way
        engine._persist_telemetry(100, 200, force=True)
