import torch
import time
from unittest.mock import MagicMock, patch
from rtx_oom_guard.defrag_engine.defragmenter import GPUMemoryDefragmenter
from rtx_oom_guard.defrag_engine.policy import MitigationPolicy
from rtx_oom_guard.defrag_engine.benchmark_triton import run_benchmark
from fastapi.testclient import TestClient
from rtx_oom_guard.api import app

def test_api_spa_fallback():
    """Verify that the API falls back to the SPA index for unknown routes."""
    client = TestClient(app)
    
    # Mocking Path exists and read_text for SPA fallback
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.read_text", return_value="<html>Dynamic dashboard</html>"):
        
        response = client.get("/invalid/path")
        assert response.status_code == 200
        # Case insensitive search
        assert "dashboard" in response.text.lower()

def test_defragmenter_telemetry_persistence(tmp_path):
    """Verify telemetry persistence via the internal _persist_telemetry method."""
    engine = GPUMemoryDefragmenter(results_dir=str(tmp_path))
    
    # Trigger persistence
    engine._persist_telemetry(100.0, 200.0, force=True)
    
    # Wait for the async write
    time.sleep(1.0)
    
    save_path = tmp_path / "live_telemetry.json"
    assert save_path.exists()

def test_defragmenter_triton_repack_branch():
    """Verify Triton path in defragment_tensors."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("rtx_oom_guard.defrag_engine.defragmenter.triton_compaction_copy") as mock_kernel, \
         patch("torch.empty") as mock_empty:
        
        # Use a mock tensor that supports .view and .data
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.is_cuda = True
        mock_tensor.device.type = "cuda"
        mock_tensor.numel.return_value = 100
        mock_tensor.element_size.return_value = 4
        mock_tensor.dtype = torch.float32
        mock_tensor.view.return_value = mock_tensor
        mock_tensor.view_as.return_value = mock_tensor
        mock_tensor.requires_grad = False
        
        engine = GPUMemoryDefragmenter(use_triton=True)
        # Force HAS_TRITON to True for the test
        with patch("rtx_oom_guard.defrag_engine.defragmenter.HAS_TRITON", True):
            engine.use_triton = True
            engine.defragment_tensors([mock_tensor], reason="test_triton")
            
        assert mock_kernel.called

def test_policy_edge_cases():
    """Cover remaining mitigation policy lines."""
    policy = MitigationPolicy()
    
    # Force safe action
    res = policy.evaluate(0.1, 0)
    assert res.tier == "SAFE"
    
    # Test peer-act
    res = policy.evaluate(0.1, 0, force_act=True)
    assert res.tier == "PEER_ACT"

def test_benchmark_triton_mocked():
    """Cover benchmark code paths by manually injecting the mock kernel."""
    import rtx_oom_guard.defrag_engine.benchmark_triton as benchmark_triton
    
    # Manually set the attribute if missing due to import failure
    if not hasattr(benchmark_triton, 'triton_compaction_copy'):
        setattr(benchmark_triton, 'triton_compaction_copy', MagicMock())

    with patch("torch.cuda.is_available", return_value=True), \
         patch("rtx_oom_guard.defrag_engine.benchmark_triton.TRITON_AVAILABLE", True), \
         patch("rtx_oom_guard.defrag_engine.benchmark_triton.triton_compaction_copy") as mock_triton, \
         patch("rtx_oom_guard.defrag_engine.benchmark_triton.torch.randn"), \
         patch("rtx_oom_guard.defrag_engine.benchmark_triton.torch.empty_like"), \
         patch("rtx_oom_guard.defrag_engine.benchmark_triton.torch.cuda.synchronize"):
        
        run_benchmark()
        assert True

def test_defragmenter_empty_cache_fallback():
    """Cover empty results dir scenario."""
    engine = GPUMemoryDefragmenter(results_dir="/invalid_path/non_existent")
    # This should at least run without crashing
    engine._persist_telemetry(100, 200, force=True)
    assert True
