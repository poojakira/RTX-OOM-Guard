import torch
import pytest
from unittest.mock import MagicMock, patch

# We mock triton before importing the kernels
with patch.dict("sys.modules", {"triton": MagicMock(), "triton.language": MagicMock()}):
    from rtx_oom_guard.defrag_engine.kernels import triton_compaction_copy, analyze_fragmentation_triton

def test_triton_compaction_copy_mocked():
    """Verify Triton compaction copy wrapper logic."""
    src = torch.randn(100, device='cpu')
    dst = torch.randn(100, device='cpu')
    
    # triton_compaction_copy handles the cuda check internally
    # We need to mock is_cuda to True for the tensors
    with patch.object(torch.Tensor, 'is_cuda', True):
        # Even on CPU, we can test the wrapper logic if we mock the kernel call
        with patch("rtx_oom_guard.defrag_engine.kernels._compaction_copy_kernel") as mock_kernel:
            res = triton_compaction_copy(src, dst)
            assert res is dst
            # The kernel is called as mock_kernel[grid](...)
            assert mock_kernel.__getitem__.return_value.called

def test_analyze_fragmentation_triton_mocked():
    """Verify Triton fragmentation analysis wrapper logic."""
    block_sizes = torch.tensor([1024, -2048, 4096], dtype=torch.long)
    
    with patch.object(torch.Tensor, 'is_cuda', True), \
         patch.object(torch.Tensor, 'cuda', return_value=block_sizes), \
         patch("rtx_oom_guard.defrag_engine.kernels._fragmentation_scan_kernel") as mock_kernel, \
         patch.object(torch.Tensor, 'mean', return_value=torch.tensor(0.5)):
        
        score = analyze_fragmentation_triton(block_sizes)
        assert score == 0.5
        assert mock_kernel.__getitem__.return_value.called

def test_triton_compaction_copy_invalid_inputs():
    """Verify error handling for mismatched or non-cuda tensors."""
    t1 = torch.randn(10)
    t2 = torch.randn(20) # Mismatched size
    
    with pytest.raises(AssertionError):
        triton_compaction_copy(t1, t2)
