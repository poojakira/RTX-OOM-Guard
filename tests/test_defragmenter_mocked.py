import torch
from unittest.mock import MagicMock, patch
from rtx_oom_guard.defrag_engine.defragmenter import GPUMemoryDefragmenter

def test_defragmenter_with_triton_mocked():
    """Verify that defragmenter calls Triton kernel when available."""
    # We need to mock torch.device because it's immutable in PyTorch
    mock_device = MagicMock(spec=torch.device)
    mock_device.type = "cuda"
    mock_device.index = 0
    
    # We need to ensure HAS_TRITON is True during the test
    with patch("rtx_oom_guard.defrag_engine.defragmenter.HAS_TRITON", True), \
         patch("rtx_oom_guard.defrag_engine.defragmenter.triton_compaction_copy") as mock_kernel, \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.empty") as mock_empty:
        
        # Setup mock tensors
        t1 = MagicMock(spec=torch.Tensor)
        t1.device = mock_device
        t1.is_cuda = True
        t1.numel.return_value = 100
        t1.element_size.return_value = 4
        t1.dtype = torch.float32
        t1.view.return_value = t1
        t1.view_as.return_value = t1
        t1.requires_grad = False
        
        engine = GPUMemoryDefragmenter(use_triton=True)
        engine.defragment_tensors([t1], reason="test")
        
        assert mock_kernel.called

def test_defragmenter_oom_on_buffer_allocation():
    """Verify that OOM during buffer allocation is handled gracefully."""
    mock_device = MagicMock(spec=torch.device)
    mock_device.type = "cuda"
    
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.empty", side_effect=RuntimeError("CUDA out of memory")):
        
        t1 = MagicMock(spec=torch.Tensor)
        t1.device = mock_device
        t1.numel.return_value = 100
        t1.element_size.return_value = 4
        t1.dtype = torch.float32
        
        engine = GPUMemoryDefragmenter()
        res = engine.defragment_tensors([t1], reason="test_oom")
        
        # Should not crash, but freed_mb should stay 0 or record empty
        assert res["freed_mb"] == 0

def test_defragmenter_chunking_logic():
    """Verify that tensors are processed in chunks."""
    mock_device = MagicMock(spec=torch.device)
    mock_device.type = "cuda"
    
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.empty") as mock_empty:
        
        # Create many small tensors to force multiple chunks
        # Chunk size is 256MB. 
        tensors = []
        for i in range(10):
            t = MagicMock(spec=torch.Tensor)
            t.device = mock_device
            t.is_cuda = True
            t.numel.return_value = 100 * 1024 * 1024 // 4 # 100MB
            t.element_size.return_value = 4
            t.dtype = torch.float32
            t.view.return_value = t
            t.view_as.return_value = t
            t.requires_grad = False
            tensors.append(t)
            
        engine = GPUMemoryDefragmenter()
        # Mock chunk_size_elements to be small
        with patch("rtx_oom_guard.defrag_engine.defragmenter.log"):
            # 10 tensors of 100MB = 1000MB. With 256MB chunks, should have 4 chunks.
            engine.defragment_tensors(tensors, reason="test_chunks")
            
        # torch.empty should be called once per chunk
        assert mock_empty.call_count >= 4

def test_defragmenter_ddp_barrier_mocked():
    """Verify DDP barrier call during defragmentation."""
    with patch("torch.distributed.is_available", return_value=True), \
         patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.barrier") as mock_barrier, \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.empty"):
        
        t1 = MagicMock(spec=torch.Tensor)
        t1.numel.return_value = 10
        t1.element_size.return_value = 4
        t1.dtype = torch.float32
        t1.device.type = "cuda"
        
        engine = GPUMemoryDefragmenter()
        engine.defragment_tensors([t1], reason="test_ddp")
        
        assert mock_barrier.called
