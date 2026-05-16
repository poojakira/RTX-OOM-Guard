from unittest.mock import MagicMock, patch
from rtx_oom_guard.trainer.ddp import DDPSyncManager

def test_ddp_manager_uninitialized():
    """Verify DDP manager behavior when distributed is not initialized."""
    with patch("torch.distributed.is_available", return_value=True), \
         patch("torch.distributed.is_initialized", return_value=False):
        
        manager = DDPSyncManager()
        assert manager.is_initialized == False
        assert manager.rank == 0
        assert manager.world_size == 1
        
        # Should just return local status
        assert manager.check_global_compaction(True) == True
        assert manager.check_global_compaction(False) == False

def test_ddp_manager_initialized_mocked():
    """Verify DDP manager synchronization logic with mocks."""
    with patch("torch.distributed.is_available", return_value=True), \
         patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=1), \
         patch("torch.distributed.get_world_size", return_value=4), \
         patch("torch.distributed.all_reduce") as mock_all_reduce, \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.current_device", return_value=0), \
         patch("torch.cuda.Event") as mock_event_cls:
        
        # Setup event timing mocks
        mock_start = MagicMock()
        mock_end = MagicMock()
        mock_event_cls.side_effect = [mock_start, mock_end]
        mock_start.elapsed_time.return_value = 5.5
        
        manager = DDPSyncManager()
        assert manager.is_initialized == True
        assert manager.rank == 1
        assert manager.world_size == 4
        
        # Mock the flag tensor result (simulating someone else having pending=True)
        def mock_all_reduce_effect(tensor, op):
            tensor[0] = 1 # Force as if someone had True
        mock_all_reduce.side_effect = mock_all_reduce_effect
        
        with patch("torch.cuda.synchronize"), \
             patch("torch.tensor") as mock_tensor_fn:
            # Bypass CUDA error in torch.tensor([1], device='cuda')
            mock_tensor_fn.return_value = MagicMock(item=lambda: 1)
            result = manager.check_global_compaction(False)
            
        assert result == True
        assert mock_all_reduce.called
        assert len(manager.sync_events) == 1
        assert manager.sync_events[0] == 5.5
        
        status = manager.get_sync_status()
        assert status["rank"] == 1
        assert status["avg_sync_overhead_ms"] == 5.5

def test_ddp_manager_no_cuda():
    """Verify DDP manager on CPU-only distributed setup."""
    with patch("torch.distributed.is_available", return_value=True), \
         patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=0), \
         patch("torch.distributed.get_world_size", return_value=1), \
         patch("torch.distributed.all_reduce"), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.current_device", return_value=0):
        
        with patch("torch.device") as mock_dev_cls, \
             patch("torch.tensor") as mock_tensor_fn:
            mock_dev_cls.return_value = MagicMock()
            mock_tensor_fn.return_value = MagicMock(item=lambda: 1 if True else 0)
            
            manager = DDPSyncManager()
            assert manager.start_event is None
            
            # Should still work but without timing
            result = manager.check_global_compaction(True)
            assert result == True
            assert len(manager.sync_events) == 0
