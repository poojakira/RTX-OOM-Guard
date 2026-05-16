import torch
import time
from unittest.mock import MagicMock, patch
from rtx_oom_guard.scheduler.monitor import DefragMonitor
from rtx_oom_guard.utils import DefragConfig

def test_monitor_trigger_defrag():
    """Verify monitor triggers defrag when frag score exceeds threshold."""
    config = DefragConfig(frag_threshold=0.5, cooldown_seconds=0.01)
    mock_engine = MagicMock()
    mock_predictor = MagicMock()
    # predictor(x) -> mock, mock.item() -> 0.8
    mock_predictor.return_value.item.return_value = 0.8  # Above threshold
    
    # Mock time.perf_counter to ensure latency is low (e.g. 1ms)
    with patch("rtx_oom_guard.scheduler.monitor.parse_memory_snapshot") as mock_snap, \
         patch("rtx_oom_guard.scheduler.monitor.torch.from_numpy") as mock_from_numpy, \
         patch("time.perf_counter", side_effect=[1.0, 1.001]), \
         patch("torch.cuda.is_available", return_value=True):
        
        mock_snap.return_value = {"frag_score": 0.3, "total_free": 1000, "total_allocated": 2000, "blocks": []}
        mock_from_numpy.return_value = torch.zeros((1, 10, 4))
        
        monitor = DefragMonitor(compactor=mock_engine, predictor=mock_predictor, config=config)
        monitor._buffer_full = True
        
        # Manually trigger check
        monitor._predict_and_act()
        
        assert mock_engine.defragment_tensors.called
        assert monitor._last_defrag_time > 0

def test_monitor_cooldown():
    """Verify monitor respects cooldown period."""
    config = DefragConfig(frag_threshold=0.5, cooldown_seconds=100.0)
    mock_engine = MagicMock()
    mock_predictor = MagicMock()
    mock_predictor.return_value.item.return_value = 0.8
    
    # Mock time.perf_counter to ensure latency is low everywhere
    # Each call to _predict_and_act uses 2 time.perf_counter calls
    with patch("rtx_oom_guard.scheduler.monitor.parse_memory_snapshot") as mock_snap, \
         patch("rtx_oom_guard.scheduler.monitor.torch.from_numpy"), \
         patch("time.perf_counter", side_effect=[1.0, 1.001, 2.0, 2.001]), \
         patch("torch.cuda.is_available", return_value=True):
        
        mock_snap.return_value = {"frag_score": 0.3, "total_free": 1000, "total_allocated": 2000, "blocks": []}
        
        monitor = DefragMonitor(compactor=mock_engine, predictor=mock_predictor, config=config)
        monitor._buffer_full = True
        
        # First defrag
        monitor._predict_and_act()
        assert mock_engine.defragment_tensors.call_count == 1
        
        # Second defrag (should be blocked by cooldown)
        monitor._predict_and_act()
        assert mock_engine.defragment_tensors.call_count == 1

def test_monitor_kill_switch():
    """Verify monitor stops if prediction latency is too high."""
    config = DefragConfig(max_prediction_latency_ms=0.001) # Extremely low to trigger kill switch
    mock_engine = MagicMock()
    mock_predictor = MagicMock()
    mock_predictor.return_value.item.return_value = 0.1
    
    # Slow prediction
    def slow_predict(*args):
        time.sleep(0.01)
        return MagicMock(item=lambda: 0.1)
    mock_predictor.side_effect = slow_predict
    
    with patch("rtx_oom_guard.scheduler.monitor.parse_memory_snapshot") as mock_snap, \
         patch("rtx_oom_guard.scheduler.monitor.torch.from_numpy"), \
         patch("torch.cuda.is_available", return_value=True):
        
        mock_snap.return_value = {"frag_score": 0.1, "total_free": 1000, "total_allocated": 2000, "blocks": []}
        
        monitor = DefragMonitor(compactor=mock_engine, predictor=mock_predictor, config=config)
        monitor._buffer_full = True # Enable prediction
        monitor._active = True # Manually simulate starting for a standalone call
        
        # This will hit 10ms sleep vs 0.001ms limit
        monitor._predict_and_act()
        
        # Should have stopped itself
        assert monitor._killed == True
        assert monitor._active == False

def test_monitor_no_cuda():
    """Verify monitor handles CPU environment gracefully."""
    with patch("torch.cuda.is_available", return_value=False):
        monitor = DefragMonitor()
        monitor.start()
        # For now, start() still starts the thread even on CPU
        # but auto_record() returns early.
        assert monitor._active == True
        monitor.stop()
        assert monitor._active == False
