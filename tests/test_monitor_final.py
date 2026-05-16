import torch
from unittest.mock import MagicMock, patch
from rtx_oom_guard.scheduler.monitor import DefragMonitor

def test_monitor_low_prediction_score():
    """Verify monitor skips defrag if prediction score is low."""
    mock_compactor = MagicMock()
    mock_predictor = MagicMock()
    # Mocking the __call__ since it's used as a model
    mock_predictor.return_value = torch.tensor([[0.1]])
    
    monitor = DefragMonitor(compactor=mock_compactor, predictor=mock_predictor)
    monitor._buffer_full = True 
    
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", return_value=50), \
         patch("torch.cuda.memory_reserved", return_value=100):
        monitor._predict_and_act()
    
    assert not mock_compactor.defragment_tensors.called

def test_monitor_high_util_trigger():
    """Verify monitor triggers defrag if risk is high."""
    mock_compactor = MagicMock()
    mock_predictor = MagicMock()
    mock_predictor.return_value = torch.tensor([[0.9]]) # risk high
    
    monitor = DefragMonitor(compactor=mock_compactor, predictor=mock_predictor)
    monitor._buffer_full = True
    
    with patch("torch.cuda.is_available", return_value=True):
        monitor._predict_and_act()
    
    assert mock_compactor.defragment_tensors.called

def test_monitor_load_model_fallback():
    """Verify monitor handles model load failure by falling back to untrained model."""
    mock_predictor = MagicMock()
    # Load fails, but from_config succeeds
    with patch("rtx_oom_guard.scheduler.monitor.FragPredictor.load", side_effect=Exception("Load fail")), \
         patch("rtx_oom_guard.scheduler.monitor.FragPredictor.from_config", return_value=mock_predictor), \
         patch("pathlib.Path.exists", return_value=True):
        
        monitor = DefragMonitor(model_path="some/path")
        monitor._load_model()
        assert monitor._model is not None

def test_monitor_thread_stop_safety():
    """Verify monitor stop() is idempotent and safe."""
    monitor = DefragMonitor()
    monitor.stop()
    monitor.stop()
    assert True

def test_monitor_record_alloc_api():
    """Verify record_alloc API usage."""
    monitor = DefragMonitor()
    monitor.record_alloc(1024, is_alloc=True)
    assert monitor._buffer_idx == 1
