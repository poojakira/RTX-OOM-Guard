import torch
from unittest.mock import MagicMock, patch
from rtx_oom_guard.scheduler.monitor import DefragMonitor
from rtx_oom_guard.utils import DefragConfig

def test_monitor_model_loading_branches():
    """Verify monitor handles all model loading fallback paths."""
    config = DefragConfig()
    
    # 1. Load from path (Line 90-92)
    with patch("os.path.exists", return_value=True), \
         patch("rtx_oom_guard.scheduler.monitor.FragPredictor.load") as mock_load:
        
        mock_model = MagicMock()
        mock_model.count_parameters.return_value = 100
        mock_load.return_value = mock_model
        
        monitor = DefragMonitor(model_path="fake.pth", config=config)
        monitor._load_model()
        assert mock_load.called
        assert monitor._model == mock_model

    # 2. Fallback to untrained model (Line 107) - Normal behavior since default_weights.py missing
    with patch("os.path.exists", return_value=False), \
         patch("rtx_oom_guard.scheduler.monitor.FragPredictor.from_config") as mock_untrained:
        
        mock_untrained.return_value = MagicMock()
        monitor = DefragMonitor(model_path="nonexistent.pth", config=config)
        # Import will fail naturally as the file is missing in the workspace
        monitor._load_model()
        assert mock_untrained.called

def test_monitor_load_model_weights_full():
    """Verify loading from base64 default weights (Line 99-105)."""
    # Create a fake default_weights module
    fake_module = MagicMock()
    fake_module.DEFAULT_WEIGHTS_B64 = "YmFzZTY0ZGF0YQ==" # "base64data"
    
    with patch.dict("sys.modules", {"rtx_oom_guard.scheduler.default_weights": fake_module}), \
         patch("os.path.exists", return_value=False), \
         patch("torch.load", return_value={}), \
         patch("rtx_oom_guard.scheduler.monitor.FragPredictor.from_config") as mock_from_cfg:
        
        mock_model = MagicMock()
        mock_from_cfg.return_value = mock_model
        monitor = DefragMonitor(model_path="nonexistent.pth")
        
        # This should trigger Line 102 (b64decode) but fail at torch.load(BytesIO) if data is bad
        # unless we mock torch.load or provided valid base64 for a real state_dict.
        # But we mocked torch.load, so it should pass.
        monitor._load_model()
        assert mock_from_cfg.called
        assert mock_model.load_state_dict.called

def test_monitor_load_model_di():
    """Verify monitor respects predictor provided via DI (Line 87)."""
    predictor = MagicMock()
    monitor = DefragMonitor(predictor=predictor)
    monitor._load_model()
    assert monitor._model == predictor

def test_monitor_already_running_warning():
    """Verify monitor handles multiple start() calls (Line 120-121)."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("rtx_oom_guard.scheduler.monitor.GPUMemoryDefragmenter"):
        monitor = DefragMonitor()
        monitor.start()
        with patch("rtx_oom_guard.scheduler.monitor.log") as mock_log:
            monitor.start()
            assert mock_log.warning.called
        monitor.stop()

def test_monitor_buffer_full_flag():
    """Verify buffer full flag is set (Line 155)."""
    config = DefragConfig()
    config.seq_len = 2
    with patch("rtx_oom_guard.scheduler.monitor.GPUMemoryDefragmenter"):
        monitor = DefragMonitor(config=config)
        monitor.record_alloc(100)
        monitor.record_alloc(100) # buffer_idx wraps to 0
        assert monitor._buffer_full

def test_monitor_auto_record():
    """Verify auto_record logic with extra distinct values (Line 164-165)."""
    mock_mem = MagicMock(side_effect=[110, 220, 330, 440]) 
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", side_effect=mock_mem), \
         patch("rtx_oom_guard.scheduler.monitor.GPUMemoryDefragmenter"):
        monitor = DefragMonitor()
        monitor._last_mem = 0
        monitor.auto_record() 
        assert monitor._buffer_idx == 1
        monitor.auto_record()
        assert monitor._buffer_idx == 2

def test_monitor_kill_switch_trigger():
    """Verify kill switch triggers when latency is high (Line 216-218)."""
    with patch("rtx_oom_guard.scheduler.monitor.parse_memory_snapshot") as mock_snap, \
         patch("rtx_oom_guard.scheduler.monitor.torch.from_numpy"), \
         patch("time.perf_counter", side_effect=[1.0, 1.1]), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("rtx_oom_guard.scheduler.monitor.GPUMemoryDefragmenter"):
        
        mock_snap.return_value = {"frag_score": 0.9}
        monitor = DefragMonitor(predictor=MagicMock())
        monitor._model.return_value = torch.tensor([0.9])
        monitor._buffer_full = True
        monitor._active = True 
        
        monitor._predict_and_act()
        assert monitor._killed == True
        assert monitor._active == False
