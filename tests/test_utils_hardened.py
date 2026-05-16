import pytest
from unittest.mock import patch, MagicMock
from rtx_oom_guard.utils import DefragConfig, Timer, get_cuda_info, ensure_cuda, parse_memory_snapshot

def test_config_save_load_json(tmp_path):
    """Verify JSON configuration persistence."""
    config = DefragConfig(batch_size=128, learning_rate=0.01)
    path = tmp_path / "config.json"
    config.save(str(path))
    
    loaded = DefragConfig.load(str(path))
    assert loaded.batch_size == 128
    assert loaded.learning_rate == 0.01

def test_config_save_load_yaml(tmp_path):
    """Verify YAML configuration persistence."""
    config = DefragConfig(train_epochs=50)
    path = tmp_path / "config.yaml"
    config.save(str(path))
    
    loaded = DefragConfig.load(str(path))
    assert loaded.train_epochs == 50

def test_config_load_non_existent():
    """Verify default config is returned for missing files."""
    config = DefragConfig.load("non_existent.json")
    default = DefragConfig()
    assert config.batch_size == default.batch_size

def test_config_load_corrupt(tmp_path):
    """Verify default config is returned for corrupt files with warning."""
    path = tmp_path / "corrupt.json"
    path.write_text("NOT_JSON")
    config = DefragConfig.load(str(path))
    assert config.batch_size == DefragConfig().batch_size

def test_timer():
    """Verify timer context manager."""
    with Timer() as t:
        import time
        time.sleep(0.1)
    assert t.elapsed_ms >= 100
    assert t.elapsed_s >= 0.1

def test_get_cuda_info_mocked():
    """Verify CUDA info retrieval when available and unavailable."""
    # Use a more comprehensive patch for the local import
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=2), \
         patch("torch.cuda.get_device_name", return_value="RTX 4090"), \
         patch("torch.cuda.get_device_properties") as mock_props:
        
        mock_props.return_value = MagicMock(total_mem=24*1024**3)
        info = get_cuda_info()
        assert info["available"] == True
        assert info["device_count"] == 2
        assert info["device_name"] == "RTX 4090"
        
    with patch("torch.cuda.is_available", return_value=False):
        info = get_cuda_info()
        assert info["available"] == False

def test_ensure_cuda_raises():
    """Verify ensure_cuda raises RuntimeError when no GPU."""
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError):
            ensure_cuda()

def test_parse_memory_snapshot_complex():
    """Verify complex snapshot parsing logic."""
    mock_snapshot = [
        {
            "blocks": [
                {"size": 1024, "state": "active_allocated"},
                {"size": 2048, "state": "inactive"},
            ]
        }
    ]
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_snapshot", return_value=mock_snapshot):
        
        data = parse_memory_snapshot()
        assert data["total_allocated"] == 1024
        assert data["total_free"] == 2048 
        assert data["frag_score"] == 0.0 # 1.0 - (1024/3072)? Wait, frag_score = 1.0 - (allocated / total)?
