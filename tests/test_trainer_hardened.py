import pandas as pd
import numpy as np
from unittest.mock import patch
from rtx_oom_guard.trainer.trainer import train
from rtx_oom_guard.scheduler.dataset import AllocationDataset, create_dataloaders
from rtx_oom_guard.utils import DefragConfig

def test_dataset_empty(tmp_path):
    """Verify dataset handles empty trace directory."""
    ds = AllocationDataset(trace_dir=str(tmp_path), seq_len=64)
    assert len(ds) == 0

def test_dataset_short_trace(tmp_path):
    """Verify dataset skips traces that are too short."""
    df = pd.DataFrame({
        "action": [1, 0, 1],
        "delta_bytes": [1024, -512, 256],
        "timestamp_ns": [100, 200, 300]
    })
    path = tmp_path / "short.parquet"
    df.to_parquet(path)
    
    ds = AllocationDataset(trace_dir=str(tmp_path), seq_len=10)
    assert len(ds) == 0

def test_dataset_full_logic(tmp_path):
    """Verify dataset windowing and feature extraction logic."""
    seq_len = 5
    # Create a trace with 20 events
    df = pd.DataFrame({
        "action": np.random.randint(0, 2, 20),
        "delta_bytes": np.random.randint(-1000, 1000, 20),
        "time_delta_ms": np.random.uniform(0.1, 10.0, 20),
        "fragmentation": np.random.uniform(0, 1, 20)
    })
    path = tmp_path / "valid.parquet"
    df.to_parquet(path)
    
    ds = AllocationDataset(trace_dir=str(tmp_path), seq_len=seq_len)
    # len = 20 - 5 - 1 = 14
    assert len(ds) == 14
    x, y = ds[0]
    assert x.shape == (5, 4)
    assert y.shape == (1,)

def test_create_dataloaders_small_data(tmp_path):
    """Verify dataloader creation with very few samples."""
    df = pd.DataFrame({
        "action": [1]*10,
        "delta_bytes": [100]*10,
        "time_delta_ms": [1.0]*10,
        "fragmentation": [0.5]*10
    })
    df.to_parquet(tmp_path / "trace.parquet")
    
    config = DefragConfig(trace_dir=str(tmp_path), seq_len=5)
    train_dl, val_dl, test_dl = create_dataloaders(config)
    # n = 10 - 5 - 1 = 4. Splits might be dummy.
    assert len(train_dl) > 0

def test_trainer_loop_mocked(tmp_path):
    """Verify the training loop runs and saves checkpoints."""
    # Create a dummy dataset
    data_dir = tmp_path / "traces"
    data_dir.mkdir()
    df = pd.DataFrame({
        "action": [1]*30,
        "delta_bytes": [100]*30,
        "time_delta_ms": [1.0]*30,
        "fragmentation": [0.5]*30
    })
    df.to_parquet(data_dir / "trace.parquet")
    
    checkpoint = tmp_path / "best.pt"
    results = tmp_path / "results"
    
    config = DefragConfig(
        trace_dir=str(data_dir),
        checkpoint_path=str(checkpoint),
        results_dir=str(results),
        train_epochs=2,
        batch_size=2,
        seq_len=5
    )
    
    with patch("torch.cuda.is_available", return_value=False):
        metrics = train(config, verbose=False)
        assert "train_loss" in metrics
        assert len(metrics["train_loss"]) == 2
        assert checkpoint.exists()
        assert (results / "training_metrics.json").exists()
