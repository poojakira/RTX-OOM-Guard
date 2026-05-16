import pytest
import pandas as pd
from rtx_oom_guard.scheduler.dataset import AllocationDataset, create_dataloaders
from rtx_oom_guard.utils import DefragConfig

def test_dataset_skipping_logic(tmp_path):
    """Verify dataset skips corrupt or short files (Line 53, 59)."""
    # 1. Missing columns (Line 53)
    df_bad = pd.DataFrame({"dummy": [1, 2, 3]})
    df_bad.to_parquet(tmp_path / "bad.parquet")
    
    # 2. Too short (Line 59)
    df_short = pd.DataFrame({"action": [1]*5, "delta_bytes": [1024]*5})
    df_short.to_parquet(tmp_path / "short.parquet")
    
    ds = AllocationDataset(trace_dir=str(tmp_path), seq_len=10)
    assert len(ds) == 0

def test_dataset_feature_proxy_and_ns(tmp_path):
    """Verify feature calculation with timestamp_ns and proxy fragmentation (Line 70, 76-78)."""
    # Create trace with timestamp_ns and NO fragmentation column
    data = {
        "action": [1] * 30,
        "delta_bytes": [1024 * i for i in range(30)],
        "timestamp_ns": [1000 * i for i in range(30)]
    }
    df = pd.DataFrame(data)
    df.to_parquet(tmp_path / "proxy.parquet")
    
    ds = AllocationDataset(trace_dir=str(tmp_path), seq_len=10)
    # len(df) = 30. seq_len = 10. loop range(30 - 10 - 1) = range(19) -> 19 windows
    assert len(ds) == 19
    # Check features[1, 2] (time_delta_ms) from timestamp_ns delta
    # np.diff([0, 1000], prepend=0) = [0, 1000] / 1e6 = 0.0, 0.001
    assert ds.windows[0][1, 2] == pytest.approx(0.001, abs=1e-6)

def test_dataloaders_small_data(tmp_path):
    """Verify handling of extremely small datasets (Line 113-114)."""
    # Create 15 events for seq_len=10 -> forms 15 - 10 - 1 = 4 windows
    df = pd.DataFrame({"action": [1]*15, "delta_bytes": [1024]*15, "fragmentation": [0.5]*15})
    df.to_parquet(tmp_path / "small.parquet")
    
    config = DefragConfig(trace_dir=str(tmp_path), seq_len=10)
    train, val, test = create_dataloaders(config)
    # n=4, so it hits the train/val/test split logic normally if n >= 3
    # Wait, create_dataloaders checks n < 3 for dummy splits (Line 112)
    assert len(train.dataset) > 0
    
    # Force n < 3
    # n = len(df) - seq_len - 1. To get n=2, we need len(df) = 2 + 10 + 1 = 13
    df_vsmall = pd.DataFrame({"action": [1]*13, "delta_bytes": [1024]*13, "fragmentation": [0.5]*13})
    vsmall_path = tmp_path / "vsmall"
    vsmall_path.mkdir()
    df_vsmall.to_parquet(vsmall_path / "vsmall.parquet")
    config_v = DefragConfig(trace_dir=str(vsmall_path), seq_len=10)
    train_v, val_v, test_v = create_dataloaders(config_v)
    assert len(train_v.dataset) == 2
    assert len(val_v.dataset) == 2
