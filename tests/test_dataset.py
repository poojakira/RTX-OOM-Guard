import pandas as pd
from rtx_oom_guard.scheduler.dataset import AllocationDataset

def test_allocation_dataset(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()

    df = pd.DataFrame({
        "action": [1.0] * 15,
        "delta_bytes": [1024] * 15,
        "time_delta_ms": [10] * 15,
        "fragmentation": [0.1] * 15
    })
    # Must be parquet
    df.to_parquet(trace_dir / "trace_1.parquet", index=False)

    # seq_len=2. We need at least seq_len + 10 = 12 events. We have 15.
    dataset = AllocationDataset(trace_dir=str(trace_dir), seq_len=2)

    assert len(dataset) == 12  # 15 - 2 - 1 = 12 windows

    x, y = dataset[0]
    assert x.shape == (2, 4)  # seq_len, 4 features
    assert y.shape == (1,)
    assert isinstance(y.item(), float)

