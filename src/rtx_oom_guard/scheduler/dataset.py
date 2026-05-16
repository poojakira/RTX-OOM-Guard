"""
rtx_oom_guard.dataset — Trace-to-tensor dataset pipeline.

Converts raw Parquet allocation traces into sliding-window (X, y) pairs
for training the FragPredictor. Handles normalization, windowing, and
train/val/test splitting.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from typing import Tuple, Optional
from rtx_oom_guard.utils import get_logger, DefragConfig

log = get_logger("dataset")


class AllocationDataset(Dataset):
    """
    Sliding-window dataset over allocation traces.

    Features per event (input_dim=4):
        0: action (1=alloc, 0=free)
        1: size_gb (absolute delta in GB, always positive)
        2: time_delta_ms (time since previous event)
        3: fragmentation (1 - allocated/reserved)

    Label: fragmentation score at window+1
    """

    def __init__(self, trace_dir: str = "data/traces", seq_len: int = 64):
        self.seq_len = seq_len
        self.windows: list = []
        self.labels: list = []

        files = sorted(glob.glob(str(Path(trace_dir) / "*.parquet")))
        if not files:
            log.warning("No trace files found in %s", trace_dir)
            return

        for fpath in files:
            self._process_file(fpath)

        log.info("Dataset: %d windows from %d files", len(self.windows), len(files))

    def _process_file(self, path: str) -> None:
        df = pd.read_parquet(path)
        required = {"action", "delta_bytes"}
        if not required.issubset(df.columns):
            log.warning("Skipping %s — missing columns: %s", path, required - set(df.columns))
            return

        # Need at least seq_len + 1 to form a window/label pair
        # Industry standard: suggest at least 1.5x seq_len for stability
        if len(df) < int(self.seq_len * 1.2):
            log.warning("Skipping %s — trace too short (%d events, seq_len=%d)", path, len(df), self.seq_len)
            return

        # Build feature matrix
        features = np.zeros((len(df), 4), dtype=np.float32)
        features[:, 0] = df["action"].values.astype(np.float32)
        features[:, 1] = np.abs(df["delta_bytes"].values.astype(np.float64)) / (1024**3)

        if "time_delta_ms" in df.columns:
            features[:, 2] = df["time_delta_ms"].values.astype(np.float32)
        elif "timestamp_ns" in df.columns:
            features[:, 2] = np.diff(df["timestamp_ns"].values, prepend=df["timestamp_ns"].values[0]) / 1e6

        if "fragmentation" in df.columns:
            features[:, 3] = df["fragmentation"].values.astype(np.float32)
        else:
            # Compute rolling fragmentation proxy from size variance
            sizes = np.abs(df["delta_bytes"].values.astype(np.float64))
            rolling_std = pd.Series(sizes).rolling(window=20, min_periods=1).std().fillna(0).values
            features[:, 3] = rolling_std / (rolling_std.max() + 1e-8)

        # Compute labels: fragmentation at next position
        labels = features[:, 3]

        # Slide windows
        for i in range(len(features) - self.seq_len - 1):
            self.windows.append(features[i : i + self.seq_len])
            self.labels.append(labels[i + self.seq_len])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.windows[idx]),
            torch.tensor([self.labels[idx]], dtype=torch.float32),
        )


def create_dataloaders(
    config: Optional[DefragConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders from trace files."""
    config = config or DefragConfig()
    dataset = AllocationDataset(trace_dir=config.trace_dir, seq_len=config.seq_len)

    if len(dataset) == 0:
        raise RuntimeError(
            f"Empty dataset — no valid trace files found in {config.trace_dir}. "
            "Run `rtx_oom_guard-collect` first."
        )

    n = len(dataset)
    if n < 3: # Need at least 1 per split
        log.warning("Extremely small dataset (%d samples); using dummy splits", n)
        return (
            DataLoader(dataset, batch_size=min(n, config.batch_size), shuffle=True),
            DataLoader(dataset, batch_size=min(n, config.batch_size), shuffle=False),
            DataLoader(dataset, batch_size=min(n, config.batch_size), shuffle=False),
        )

    train_n = int(n * 0.8)
    val_n = int(n * 0.1)
    test_n = n - train_n - val_n

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_n, val_n, test_n], generator=torch.Generator().manual_seed(42)
    )

    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True),
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True),
        DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True),
    )
