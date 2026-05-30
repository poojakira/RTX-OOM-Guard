"""
scripts/train_senior_predictor.py
==================================
Train the FragPredictor (Transformer) on the senior trace dataset.

Loads traces from data/traces/senior_v1/, creates sliding-window
(X, y) pairs, and trains with CosineAnnealingLR + gradient clipping.
Exports the trained checkpoint to checkpoints/.

Usage::

    python scripts/train_senior_predictor.py
    python scripts/train_senior_predictor.py --trace-dir data/traces/senior_v1 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from rtx_oom_guard.scheduler.predictor import FragPredictor
from rtx_oom_guard.scheduler.dataset import AllocationDataset
from rtx_oom_guard.utils import get_logger, DefragConfig

log = get_logger("senior-trainer")


def train(
    trace_dir: str = "data/traces/senior_v1",
    epochs: int = 20,
    batch_size: int = 64,
    seq_len: int = 64,
    lr: float = 3e-4,
    checkpoint_dir: str = "checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Dataset ──
    dataset = AllocationDataset(trace_dir=trace_dir, seq_len=seq_len)
    if len(dataset) == 0:
        print(f"ERROR: No valid traces in {trace_dir}. "
              f"Run 'python scripts/generate_senior_dataset.py' first.")
        sys.exit(1)

    n = len(dataset)
    train_n = int(n * 0.8)
    val_n = int(n * 0.1)
    test_n = n - train_n - val_n

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    log.info("Dataset: %d total, %d train, %d val, %d test",
             n, train_n, val_n, test_n)

    # ── Model ──
    config = DefragConfig()
    config.seq_len = seq_len
    config.input_dim = 4
    model = FragPredictor.from_config(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    log.info("FragPredictor: %d parameters", param_count)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    mae_fn = nn.L1Loss()

    # ── Training loop ──
    best_val_loss = float("inf")
    ckpt_dir = Path(ROOT / checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "frag_predictor_senior.pt"

    metrics = {"train_loss": [], "val_loss": [], "val_mae": [], "lr": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
                val_mae += mae_fn(pred, y).item()
        val_loss /= max(len(val_loader), 1)
        val_mae /= max(len(val_loader), 1)

        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_mae"].append(val_mae)
        metrics["lr"].append(cur_lr)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(str(ckpt_path))
            improved = " ← saved"

        log.info(
            "Epoch %2d/%d  train=%.6f  val=%.6f  MAE=%.6f  lr=%.2e%s",
            epoch, epochs, train_loss, val_loss, val_mae, cur_lr, improved,
        )

    elapsed = time.time() - t0

    # ── Test ──
    model = FragPredictor.load(str(ckpt_path), config, device=str(device))
    test_loss = 0.0
    test_mae = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += criterion(pred, y).item()
            test_mae += mae_fn(pred, y).item()
    test_loss /= max(len(test_loader), 1)
    test_mae /= max(len(test_loader), 1)

    metrics["test_loss"] = test_loss
    metrics["test_mae"] = test_mae
    metrics["total_time_s"] = round(elapsed, 2)
    metrics["dataset_size"] = n
    metrics["best_val_loss"] = best_val_loss

    # Save metrics
    metrics_path = ckpt_dir / "training_metrics_senior.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    import logging; logging.info(f"\n{'='*60}")
    import logging; logging.info(f"Training complete in {elapsed:.1f}s")
    import logging; logging.info(f"  Best val loss: {best_val_loss:.6f}")
    import logging; logging.info(f"  Test loss:     {test_loss:.6f}")
    import logging; logging.info(f"  Test MAE:      {test_mae:.6f}")
    import logging; logging.info(f"  Checkpoint:    {ckpt_path}")
    import logging; logging.info(f"  Metrics:       {metrics_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train FragPredictor on senior dataset")
    ap.add_argument("--trace-dir", default="data/traces/senior_v1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    args = ap.parse_args()

    train(
        trace_dir=args.trace_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )
