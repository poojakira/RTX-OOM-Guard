"""
rtx_oom_guard.trainer — Training pipeline for the FragPredictor.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
from rtx_oom_guard.predictor.model import FragPredictor
from rtx_oom_guard.scheduler.dataset import create_dataloaders
from rtx_oom_guard.utils import get_logger, DefragConfig

log = get_logger("trainer")


def train(config: DefragConfig | None = None, verbose: bool = True) -> dict:
    """
    Train the FragPredictor on collected traces.

    Returns:
        Dict with training metrics (losses, MAE, etc.)
    """
    config = config or DefragConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(config)
    log.info("Data loaded: %d train, %d val, %d test batches", len(train_loader), len(val_loader), len(test_loader))

    # Model
    model = FragPredictor.from_config(config).to(device)
    log.info("Model: %d parameters", model.count_parameters())

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train_epochs)
    criterion = nn.MSELoss()
    mae_fn = nn.L1Loss()

    best_val_loss = float("inf")
    metrics = {"train_loss": [], "val_loss": [], "val_mae": [], "lr": []}

    for epoch in range(config.train_epochs):
        # ── Train ──
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
        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
                val_mae += mae_fn(pred, y).item()
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_mae"].append(val_mae)
        metrics["lr"].append(lr)

        if verbose:
            log.info(
                "Epoch %2d/%d — Train: %.6f | Val: %.6f | MAE: %.6f | LR: %.2e",
                epoch + 1, config.train_epochs, train_loss, val_loss, val_mae, lr,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(config.checkpoint_path)
            if verbose:
                log.info("  → Saved best model (val_loss=%.6f)", val_loss)

    # ── Test ──
    model = FragPredictor.load(config.checkpoint_path, config, device=str(device))
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

    log.info("Test Loss: %.6f | Test MAE: %.6f", test_loss, test_mae)

    # Save metrics
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{config.results_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
