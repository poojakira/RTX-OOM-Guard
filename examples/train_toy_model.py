"""
examples/train_toy_model.py
============================
Smallest end-to-end example showing AllocatorLogger + OOMRiskModel +
TrainingHook + MitigationPolicy working together.

Trains a 2-layer MLP on random data for 20 steps.
Runs on CPU or GPU.

Usage::

    python examples/train_toy_model.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from rtx_oom_guard.profiler.allocator_logger import AllocatorLogger
from rtx_oom_guard.scheduler.risk_model import OOMRiskModel
from rtx_oom_guard.trainer.training_hook import TrainingHook
from rtx_oom_guard.defrag_engine.policy import MitigationPolicy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # -- Toy model ---------------------------------------------------------
    model = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # -- Wire up rtx_oom_guard src modules -------------------------------------
    logger = AllocatorLogger()
    risk_model = OOMRiskModel(mode="rule")
    policy = MitigationPolicy(warn_threshold=0.5, act_threshold=0.8)
    hook = TrainingHook(logger=logger, risk_model=risk_model)

    batch_size = 32
    n_steps = 20

    print(f"\nTraining {n_steps} steps (batch_size={batch_size}) …\n")

    for step in range(n_steps):
        x = torch.randn(batch_size, 64, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)

        # Use the hook's context manager for an ergonomic demo
        with hook.wrap_step(batch_size=batch_size):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        risk = hook.last_risk
        action = policy.evaluate(risk, current_batch_size=batch_size)

        if step % 5 == 0:
            rec = logger.records[-1]
            print(
                f"  Step {step:3d} | loss={loss.item():.4f} | "
                f"alloc={rec.allocated_mb:.1f} MB | frag={rec.fragmentation_ratio:.4f} | "
                f"risk={risk:.4f} | tier={action.tier}"
            )

    # -- Summary -----------------------------------------------------------
    summary = logger.summary()
    counts = policy.action_counts
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total steps        : {summary.get('total_steps', 0)}")
    print(f"  Avg allocated (MB) : {summary.get('avg_allocated_mb', 0):.2f}")
    print(f"  Peak reserved (MB) : {summary.get('peak_reserved_mb', 0):.2f}")
    print(f"  Avg fragmentation  : {summary.get('avg_fragmentation', 0):.6f}")
    print(f"  Avg step time (s)  : {summary.get('avg_step_time_s', 0):.6f}")
    print(f"  Policy — SAFE={counts['SAFE']}  WARN={counts['WARN']}  ACT={counts['ACT']}")
    print("=" * 60)

    # Export
    logger.to_json("results/toy_model_log.json")
    print("\nLog saved → results/toy_model_log.json")


if __name__ == "__main__":
    main()
