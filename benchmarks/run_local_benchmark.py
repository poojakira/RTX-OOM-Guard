"""
benchmarks/run_local_benchmark.py
==================================
Run the local RTX-class experiment described in the README.

Executes 5 independent training runs of a GPT-2-style model with
synthetic fragmentation pressure.  Collects per-run JSON, aggregated
CSV, and a fragmentation-vs-time plot.

Works on CPU (simulated metrics) when no GPU is present so that
result artefacts can be regenerated in CI.

Usage::

    python benchmarks/run_local_benchmark.py
    python benchmarks/run_local_benchmark.py --runs 3 --steps 50
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from rtx_oom_guard.profiler.allocator_logger import AllocatorLogger
from rtx_oom_guard.scheduler.risk_model import OOMRiskModel
from rtx_oom_guard.trainer.training_hook import TrainingHook
from rtx_oom_guard.defrag_engine.policy import MitigationPolicy

# Detect GPU

try:
    import torch
    import torch.nn as nn
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

GPU_NAME = "NVIDIA GeForce RTX 4060" if not HAS_CUDA else (
    torch.cuda.get_device_name(0) if HAS_CUDA else "CPU-simulated"
)


# Tiny GPT-2 for the benchmark (reused pattern from rtx_oom_guard._models)

if HAS_CUDA:
    class _BenchModel(nn.Module):
        def __init__(self, vocab=50257, d=768, layers=6, heads=12):
            super().__init__()
            self.tok = nn.Embedding(vocab, d)
            self.pos = nn.Embedding(1024, d)
            self.enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.0,
                                           batch_first=True, norm_first=True),
                num_layers=layers,
            )
            self.head = nn.Linear(d, vocab, bias=False)

        def forward(self, x):
            B, T = x.shape
            h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
            return self.head(self.enc(h))


# Fragmentation helper

def _fragment_gpu(n_chunks: int = 40, chunk_mb: int = 8):
    """Punch holes in the CUDA allocator pool."""
    if not HAS_CUDA:
        return []
    pairs = []
    for _ in range(n_chunks):
        big = torch.empty(chunk_mb * 1024 * 256, device="cuda")
        small = torch.empty(64 * 1024, device="cuda")
        pairs.append((big, small))
    survivors = []
    for big, small in pairs:
        del big
        survivors.append(small)
    return survivors


# Single run (GPU)

def _run_gpu(run_id: int, steps: int, batch_size: int, seq_len: int):
    """Execute one benchmark run on a real GPU."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = _BenchModel(layers=6).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    logger = AllocatorLogger()
    risk_model = OOMRiskModel()
    policy = MitigationPolicy()
    hook = TrainingHook(logger=logger, risk_model=risk_model)

    oom_count = 0
    t_start = time.perf_counter()

    for step in range(steps):
        anchors = None
        try:
            anchors = _fragment_gpu(40, 8)
            hook.on_forward_begin()
            x = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
            y = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            hook.on_forward_end()

            hook.on_backward_begin()
            loss = loss_fn(logits.view(-1, 50257), y.view(-1))
            loss.backward()
            hook.on_backward_end()

            hook.on_optimizer_step()
            opt.step()
            risk = hook.on_step_complete(batch_size=batch_size)
            policy.evaluate(risk, current_batch_size=batch_size)

        except (torch.cuda.OutOfMemoryError if HAS_CUDA else RuntimeError):
            oom_count += 1
            torch.cuda.empty_cache()
        finally:
            if anchors:
                del anchors

    elapsed = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_reserved() / (1024 ** 2)
    del model, opt
    torch.cuda.empty_cache()

    summary = logger.summary()
    return {
        "run_id": run_id,
        "gpu": GPU_NAME,
        "steps": steps,
        "batch_size": batch_size,
        "oom_count": oom_count,
        "peak_reserved_mb": round(peak, 1),
        "avg_fragmentation": summary.get("avg_fragmentation", 0.0),
        "avg_step_time_s": summary.get("avg_step_time_s", 0.0),
        "throughput_iter_s": round(steps / elapsed, 3) if elapsed > 0 else 0,
        "total_time_s": round(elapsed, 3),
        "policy_actions": policy.action_counts,
        "memory_log": logger.to_dicts(),
    }


# CPU simulation removed — was generating fake metrics with random numbers.
# Use notebooks/colab_t4_validation.ipynb for real GPU benchmarks.


# Main

def main():
    ap = argparse.ArgumentParser(description="Local RTX benchmark for rtx_oom_guard")
    ap.add_argument("--runs", type=int, default=5, help="Number of independent runs")
    ap.add_argument("--steps", type=int, default=100, help="Training steps per run")
    ap.add_argument("--batch-size", type=int, default=6, help="Batch size")
    ap.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    ap.add_argument("--results-dir", default="results", help="Output directory")
    args = ap.parse_args()

    results_dir = Path(ROOT / args.results_dir)
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    run_fn = _run_gpu if HAS_CUDA else None
    mode_label = "GPU" if HAS_CUDA else "NO GPU"

    if run_fn is None:
        import logging; logging.info("ERROR: No CUDA GPU available. This benchmark requires a real GPU.")
        import logging; logging.info("Run notebooks/colab_t4_validation.ipynb on a free Colab T4 instead.")
        sys.exit(1)

    import logging; logging.info(f"Running {args.runs} benchmark runs ({mode_label}) …")

    all_results = []
    for i in range(1, args.runs + 1):
        import logging; logging.info(f"  Run {i}/{args.runs} …", end=" ", flush=True)
        result = run_fn(i, args.steps, args.batch_size, args.seq_len)
        all_results.append(result)

        # Save per-run JSON
        run_path = results_dir / f"run_{i}.json"
        with open(run_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✓  OOM={result['oom_count']}  peak={result['peak_reserved_mb']} MB  "
              f"frag={result['avg_fragmentation']:.4f}  throughput={result['throughput_iter_s']} it/s")

    # -- Summary CSV -------------------------------------------------------
    csv_path = results_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run", "oom_count", "throughput_iter_s", "peak_reserved_mb",
            "avg_fragmentation", "avg_step_time_s",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "run": r["run_id"],
                "oom_count": r["oom_count"],
                "throughput_iter_s": r["throughput_iter_s"],
                "peak_reserved_mb": r["peak_reserved_mb"],
                "avg_fragmentation": r["avg_fragmentation"],
                "avg_step_time_s": r["avg_step_time_s"],
            })
    import logging; logging.info(f"\n  Summary CSV → {csv_path}")

    # -- Fragmentation-vs-time plot ----------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        for r in all_results:
            step_records = [e for e in r["memory_log"] if e["phase"] == "step"]
            xs = [e["step"] for e in step_records]
            ys = [e["fragmentation_ratio"] for e in step_records]
            ax.plot(xs, ys, alpha=0.7, label=f"Run {r['run_id']}")

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Fragmentation Ratio")
        ax.set_title("Fragmentation vs. Training Step (5 Runs)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = plots_dir / "fragmentation_vs_time.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        import logging; logging.info(f"  Plot → {plot_path}")
    except ImportError:
        import logging; logging.info("  (matplotlib not available — skipping plot)")

    import logging; logging.info("\nBenchmark complete.")


if __name__ == "__main__":
    main()
