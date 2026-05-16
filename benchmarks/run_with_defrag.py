"""
Benchmark: Training WITH predictive defragmentation enabled.
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import time
import json
import os
import sys
from datetime import datetime
import mlflow  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtx_oom_guard.trainer._models import SimpleGPT2  # type: ignore
from rtx_oom_guard.trainer.callback import DefragCallback  # type: ignore
from rtx_oom_guard.utils import get_logger, ensure_cuda  # type: ignore

log = get_logger("benchmark.defrag")


def simulate_fragmentation():
    """Same fragmentation pattern as baseline for fair comparison."""
    tensors = []
    for _ in range(50):
        tensors.append(torch.empty(1024 * 1024 * 10, device="cuda"))
        tensors.append(torch.empty(1024 * 1024 * 1, device="cuda"))
    for i in range(0, len(tensors), 2):
        tensors[i] = None
    for _ in range(25):
        tensors.append(torch.empty(1024 * 1024 * 2, device="cuda"))
    return tensors


def run_benchmark_with_defrag(iterations: int = 100, batch_size: int = 8, seq_len: int = 512) -> dict:
    ensure_cuda()
    model = SimpleGPT2(n_layers=6).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    mlflow.start_run(run_name="with_defrag")
    mlflow.log_params({"iterations": iterations, "batch_size": batch_size, "seq_len": seq_len})

    callback = DefragCallback(threshold=0.7)
    callback.monitor.config.max_prediction_latency_ms = 200  # Higher for CPU-inference benchmark
    callback.on_train_begin()

    oom_errors: int = 0
    iteration_times: list[float] = []
    peak_memory_mb = 0.0
    memory_snapshots = []

    log.info("Defrag benchmark: %d iterations, batch=%d, seq=%d", iterations, batch_size, seq_len)

    memory_sum: float = 0.0
    for i in range(iterations):
        t0 = time.perf_counter()
        try:
            callback.on_step_begin()

            frag_tensors = simulate_fragmentation()

            inputs = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
            targets = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 50257), targets.view(-1))
            loss.backward()
            optimizer.step()

            frag_tensors = None

            callback.on_step_end()

            elapsed = time.perf_counter() - t0
            iteration_times.append(elapsed)

            allocated = float(torch.cuda.memory_allocated()) / (1024**2)
            reserved = float(torch.cuda.memory_reserved()) / (1024**2)
            peak = float(torch.cuda.max_memory_allocated()) / (1024**2)
            frag = 1.0 - (allocated / reserved) if reserved > 0 else 0.0

            peak_memory_mb = max(peak_memory_mb, peak)
            memory_sum += allocated  # type: ignore

            if i % 10 == 0:
                memory_snapshots.append({
                    "iteration": i, "allocated_mb": allocated,
                    "reserved_mb": reserved, "frag": frag,
                })
                mlflow.log_metric("allocated_mb", allocated, step=i)
                mlflow.log_metric("peak_memory_mb", peak, step=i)
                mlflow.log_metric("fragmentation_pct", frag * 100, step=i)
                mlflow.log_metric("iteration_time", elapsed, step=i)

                log.info("  Iter %3d/%d — %.2fs — Alloc: %.0fMB — Frag: %.1f%%",
                         i, iterations, elapsed, allocated, frag * 100)

        except torch.cuda.OutOfMemoryError:
            log.error("OOM at iteration %d", i)
            oom_errors += 1  # type: ignore
            torch.cuda.empty_cache()

    callback.on_train_end()

    stats = {
        "timestamp": datetime.now().isoformat(),
        "system": "rtx_oom_guard",
        "oom_errors": oom_errors,
        "restarts": 0,
        "iteration_times": iteration_times,
        "peak_memory_mb": peak_memory_mb,
        "avg_memory_mb": memory_sum / max(iterations, 1),  # type: ignore
        "memory_snapshots": memory_snapshots,
        "avg_iteration_time": sum(iteration_times) / max(len(iteration_times), 1),
        "monitor_stats": callback.stats()
    }

    os.makedirs("results", exist_ok=True)
    with open("results/defrag.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    mlflow.log_metric("total_oom_errors", oom_errors)
    mlflow.log_metric("avg_iteration_time", stats["avg_iteration_time"])
    mlflow.log_metric("avg_memory_mb", stats["avg_memory_mb"])
    mlflow.log_metric("total_compactions", callback.stats()["total_compactions"])
    mlflow.log_artifact("results/defrag.json")
    mlflow.end_run()

    log.info("Defrag done. OOM: %d | Avg time: %.3fs | Peak: %.0fMB | Compactions: %d",
             oom_errors, stats["avg_iteration_time"], peak_memory_mb,
             callback.stats()["total_compactions"])
    return stats


if __name__ == "__main__":
    run_benchmark_with_defrag()
