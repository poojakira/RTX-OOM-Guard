"""
Benchmark: Baseline training WITHOUT defragmentation.
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
from rtx_oom_guard.utils import get_logger, ensure_cuda  # type: ignore

log = get_logger("benchmark.baseline")


def simulate_fragmentation():
    """Interleave large and small tensor allocations to fragment CUDA cache."""
    tensors = []
    for _ in range(50):
        tensors.append(torch.empty(1024 * 1024 * 10, device="cuda"))  # 10MB
        tensors.append(torch.empty(1024 * 1024 * 1, device="cuda"))   # 1MB

    # Free big blocks → create holes
    for i in range(0, len(tensors), 2):
        tensors[i] = None

    # Fill holes with medium blocks
    for _ in range(25):
        tensors.append(torch.empty(1024 * 1024 * 2, device="cuda"))  # 2MB

    return tensors


def run_benchmark(iterations: int = 100, batch_size: int = 8, seq_len: int = 512) -> dict:
    ensure_cuda()
    model = SimpleGPT2(n_layers=6).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    mlflow.start_run(run_name="baseline")
    mlflow.log_params({"iterations": iterations, "batch_size": batch_size, "seq_len": seq_len})

    stats = {
        "timestamp": datetime.now().isoformat(),
        "system": "baseline",
        "oom_errors": 0,
        "restarts": 0,
        "iteration_times": [],
        "peak_memory_mb": 0.0,
        "avg_memory_mb": 0.0,
        "memory_snapshots": [],
    }

    log.info("Baseline benchmark: %d iterations, batch=%d, seq=%d", iterations, batch_size, seq_len)

    memory_sum = 0.0
    for i in range(iterations):
        t0 = time.perf_counter()
        try:
            frag_tensors = simulate_fragmentation()

            inputs = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")
            targets = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 50257), targets.view(-1))
            loss.backward()
            optimizer.step()

            frag_tensors = None

            elapsed = time.perf_counter() - t0
            stats["iteration_times"].append(elapsed)  # type: ignore

            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            frag = 1.0 - (allocated / reserved) if reserved > 0 else 0.0

            stats["peak_memory_mb"] = max(stats["peak_memory_mb"], peak)  # type: ignore
            memory_sum += allocated

            if i % 10 == 0:
                stats["memory_snapshots"].append({  # type: ignore
                    "iteration": i, "allocated_mb": allocated,
                    "reserved_mb": reserved, "frag": frag,
                })
                mlflow.log_metric("allocated_mb", allocated, step=i)
                mlflow.log_metric("peak_memory_mb", peak, step=i)
                mlflow.log_metric("fragmentation_pct", frag * 100, step=i)
                mlflow.log_metric("iteration_time", elapsed, step=i)
                
                log.info("  Iter %3d/%d — %.2fs — Alloc: %.0fMB — Peak: %.0fMB — Frag: %.1f%%",
                         i, iterations, elapsed, allocated, peak, frag * 100)

        except torch.cuda.OutOfMemoryError:
            log.error("OOM at iteration %d", i)
            stats["oom_errors"] += 1  # type: ignore
            torch.cuda.empty_cache()

    stats["avg_iteration_time"] = sum(stats["iteration_times"]) / max(len(stats["iteration_times"]), 1)  # type: ignore
    stats["avg_memory_mb"] = memory_sum / max(iterations, 1)  # type: ignore

    os.makedirs("results", exist_ok=True)
    with open("results/baseline.json", "w") as f:
        json.dump(stats, f, indent=2)

    mlflow.log_metric("total_oom_errors", stats["oom_errors"])
    mlflow.log_metric("avg_iteration_time", stats["avg_iteration_time"])
    mlflow.log_metric("avg_memory_mb", stats["avg_memory_mb"])
    mlflow.log_artifact("results/baseline.json")
    mlflow.end_run()

    log.info("Baseline done. OOM: %d | Avg time: %.3fs | Peak: %.0fMB",
             stats["oom_errors"], stats["avg_iteration_time"], stats["peak_memory_mb"])
    return stats


if __name__ == "__main__":
    run_benchmark()
