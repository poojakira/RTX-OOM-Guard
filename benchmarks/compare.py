"""
Benchmark: Compare baseline vs defrag and generate reports.
"""

import json
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtx_oom_guard.utils import get_logger

log = get_logger("benchmark.compare")


def run_comparison(iterations: int = 100):
    """Run both benchmarks and produce comparison reports."""
    from benchmarks.run_baseline import run_benchmark
    from benchmarks.run_with_defrag import run_benchmark_with_defrag

    log.info("═" * 60)
    log.info("PHASE 1: Running BASELINE (no defrag)")
    log.info("═" * 60)
    baseline = run_benchmark(iterations)

    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    log.info("")
    log.info("═" * 60)
    log.info("PHASE 2: Running WITH rtx_oom_guard")
    log.info("═" * 60)
    defrag = run_benchmark_with_defrag(iterations)

    # ── Generate comparison ──
    comparison = {
        "iterations": iterations,
        "baseline": {
            "oom_errors": baseline["oom_errors"],
            "avg_time": baseline["avg_iteration_time"],
            "peak_memory_mb": baseline["peak_memory_mb"],
        },
        "rtx_oom_guard": {
            "oom_errors": defrag["oom_errors"],
            "avg_time": defrag["avg_iteration_time"],
            "peak_memory_mb": defrag["peak_memory_mb"],
            "compactions": defrag.get("monitor_stats", {}).get("total_compactions", 0),
        },
        "improvement": {
            "oom_reduction_pct": (
                ((baseline["oom_errors"] - defrag["oom_errors"]) / max(baseline["oom_errors"], 1)) * 100
                if baseline["oom_errors"] > 0 else 0.0
            ),
            "time_change_pct": (
                ((baseline["avg_iteration_time"] - defrag["avg_iteration_time"]) / baseline["avg_iteration_time"]) * 100
            ),
            "memory_change_pct": (
                ((baseline["peak_memory_mb"] - defrag["peak_memory_mb"]) / baseline["peak_memory_mb"]) * 100
            ),
        },
    }

    os.makedirs("results", exist_ok=True)

    with open("results/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # CSV
    with open("results/comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Baseline", "rtx_oom_guard", "Change (%)"])
        w.writerow(["OOM Errors", baseline["oom_errors"], defrag["oom_errors"],
                     f"{comparison['improvement']['oom_reduction_pct']:.1f}%"])
        w.writerow(["Avg Iteration Time (s)", f"{baseline['avg_iteration_time']:.4f}",
                     f"{defrag['avg_iteration_time']:.4f}",
                     f"{comparison['improvement']['time_change_pct']:.1f}%"])
        w.writerow(["Peak Memory (MB)", f"{baseline['peak_memory_mb']:.0f}",
                     f"{defrag['peak_memory_mb']:.0f}",
                     f"{comparison['improvement']['memory_change_pct']:.1f}%"])

    log.info("")
    log.info("═" * 60)
    log.info("RESULTS")
    log.info("═" * 60)
    log.info("%-25s %-12s %-12s %-10s", "Metric", "Baseline", "rtx_oom_guard", "Δ")
    log.info("-" * 60)
    log.info("%-25s %-12d %-12d %-10s", "OOM Errors",
             baseline["oom_errors"], defrag["oom_errors"],
             f"{comparison['improvement']['oom_reduction_pct']:.1f}%")
    log.info("%-25s %-12.4f %-12.4f %-10s", "Avg Iter Time (s)",
             baseline["avg_iteration_time"], defrag["avg_iteration_time"],
             f"{comparison['improvement']['time_change_pct']:.1f}%")
    log.info("%-25s %-12.0f %-12.0f %-10s", "Peak Memory (MB)",
             baseline["peak_memory_mb"], defrag["peak_memory_mb"],
             f"{comparison['improvement']['memory_change_pct']:.1f}%")

    log.info("\nReports saved: results/comparison.json, results/comparison.csv")
    return comparison


if __name__ == "__main__":
    run_comparison()
