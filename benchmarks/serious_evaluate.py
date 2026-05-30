"""
benchmark.serious_evaluate
==========================

Rigorous benchmarking script running 5 distinct trials to measure the impact of
rtx_oom_guard vs vanilla PyTorch on identical workloads.

Generates:
1. Console table
2. results/experiments_table.csv
3. results/benchmark_graphs.png
"""

import sys
import os
import csv
import statistics

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtx_oom_guard.utils import get_logger

log = get_logger("serious_evaluate")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def run_serious_benchmarks(n_trials=5, iterations_per_trial=100):
    from benchmark.run_baseline import run_benchmark
    from benchmark.run_with_defrag import run_benchmark_with_defrag
    import torch

    log.info(f"Starting Serious Evaluation: {n_trials} trials, {iterations_per_trial} iterations each.")

    results_baseline = []
    results_defrag = []

    for trial in range(n_trials):
        log.info(f"\n--- Trial {trial + 1}/{n_trials} ---")
        
        # 1. Baseline
        log.info("Running Baseline (Vanilla PyTorch)...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_res = run_benchmark(iterations_per_trial)
        results_baseline.append(base_res)
        
        # 2. Defrag
        log.info("Running rtx_oom_guard...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        defrag_res = run_benchmark_with_defrag(iterations_per_trial)
        results_defrag.append(defrag_res)

    # Compile Statistics
    def get_stats(results, key):
        vals = [r[key] for r in results]
        return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0

    b_oom_mean, b_oom_std = get_stats(results_baseline, "oom_errors")
    d_oom_mean, d_oom_std = get_stats(results_defrag, "oom_errors")

    b_time_mean, b_time_std = get_stats(results_baseline, "avg_iteration_time")
    d_time_mean, d_time_std = get_stats(results_defrag, "avg_iteration_time")

    b_mem_mean, b_mem_std = get_stats(results_baseline, "peak_memory_mb")
    d_mem_mean, d_mem_std = get_stats(results_defrag, "peak_memory_mb")

    # Output Table CSV
    os.makedirs("results", exist_ok=True)
    csv_file = "results/experiments_table.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Baseline (Mean ± Std)", "rtx_oom_guard (Mean ± Std)", "Improvement"])
        
        oom_imp = ((b_oom_mean - d_oom_mean) / max(b_oom_mean, 0.001)) * 100
        time_imp = ((b_time_mean - d_time_mean) / b_time_mean) * 100
        mem_imp = ((b_mem_mean - d_mem_mean) / b_mem_mean) * 100
        
        writer.writerow(["OOM Errors", f"{b_oom_mean:.1f} ± {b_oom_std:.1f}", f"{d_oom_mean:.1f} ± {d_oom_std:.1f}", f"{oom_imp:.1f}%"])
        writer.writerow(["Iteration Time (s)", f"{b_time_mean:.3f} ± {b_time_std:.3f}", f"{d_time_mean:.3f} ± {d_time_std:.3f}", f"{time_imp:.1f}%"])
        writer.writerow(["Peak Memory (MB)", f"{b_mem_mean:.0f} ± {b_mem_std:.0f}", f"{d_mem_mean:.0f} ± {d_mem_std:.0f}", f"{mem_imp:.1f}%"])

    log.info(f"\nSaved experimental results table to {csv_file}")

    # Console Print
    import logging; logging.info("=" * 60)
    import logging; logging.info("SERIOUS EVALUATION RESULTS (N=5 Trials)")
    import logging; logging.info("=" * 60)
    import logging; logging.info(f"{'Metric':<20} | {'Baseline':<15} | {'rtx_oom_guard':<15} | {'Improvement':<10}")
    import logging; logging.info("-" * 60)
    import logging; logging.info(f"{'OOM Errors':<20} | {b_oom_mean:<15.1f} | {d_oom_mean:<15.1f} | {oom_imp:.1f}%")
    import logging; logging.info(f"{'Iteration Time (s)':<20} | {b_time_mean:<15.3f} | {d_time_mean:<15.3f} | {time_imp:.1f}%")
    import logging; logging.info(f"{'Peak Memory (MB)':<20} | {b_mem_mean:<15.0f} | {d_mem_mean:<15.0f} | {mem_imp:.1f}%")
    import logging; logging.info("=" * 60)
    import logging; logging.info("WHY IMPROVEMENT HAPPENS:")
    import logging; logging.info("By tracking allocation patterns, the Transformer model predicts PyTorch memory gaps.")
    import logging; logging.info("When fragmentation reaches a critical threshold, it triggers a synchronous eviction and")
    import logging; logging.info("compaction mapping using the Triton engine BEFORE memory exhausts, avoiding OOM cascades.")

    # Graphing
    if MATPLOTLIB_AVAILABLE:
        labels = ['OOM Errors', 'Iteration Time (s)', 'Peak Memory (MB/1000)']
        
        b_means = [b_oom_mean, b_time_mean, b_mem_mean / 1000]
        d_means = [d_oom_mean, d_time_mean, d_mem_mean / 1000]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, b_means, width, label='Vanilla PyTorch', color='#ff7f0e')
        rects2 = ax.bar(x + width/2, d_means, width, label='rtx_oom_guard', color='#1f77b4')

        ax.set_ylabel('Value')
        ax.set_title(f'Performance Comparison (Average across {n_trials} trials)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        fig.tight_layout()
        graph_file = "results/benchmark_graphs.png"
        plt.savefig(graph_file, dpi=300)
        log.info(f"Saved benchmark graph to {graph_file}")

if __name__ == "__main__":
    run_serious_benchmarks(n_trials=5, iterations_per_trial=100)
