"""
benchmark/plot_results.py
=========================

Generates before vs after charts based on results/comparison.json
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_before_after():
    results_path = Path("results/comparison.json")
    if not results_path.exists():
        import logging; logging.info(f"Error: {results_path} not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    metrics = {
        "Peak Memory (MB)": (data["baseline"]["peak_memory_mb"], data["rtx_oom_guard"]["peak_memory_mb"]),
        "Training Time per Iter (s)": (data["baseline"]["avg_time"], data["rtx_oom_guard"]["avg_time"]),
        "Throughput (iters/s)": (
            data["baseline"].get("throughput_iters_per_sec", 0),
            data["rtx_oom_guard"].get("throughput_iters_per_sec", 0)
        )
    }

    labels = list(metrics.keys())
    baseline_vals = [m[0] for m in metrics.values()]
    defrag_vals = [m[1] for m in metrics.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline PyTorch', color='#ff7f0e')
    rects2 = ax.bar(x + width/2, defrag_vals, width, label='rtx_oom_guard', color='#1f77b4')

    ax.set_ylabel('Scores')
    ax.set_title('rtx-oom-guard: Baseline vs Proactive Defragmentation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add numeric labels atop bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    
    out_path = Path("results/benchmark_charts.png")
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, dpi=300)
    import logging; logging.info(f"Saved chart to {out_path}")

if __name__ == "__main__":
    plot_before_after()
