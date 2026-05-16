"""
benchmark/evaluate.py
=====================

Comprehensive evaluation suite for rtx_oom_guard.
Runs the baseline vs defrag comparison, measures throughput, memory, 
and generates visualizations of the results.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtx_oom_guard.utils import get_logger

log = get_logger("evaluate")

def run_evaluation():
    log.info("Starting comprehensive evaluation suite...")
    
    # 1. Run the comparison script
    compare_script = Path(__file__).parent / "compare.py"
    log.info(f"Executing {compare_script} ...")
    
    try:
        subprocess.run([sys.executable, str(compare_script)], check=True)
    except subprocess.CalledProcessError:
        log.error("Comparison script failed.")
        sys.exit(1)
        
    # 2. Check if results exist
    results_file = Path("results/comparison.json")
    if not results_file.exists():
        log.error(f"Results file {results_file} not found after benchmark.")
        sys.exit(1)
        
    # 3. Read results and print throughput
    with open(results_file, "r") as f:
        data = json.load(f)
        
    baseline_time = data["baseline"]["avg_time"]
    defrag_time = data["rtx_oom_guard"]["avg_time"]
    
    # Compute Throughput (iterations / sec)
    baseline_throughput = 1.0 / baseline_time if baseline_time > 0 else 0
    defrag_throughput = 1.0 / defrag_time if defrag_time > 0 else 0
    
    # Add throughput to data and save
    data["baseline"]["throughput_iters_per_sec"] = baseline_throughput
    data["rtx_oom_guard"]["throughput_iters_per_sec"] = defrag_throughput
    data["improvement"]["throughput_change_pct"] = (
        ((defrag_throughput - baseline_throughput) / max(baseline_throughput, 1e-9)) * 100
    )
    
    with open(results_file, "w") as f:
        json.dump(data, f, indent=2)
        
    log.info("═" * 60)
    log.info("THROUGHPUT RESULTS")
    log.info("═" * 60)
    log.info(f"Baseline:  {baseline_throughput:.2f} iters/sec")
    log.info(f"rtx_oom_guard: {defrag_throughput:.2f} iters/sec")
    log.info(f"Delta:     {data['improvement']['throughput_change_pct']:.1f}%")
    log.info("═" * 60)
    
    # 4. Generate plots
    plot_script = Path(__file__).parent / "plot_results.py"
    log.info(f"Generating plots using {plot_script} ...")
    try:
        subprocess.run([sys.executable, str(plot_script)], check=True)
    except subprocess.CalledProcessError:
        log.error("Plotting script failed.")
        sys.exit(1)
        
    log.info("Evaluation complete. Check `results/` for charts and data.")

if __name__ == "__main__":
    run_evaluation()
