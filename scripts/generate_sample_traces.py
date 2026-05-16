"""
scripts/generate_sample_traces.py
=================================
Generates sample CUDA allocation traces (Parquet) for the research prototype.
"""

import sys
import time
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from rtx_oom_guard.profiler.collector import AllocationCollector

def generate_traces():
    trace_dir = ROOT / "data" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating sample traces in {trace_dir}...")
    
    configs = [
        ("gpt2_trace.parquet", 256),
        ("resnet50_trace.parquet", 128),
        ("bert_trace.parquet", 192)
    ]
    
    for filename, n_events in configs:
        collector = AllocationCollector()
        
        current_alloc = 0
        current_res = 8192 * 1024 * 1024 # 8GB
        
        for i in range(n_events):
            is_alloc = (i % 2 == 0)
            # Random size between 1MB and 100MB
            size = np.random.randint(1, 100) * 1024 * 1024
            
            if is_alloc:
                current_alloc += size
            else:
                current_alloc = max(0, current_alloc - size)
            
            ts = time.time()
            event = {
                "timestamp_ns": int(ts * 1e9) + i,
                "delta_bytes": size if is_alloc else -size,
                "action": 1 if is_alloc else 0,
                "abs_allocated": current_alloc,
                "abs_reserved": current_res,
                "fragmentation": 1.0 - (current_alloc / current_res) if current_res > 0 else 0.0,
            }
            collector._events.append(event)
            
        collector.save(str(trace_dir / filename))
    
    print("Traces generated successfully.")

if __name__ == "__main__":
    generate_traces()
