"""
Real-world demo script for the Predictive GPU Memory Defragmenter.
This script runs a synthetic training loop that creates memory pressure,
allowing the user to see the dashboard respond in real-time.
"""

import time
import torch # type: ignore
from rtx_oom_guard import DefragMonitor # type: ignore
from rtx_oom_guard.utils import DefragConfig, get_logger # type: ignore

log = get_logger("live-demo")

def run_simulation(iterations=500):
    ensure_cuda()
    
    # 1. Initialize Monitor
    config = DefragConfig()
    config.max_prediction_latency_ms = 500.0 # Be more lenient for the demo
    monitor = DefragMonitor(threshold=0.6, config=config)
    monitor.start()
    
    log.info("Simulation started. Open dashboard at http://localhost:5173 to watch!")
    
    # Dashboard sync path
    public_path = "dashboard/public/live/live_telemetry.json"
    
    # 2. Setup a dummy workload that creates memory pressure
    device = "cuda"
    tensors = []
    
    try:
        for i in range(iterations):
            # 1. Simulation step: Big allocation every 10 iterations
            if i % 10 == 0:
                t = torch.randn(1024, 1024, 64, device=device) # ~256MB
                tensors.append(t)
                log.info(f"Iteration {i:03d} | Memory allocated: 256MB | Total blocks: {len(tensors)}")
            
            # 2. Memory threshold management
            if len(tensors) > 15:
                tensors.pop(0)
                torch.cuda.empty_cache()
            
            # 3. Synchronize monitor and export telemetry
            monitor.auto_record()
            
            # Atomic sync to dashboard public folder
            try:
                import shutil
                shutil.copy2("results/live_telemetry.json", public_path)
            except Exception as e:
                log.debug(f"Dashboard sync skipped: {e}")

            time.sleep(0.5) 
            
            if i % 50 == 0:
                stats = monitor.stats()
                log.info(f">> Status: {stats['total_compactions']} proactive compactions, {stats['total_freed_mb']:.0f}MB salvaged.")

    except KeyboardInterrupt:
        log.info("Simulation interrupted by user.")
    finally:
        monitor.stop()
        log.info("Simulation complete.")

def ensure_cuda():
    if not torch.cuda.is_available():
        log.error("CUDA is not available. This demo requires a GPU.")
        exit(1)

if __name__ == "__main__":
    run_simulation()
