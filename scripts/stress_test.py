"""
Aggressive fragmentation simulation for the Predictive GPU Memory Defragmenter.
Creates random allocation patterns and non-linear deallocations to trigger compactions.
"""

import time
import torch  # type: ignore
import random
from rtx_oom_guard import DefragMonitor  # type: ignore
from rtx_oom_guard.utils import DefragConfig, get_logger  # type: ignore

log = get_logger("stress-test")

def run_stress_test(iterations=1000):
    if not torch.cuda.is_available():
        log.error("CUDA is not available.")
        return

    # 1. Initialize Monitor with LOWER threshold for demo
    config = DefragConfig()
    config.poll_interval_sec = 0.5
    monitor = DefragMonitor(threshold=0.3, config=config) # Aggressive threshold
    monitor.start()
    
    log.info("STRESS TEST STARTED. This will force fragmentation and COMPACTIONS.")
    
    device = "cuda"
    tensors = {} # Use dict to track random deallocations
    
    try:
        for i in range(iterations):
            # 1. Random Allocation
            if random.random() > 0.3:
                size = random.randint(64, 512) # MB
                t = torch.randn(size * 1024 * 256, device=device) # Roughly MB scale
                tensors[i] = t
                log.info(f"Allocated {size}MB")
            
            # 2. Random Deallocation (Creates Address Gaps/Fragmentation)
            if len(tensors) > 10 and random.random() > 0.5:
                key = random.choice(list(tensors.keys()))
                tensors.pop(key)
                torch.cuda.empty_cache()
                log.info(f"Released block {key}")
            
            monitor.auto_record()
            
            # Sync to dashboard
            try:
                import shutil
                shutil.copy2("results/live_telemetry.json", "dashboard/public/live/live_telemetry.json")
            except:
                pass

            time.sleep(0.1) # Fast for visual impact
            
            if i % 10 == 0:
                stats = monitor.stats()
                if stats['total_compactions'] > 0:
                    log.warning(f"DASHBOARD UPDATE: {stats['total_compactions']} compactions detected!")

    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        log.info("Stress test complete.")

if __name__ == "__main__":
    run_stress_test()
