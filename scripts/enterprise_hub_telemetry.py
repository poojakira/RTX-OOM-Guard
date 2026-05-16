import json
import time
import math
import random
from pathlib import Path
from rtx_oom_guard.utils import get_logger

log = get_logger("enterprise-sim")

def run():
    target = Path("results")
    target.mkdir(exist_ok=True)
    live_file = target / "live_telemetry.json"
    
    log.info(f"Pumping high-grade synthetic enterprise telemetry to {live_file}...")
    log.info("This enables the AeroGrid Dashboard to visualize realistic DDP training loads.")
    
    total_compactions = 120
    total_freed_mb = 14500.0
    history = []
    
    # Pre-populate history with some sweeps
    for i in range(1, 11):
        history.append({
            "id": i,
            "elapsedMs": round(random.uniform(2.0, 6.5), 2),
            "recoveredMb": round(random.uniform(50.0, 300.0), 2)
        })
    
    tick = 0
    while True:
        tick += 1
        
        # Simulate PyTorch oscillating memory pressure (sine wave + noise)
        base_alloc = 18000
        base_resv = 24000
        oscillation = math.sin(tick / 10.0) * 3000
        noise = random.uniform(-500, 500)
        
        current_alloc = max(0, base_alloc + oscillation + noise)
        current_reserved = base_resv
        
        current_frag = 1.0 - (current_alloc / current_reserved)
        
        # Every 15 ticks (~7.5 seconds), simulate a compaction ray firing
        if tick % 15 == 0:
            sweep_id = total_compactions + 1
            ms = round(random.uniform(3.0, 14.5), 2)
            freed = round(random.uniform(128.0, 1024.0), 2)
            
            history.append({
                "id": sweep_id,
                "elapsedMs": ms,
                "recoveredMb": freed
            })
            if len(history) > 20:
                history.pop(0)
                
            total_compactions += 1
            total_freed_mb += freed
            current_frag = max(0.05, current_frag - 0.2) # fragmentation drops
            log.info(f"Sweep #{sweep_id} triggered! Recovered {freed}MB in {ms}ms")
            
        data = {
            "current_allocated_mb": round(current_alloc, 2),
            "current_reserved_mb": round(current_reserved, 2),
            "free_estimate_mb": round(current_reserved - current_alloc, 2),
            "current_frag": round(current_frag * 100, 2), # API uses 0.0-1.0, but UI handles its format
            "cuda_available": True,
            "total_compactions": total_compactions,
            "total_freed_mb": round(total_freed_mb, 2),
            "avg_latency_ms": round(sum(h["elapsedMs"] for h in history) / len(history), 2) if history else 0.0,
            "compaction_history": history
        }
        
        # Note: UI expects `fragPercent` out of API as current_frag if we pass it directly.
        # Wait, the API returns current_frag between 0.0 and 1.0. Let's trace it: 
        data["current_frag"] = round(current_frag * 100, 2)
        
        # Safe atomic write
        temp_file = target / "live_telemetry.tmp.json"
        with open(temp_file, "w") as f:
            json.dump(data, f)
        temp_file.replace(live_file)
        
        time.sleep(0.5)

if __name__ == "__main__":
    run()
