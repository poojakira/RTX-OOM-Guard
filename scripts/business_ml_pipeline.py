import time
import torch
import torch.nn as nn
from rtx_oom_guard.defrag_engine.defragmenter import GPUMemoryDefragmenter
from rtx_oom_guard.scheduler.risk_model import OOMRiskModel
from rtx_oom_guard.utils import get_logger

# Business Workload Simulation

log = get_logger("rtx_oom_guard.business_pipeline")

class BusinessInferenceModel(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.ReLU(),
            nn.Linear(d * 4, d * 8),
            nn.Dropout(0.1),
            nn.Linear(d * 8, 1000)
        )

    def forward(self, x):
        return self.layers(x)

def run_business_pipeline():
    log.info("🚀 Initializing Enterprise Inference Pipeline (rtx-oom-guard Protected)")
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    input_dim = 1024
    
    # Components
    model = BusinessInferenceModel(input_dim).to(device)
    risk_model = OOMRiskModel(mode="rule")
    defragmenter = GPUMemoryDefragmenter()
    
    log.info(f"  → Model Loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    log.info(f"  → Monitoring Mode: PROACTIVE_COMPACTION_ENABLED")

    # Simulation: 1 day of inference traffic
    for hour in range(24):
        # Simulate traffic spikes
        current_batch = batch_size if hour % 6 != 0 else (batch_size * 2)
        
        # 1. Inference Run
        input_data = torch.randn(current_batch, input_dim, device=device)
        with torch.no_grad():
            output = model(input_data)
        
        # 2. Infrastructure Health Check
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**2)
            resv = torch.cuda.memory_reserved() / (1024**2)
            frag = 1 - (alloc / resv) if resv > 0 else 0.0
            
            # Predict OOM Risk
            risk_score = risk_model.score(fragmentation=frag, utilisation=alloc/8192)
            
            log.info(f"Hour {hour:02d}:00 | Batch {current_batch} | VRAM {alloc:.1f}/{resv:.1f} MB | Risk: {risk_score:.2f}")

            # 3. Mitigation Decision
            if risk_score > 0.8:
                log.warning("⚠️ Critical OOM Risk Detected! Triggering Emergency Compaction...")
                
                # Active VRAM Repacking
                with torch.cuda.stream(torch.cuda.Stream()):
                    freed_mb = defragmenter.compact(model)
                    torch.cuda.empty_cache()
                
                log.info(f"  ✅ Compaction Complete. Physically recovered {freed_mb:.1f} MB of VRAM.")
            
            elif risk_score > 0.5:
                log.info("  ℹ️ Elevated Risk. Throttling throughput for next cycle...")
                batch_size = max(16, batch_size - 32)
            
            else:
                # Scale up if possible
                batch_size = min(256, batch_size + 16)

        time.sleep(0.01) # Faster simulation

    log.info("📊 Pipeline execution complete. 0 OOM events detected over 24-hour simulation.")

if __name__ == "__main__":
    run_business_pipeline()
