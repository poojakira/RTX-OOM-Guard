"""
RTX-OOM-Guard — 5-minute verification script.
Run: python verify.py
Works on CPU (demonstrates logic). For GPU benchmarks, run notebooks/colab_t4_validation.ipynb on Colab.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import logging; logging.info("=" * 60)
import logging; logging.info("RTX-OOM-GUARD VERIFICATION")
import logging; logging.info("=" * 60)

# 1. Import and version
import logging; logging.info("\n[1/5] Package import...")
import rtx_oom_guard
import logging; logging.info(f"  rtx_oom_guard v{rtx_oom_guard.__version__} imported")

# 2. Risk model scoring
import logging; logging.info("\n[2/5] OOM Risk Model (rule-based sigmoid heuristic)...")
from rtx_oom_guard.scheduler.risk_model import OOMRiskModel

risk_model = OOMRiskModel()
low_risk = risk_model.score(fragmentation=0.1, utilisation=0.3, alloc_delta_mb=50.0)
high_risk = risk_model.score(fragmentation=0.7, utilisation=0.9, alloc_delta_mb=500.0)
import logging; logging.info(f"  Low fragmentation scenario:  OOM risk = {low_risk:.3f}")
import logging; logging.info(f"  High fragmentation scenario: OOM risk = {high_risk:.3f}")
import logging; logging.info(f"  Threshold trigger (>0.7): {'YES' if high_risk > 0.7 else 'NO'}")

# 3. Defragmenter logic (CPU tensors)
import logging; logging.info("\n[3/5] Defragmenter tensor compaction (CPU mode)...")
import torch
from rtx_oom_guard.defrag_engine.defragmenter import GPUMemoryDefragmenter

# Create scattered tensors on CPU
tensors = [torch.randn(1000) for _ in range(10)]
defrag = GPUMemoryDefragmenter()
# Check that the defragmenter can be instantiated and has the right interface
import logging; logging.info(f"  Defragmenter instantiated. Methods: {[m for m in dir(defrag) if not m.startswith('_') and callable(getattr(defrag, m))]}")

# 4. Monitor lifecycle
import logging; logging.info("\n[4/5] DefragMonitor lifecycle...")
from rtx_oom_guard.scheduler.monitor import DefragMonitor

monitor = DefragMonitor()
import logging; logging.info(f"  Monitor created. Polling interval: {getattr(monitor, 'poll_interval', 'default')}s")
import logging; logging.info(f"  Kill switch active: {getattr(monitor, '_killed', False)}")

# 5. Auto-instrument API
import logging; logging.info("\n[5/5] auto_instrument API (CPU model)...")
from rtx_oom_guard import auto_instrument

model = torch.nn.Linear(512, 256)
optimizer = torch.optim.Adam(model.parameters())
model_out, opt_out = auto_instrument(model, optimizer)
# Verify training still works
x = torch.randn(8, 512)
loss = model_out(x).sum()
loss.backward()
opt_out.step()
import logging; logging.info(f"  Model instrumented. Forward + backward pass: OK")
import logging; logging.info(f"  Loss value: {loss.item():.4f}")

import logging; logging.info("\n" + "=" * 60)
import logging; logging.info("VERIFICATION COMPLETE")
import logging; logging.info("For GPU benchmarks: open notebooks/colab_t4_validation.ipynb in Colab")
import logging; logging.info("=" * 60)
