"""
rtx_oom_guard — Predictive GPU Memory Defragmenter
===================================================

A Transformer-driven proactive CUDA memory optimizer for PyTorch.

Quick Start::

    from rtx_oom_guard import auto_instrument

    # Wrap your model and optimizer with zero code changes
    model, optimizer = auto_instrument(model, optimizer)

    # ... standard training loop ...
"""

__version__ = "2.0.0"
__author__ = "GPU Defrag Infrastructure Team"

from rtx_oom_guard.scheduler.monitor import DefragMonitor
from rtx_oom_guard.trainer.callback import DefragCallback
from rtx_oom_guard.trainer.auto_instrument import auto_instrument
from rtx_oom_guard.trainer.ddp import DDPSyncManager
from rtx_oom_guard.profiler.collector import AllocationCollector
from rtx_oom_guard.predictor.model import FragPredictor
from rtx_oom_guard.defrag_engine.defragmenter import GPUMemoryDefragmenter

# Re-exported from migrated modules for unified namespace
from rtx_oom_guard.profiler.allocator_logger import AllocatorLogger
from rtx_oom_guard.scheduler.risk_model import OOMRiskModel
from rtx_oom_guard.trainer.training_hook import TrainingHook
from rtx_oom_guard.defrag_engine.policy import MitigationPolicy

__all__ = [
    "DefragMonitor",
    "DefragCallback",
    "auto_instrument",
    "DDPSyncManager",
    "AllocationCollector",
    "FragPredictor",
    "GPUMemoryDefragmenter",
    "AllocatorLogger",
    "OOMRiskModel",
    "TrainingHook",
    "MitigationPolicy",
]
