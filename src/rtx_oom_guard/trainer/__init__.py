"""rtx_oom_guard.trainer — Training pipeline, auto-instrumentation, and DDP support."""

from rtx_oom_guard.trainer.callback import DefragCallback
from rtx_oom_guard.trainer.auto_instrument import auto_instrument
from rtx_oom_guard.trainer.ddp import DDPSyncManager
from rtx_oom_guard.trainer.training_hook import TrainingHook

__all__ = ["DefragCallback", "auto_instrument", "DDPSyncManager", "TrainingHook"]
