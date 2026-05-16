"""
rtx_oom_guard.callback — PyTorch training loop integration.

Provides a zero-config callback that hooks into any training loop to
enable automatic predictive defragmentation.

Usage::

    from rtx_oom_guard import DefragCallback

    callback = DefragCallback()
    callback.on_train_begin()

    for epoch in range(num_epochs):
        for batch in dataloader:
            callback.on_step_begin()
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            callback.on_step_end()

    callback.on_train_end()
"""

from typing import Optional
from rtx_oom_guard.scheduler.monitor import DefragMonitor
from rtx_oom_guard.trainer.ddp import DDPSyncManager
from rtx_oom_guard.utils import get_logger, DefragConfig

log = get_logger("callback")


class DefragCallback:
    """
    Training loop callback for automatic predictive defragmentation.

    Drop-in compatible with custom training loops and framework callbacks.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        model_path: Optional[str] = None,
        config: Optional[DefragConfig] = None,
    ):
        self.monitor = DefragMonitor(
            model_path=model_path,
            threshold=threshold,
            config=config,
        )
        self._step_count = 0
        self.ddp_manager = DDPSyncManager()

    def on_train_begin(self) -> None:
        """Called at the start of training."""
        self.monitor.start()
        log.info("DefragCallback activated.")

    def on_train_end(self) -> None:
        """Called at the end of training."""
        stats = self.monitor.stop().stats()
        log.info(
            "Training complete. %d steps, %d compactions, %.1f MB freed.",
            self._step_count,
            stats["total_compactions"],
            stats["total_freed_mb"],
        )

    def on_step_begin(self) -> None:
        """Called before each training step."""
        self.monitor.auto_record()

    def on_step_end(self) -> None:
        """Called after each training step."""
        self.monitor.auto_record()
        self._step_count += 1

        # Handle deferred compactions (DDP or Async)
        if hasattr(self.monitor, 'pending_compaction') and self.monitor.pending_compaction:
            should_compact = True

            # Sync across DDP ranks if enabled
            if self.monitor.config.ddp_sync:
                should_compact = self.ddp_manager.check_global_compaction(local_pending=True)

            if should_compact:
                score = self.monitor.last_predicted_score
                reason = (
                    f"ddp_async_frag={score:.3f}"
                    if self.monitor.config.ddp_sync
                    else f"async_frag={score:.3f}"
                )
                self.monitor.compactor.defragment_tensors(self.monitor._get_tracked_tensors(), reason=reason)

            self.monitor.pending_compaction = False

    def stats(self) -> dict:
        """Return monitor statistics and DDP network overhead."""
        base_stats = self.monitor.stats()
        base_stats["ddp_sync_overhead_ms"] = self.ddp_manager.get_avg_overhead()
        return base_stats
