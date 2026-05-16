"""
rtx_oom_guard.trainer.training_hook — PyTorch training-loop instrumentation.

Attaches to a toy (or real) PyTorch training loop and logs GPU memory
state before/after forward, backward, and optimizer step via an
``AllocatorLogger``.  Optionally evaluates an ``OOMRiskModel`` at each
checkpoint.

Usage::

    hook = TrainingHook()
    for batch in dataloader:
        hook.on_forward_begin()
        out = model(batch)
        hook.on_forward_end()

        hook.on_backward_begin()
        loss.backward()
        hook.on_backward_end()

        hook.on_optimizer_step()
        optimizer.step()
        hook.on_step_complete(batch_size=len(batch))
"""

import logging
import time
from contextlib import contextmanager
from typing import Optional, Generator

from rtx_oom_guard.profiler.allocator_logger import AllocatorLogger
from rtx_oom_guard.scheduler.risk_model import OOMRiskModel

log = logging.getLogger("rtx_oom_guard.training_hook")


class TrainingHook:
    """
    Lightweight PyTorch training-loop hook.

    Parameters
    ----------
    logger : AllocatorLogger, optional
        Reuse an existing logger (creates one if omitted).
    risk_model : OOMRiskModel, optional
        Evaluate OOM risk at each step boundary.
    """

    def __init__(
        self,
        logger: Optional[AllocatorLogger] = None,
        risk_model: Optional[OOMRiskModel] = None,
    ) -> None:
        self.logger = logger or AllocatorLogger()
        self.risk_model = risk_model
        self._step_start: float = 0.0
        self._last_risk: float = 0.0

    # -- Phase hooks -------------------------------------------------------

    def on_forward_begin(
        self,
        allocated_mb: Optional[float] = None,
        reserved_mb: Optional[float] = None,
    ) -> None:
        self._step_start = time.perf_counter()
        self.logger.snapshot(
            phase="forward_begin",
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
        )

    def on_forward_end(
        self,
        allocated_mb: Optional[float] = None,
        reserved_mb: Optional[float] = None,
    ) -> None:
        self.logger.snapshot(
            phase="forward_end",
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
        )

    def on_backward_begin(
        self,
        allocated_mb: Optional[float] = None,
        reserved_mb: Optional[float] = None,
    ) -> None:
        self.logger.snapshot(
            phase="backward_begin",
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
        )

    def on_backward_end(
        self,
        allocated_mb: Optional[float] = None,
        reserved_mb: Optional[float] = None,
    ) -> None:
        self.logger.snapshot(
            phase="backward_end",
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
        )

    def on_optimizer_step(
        self,
        allocated_mb: Optional[float] = None,
        reserved_mb: Optional[float] = None,
    ) -> None:
        self.logger.snapshot(
            phase="optimizer_step",
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
        )

    def on_step_complete(
        self,
        batch_size: int = 0,
        allocated_mb: Optional[float] = None,
        reserved_mb: Optional[float] = None,
    ) -> float:
        """
        Finalise the step: record elapsed time, optionally compute risk.

        Returns
        -------
        float
            OOM risk score (0 if no risk model attached).
        """
        elapsed = time.perf_counter() - self._step_start if self._step_start else 0.0
        rec = self.logger.snapshot(
            phase="step",
            step_time_s=elapsed,
            batch_size=batch_size,
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
        )
        # Manually increment step index if using raw snapshots
        # (AllocatorLogger.end_step normally does this, but we're calling snapshot directly)
        self.logger._step_idx += 1

        risk = 0.0
        if self.risk_model is not None:
            total_mb = self._total_gpu_mb()
            utilisation = rec.allocated_mb / total_mb if total_mb > 0 else 0.0
            risk = self.risk_model.score(
                fragmentation=rec.fragmentation_ratio,
                utilisation=utilisation,
                alloc_delta_mb=0.0,
            )
            self._last_risk = risk

        return risk

    # -- Context manager for ergonomic use ---------------------------------

    @contextmanager
    def wrap_step(self, batch_size: int = 0) -> Generator[None, None, None]:
        """
        Context manager wrapping an entire training step.

        Logs forward_begin at entry, step at exit.

        Usage::

            with hook.wrap_step(batch_size=32):
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
        """
        self.on_forward_begin()
        try:
            yield
        finally:
            self.on_step_complete(batch_size=batch_size)

    # -- Properties --------------------------------------------------------

    @property
    def last_risk(self) -> float:
        return self._last_risk

    @property
    def records(self):
        return self.logger.records

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _total_gpu_mb() -> float:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_mem / (1024 ** 2)  # pragma: no cover
        except ImportError:
            pass
        return 8192.0  # Default 8GB for simulation
