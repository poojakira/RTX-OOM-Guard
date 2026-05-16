"""
tests/test_hook.py — Prove the TrainingHook actually runs.
"""

import torch
import torch.nn as nn

from rtx_oom_guard.trainer.training_hook import TrainingHook
from rtx_oom_guard.profiler.allocator_logger import AllocatorLogger
from rtx_oom_guard.scheduler.risk_model import OOMRiskModel


class TestTrainingHook:
    """Training-hook tests — all run on CPU."""

    def test_hook_lifecycle(self):
        """forward/backward/optimizer events all produce log entries."""
        hook = TrainingHook()

        hook.on_forward_begin()
        hook.on_forward_end()
        hook.on_backward_begin()
        hook.on_backward_end()
        hook.on_optimizer_step()
        hook.on_step_complete(batch_size=16)

        phases = [r.phase for r in hook.records]
        assert "forward_begin" in phases
        assert "forward_end" in phases
        assert "backward_begin" in phases
        assert "backward_end" in phases
        assert "optimizer_step" in phases
        assert "step" in phases

    def test_wrap_step(self):
        """Context manager produces correct number of log entries."""
        hook = TrainingHook()

        with hook.wrap_step(batch_size=8):
            pass  # simulate a training step

        # wrap_step logs: forward_begin at entry, step at exit = 2 entries
        assert len(hook.records) == 2
        assert hook.records[0].phase == "forward_begin"
        assert hook.records[1].phase == "step"
        assert hook.records[1].batch_size == 8

    def test_hook_with_risk_model(self):
        """Risk scores are populated when a model is passed."""
        risk_model = OOMRiskModel(mode="rule")
        hook = TrainingHook(risk_model=risk_model)

        hook.on_forward_begin()
        hook.on_forward_end()
        hook.on_backward_begin()
        hook.on_backward_end()
        hook.on_optimizer_step()
        risk = hook.on_step_complete(batch_size=32)

        assert isinstance(risk, float)
        assert 0.0 <= risk <= 1.0
        assert hook.last_risk == risk
        # Risk model should have recorded one entry
        assert len(risk_model.history) == 1

    def test_multiple_steps(self):
        """Multiple steps accumulate correctly."""
        hook = TrainingHook()

        for i in range(5):
            with hook.wrap_step(batch_size=4):
                pass

        step_records = [r for r in hook.records if r.phase == "step"]
        assert len(step_records) == 5
        assert step_records[-1].step == 4

    def test_with_real_model(self):
        """Hook works in a real (tiny) PyTorch training loop."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        hook = TrainingHook(risk_model=OOMRiskModel())

        for _ in range(3):
            hook.on_forward_begin()
            x = torch.randn(4, 10)
            out = model(x)
            hook.on_forward_end()

            hook.on_backward_begin()
            loss = criterion(out, torch.randint(0, 2, (4,)))
            loss.backward()
            hook.on_backward_end()

            hook.on_optimizer_step()
            optimizer.step()
            optimizer.zero_grad()
            risk = hook.on_step_complete(batch_size=4)
            assert 0.0 <= risk <= 1.0

        assert len([r for r in hook.records if r.phase == "step"]) == 3

    def test_shared_logger(self):
        """Hook uses a shared logger when provided."""
        logger = AllocatorLogger()
        hook = TrainingHook(logger=logger)

        with hook.wrap_step(batch_size=2):
            pass

        # Logger and hook share the same records list
        assert len(logger.records) == 2
        assert hook.records is logger.records
