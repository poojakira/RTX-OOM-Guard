"""Tests for RTX-OOM-Guard fixes — migration, estimator, DDP coord, kill switch."""

import time
import torch
import numpy as np
import pytest

from rtx_oom_guard.frag_estimator import FragmentationEstimator
from rtx_oom_guard.kill_switch import KillSwitch
from rtx_oom_guard.ddp_coord import DDPCoordinator


class TestFragmentationEstimator:
    def test_initial_state(self):
        est = FragmentationEstimator(threshold=0.3)
        assert est.current_estimate == 0.0
        assert not est.should_compact

    def test_ewma_update_without_gpu(self):
        est = FragmentationEstimator(alpha=0.5, threshold=0.3)
        # Without CUDA, returns 0
        result = est.update()
        assert result == 0.0

    def test_threshold_logic(self):
        est = FragmentationEstimator(threshold=0.3)
        est._ewma = 0.2
        assert not est.should_compact
        est._ewma = 0.4
        assert est.should_compact

    def test_reset(self):
        est = FragmentationEstimator()
        est._ewma = 0.5
        est._samples = 100
        est.reset()
        assert est._ewma == 0.0
        assert est._samples == 0


class TestKillSwitch:
    def test_starts_active(self):
        ks = KillSwitch()
        assert ks.is_active

    def test_single_failure_triggers_cooldown(self):
        ks = KillSwitch(max_latency_ms=5.0, max_failures=3, base_cooldown_s=0.1)
        ks.record_latency(10.0)  # Over threshold
        assert not ks.is_active  # In cooldown
        time.sleep(0.15)
        assert ks.is_active  # Recovered

    def test_success_resets_failures(self):
        ks = KillSwitch(max_latency_ms=5.0, max_failures=3)
        ks.record_latency(10.0)
        ks._cooldown_until = 0  # Skip cooldown for test
        ks.record_latency(2.0)  # Success
        assert ks._consecutive_failures == 0

    def test_max_failures_permanently_disables(self):
        ks = KillSwitch(max_latency_ms=5.0, max_failures=3, base_cooldown_s=0.001)
        ks.record_latency(10.0)
        ks._cooldown_until = 0
        ks.record_latency(10.0)
        ks._cooldown_until = 0
        ks.record_latency(10.0)
        assert ks._permanently_disabled
        assert not ks.is_active

    def test_manual_reset(self):
        ks = KillSwitch(max_failures=1)
        ks.record_latency(100.0)
        assert not ks.is_active
        ks.reset()
        assert ks.is_active


class TestDDPCoordinator:
    def test_non_distributed_request(self):
        coord = DDPCoordinator()
        assert not coord.is_distributed
        coord.request_compaction()
        assert coord.check_and_sync() is True

    def test_no_request_returns_false(self):
        coord = DDPCoordinator()
        assert coord.check_and_sync() is False

    def test_request_consumed_after_check(self):
        coord = DDPCoordinator()
        coord.request_compaction()
        coord.check_and_sync()
        # Second check should be False (consumed)
        assert coord.check_and_sync() is False


class TestMigration:
    def test_migrate_optimizer_state_cpu(self):
        """Test optimizer migration logic on CPU (validates the code path)."""
        from rtx_oom_guard.migration import migrate_optimizer_state

        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Do a forward/backward to populate optimizer state
        x = torch.randn(3, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Verify optimizer has state
        assert len(optimizer.state) > 0

        # Migration on CPU should work (no-op for CUDA check)
        migrated = migrate_optimizer_state(optimizer, {}, torch.device("cpu"), non_blocking=False)
        # CPU tensors won't be migrated (is_cuda check)
        assert migrated == 0

    def test_migrate_gradients_cpu(self):
        from rtx_oom_guard.migration import migrate_gradients

        model = torch.nn.Linear(10, 5)
        x = torch.randn(3, 10)
        loss = model(x).sum()
        loss.backward()

        # CPU gradients won't be migrated
        migrated = migrate_gradients(model.parameters(), torch.device("cpu"), non_blocking=False)
        assert migrated == 0
