"""Unit tests for the AllocationCollector."""

import torch
import pytest

from rtx_oom_guard.profiler.collector import AllocationCollector
from rtx_oom_guard.utils import DefragConfig


class TestAllocationCollector:
    @pytest.fixture(autouse=True)
    def mock_cuda(self, monkeypatch):
        # We enforce execution of the CUDA collector telemetry even on CPU machines
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        # Mock memory stats simulator
        def mock_alloc():
            mock_alloc.calls += 1
            return mock_alloc.calls * 1024 
        mock_alloc.calls = 0
        monkeypatch.setattr(torch.cuda, "memory_allocated", mock_alloc)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 8192)

    def test_manual_record(self):
        collector = AllocationCollector()
        
        # Trigger synthetic deltas mathematically modeled by the mock fixture
        collector.record()
        collector.record()
        
        # Validate that the telemetry daemon successfully traced the metrics
        assert collector.event_count == 2

    def test_to_dataframe(self):
        collector = AllocationCollector()
        df = collector.to_dataframe()
        assert df.empty  # No events yet

    def test_save_empty(self, tmp_path):
        collector = AllocationCollector()
        path = str(tmp_path / "test.parquet")
        collector.save(path)
        # Should warn and not create file for empty collector

    def test_clear(self):
        collector = AllocationCollector()
        collector.clear()
        assert collector.event_count == 0


class TestDefragConfig:
    def test_save_load(self, tmp_path):
        config = DefragConfig(frag_threshold=0.8, seq_len=128)
        path = str(tmp_path / "config.json")
        config.save(path)
        loaded = DefragConfig.load(path)
        assert loaded.frag_threshold == 0.8
        assert loaded.seq_len == 128
