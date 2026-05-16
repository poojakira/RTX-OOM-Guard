"""Unit tests for the MemoryCompactor."""

import torch

from rtx_oom_guard.defrag_engine.compactor import MemoryCompactor


class TestMemoryCompactor:
    def test_compact_without_cuda(self):
        compactor = MemoryCompactor(force_gc=False)
        # Should not crash even if CUDA unavailable (returns skipped)
        result = compactor.compact(reason="test")
        if not torch.cuda.is_available():
            assert result.get("skipped") is True

    def test_history_tracking(self):
        compactor = MemoryCompactor(force_gc=False)
        if torch.cuda.is_available():
            compactor.compact(reason="test1")
            compactor.compact(reason="test2")
            assert compactor.total_compactions == 2
            assert len(compactor.history) == 2
            assert compactor.history[0]["reason"] == "test1"

    def test_total_freed(self):
        compactor = MemoryCompactor()
        if torch.cuda.is_available():
            # Allocate and free to create something to compact
            t = torch.randn(1024, 1024, device="cuda")
            del t
            result = compactor.compact()
            assert result["elapsed_ms"] >= 0
            assert "freed_mb" in result
