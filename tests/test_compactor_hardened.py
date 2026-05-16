from unittest.mock import patch
from rtx_oom_guard.defrag_engine.compactor import MemoryCompactor

def test_compactor_lifecycle_mocked():
    """Verify compaction cycle with mocked CUDA."""
    compactor = MemoryCompactor()
    
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", side_effect=[1000, 1000, 1000, 1000]), \
         patch("torch.cuda.memory_reserved", side_effect=[2000, 1500]), \
         patch("torch.cuda.synchronize"), \
         patch("torch.cuda.empty_cache"), \
         patch("gc.collect"):
        
        # 1000/2000 = 0.5 pre-frag. 1000/1500 = 0.33 post-frag.
        res = compactor.compact(reason="high_frag")
        
        assert res["reason"] == "high_frag"
        assert res["freed_mb"] == (2000 - 1500) / (1024**2)
        assert compactor.total_compactions == 1
        assert len(compactor.history) == 1

def test_compactor_no_cuda():
    """Verify compaction handles CPU gracefully."""
    compactor = MemoryCompactor()
    with patch("torch.cuda.is_available", return_value=False):
        res = compactor.compact()
        assert res["skipped"] == True

def test_compactor_stats_aggregation():
    """Verify compactor stats properties."""
    compactor = MemoryCompactor()
    compactor._history = [
        {"freed_mb": 10.0},
        {"freed_mb": 20.0},
    ]
    assert compactor.total_freed_mb == 30.0
    assert compactor.total_compactions == 0 # manually edited history doesn't increment counter
