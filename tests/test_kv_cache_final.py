from unittest.mock import MagicMock, patch
from rtx_oom_guard.llm_system.kv_cache_manager import PagedKVCacheAdapter

def test_kv_cache_initialization():
    """Verify KV cache initialization with physical memory handles."""
    adapter = PagedKVCacheAdapter(num_blocks=10, block_size=16, block_byte_size=1024*1024)
    assert adapter.num_blocks == 10
    assert len(adapter.free_physical_blocks) == 10

def test_kv_cache_allocate_oom():
    """Verify OOM risk handling when blocks are unavailable (Line 38-39)."""
    adapter = PagedKVCacheAdapter(num_blocks=1, block_size=16, block_byte_size=1024)
    # Attempt to allocate 2 blocks when only 1 exists
    with patch("rtx_oom_guard.llm_system.kv_cache_manager.log") as mock_log:
        success = adapter.allocate(sequence_id=42, num_blocks=2)
        assert success is False
        assert mock_log.warning.called
        assert "OOM risk" in mock_log.warning.call_args[0][0]

def test_kv_cache_fragmentation_scores():
    """Verify fragmentation logic across empty, full, and scattered states."""
    # 1. Zero blocks (Line 57)
    adapter = PagedKVCacheAdapter(num_blocks=0, block_size=16, block_byte_size=1024)
    assert adapter.get_fragmentation_score() == 0.0

    # 2. Fully allocated (Line 61)
    adapter = PagedKVCacheAdapter(num_blocks=5, block_size=16, block_byte_size=1024)
    adapter.allocate(sequence_id=1, num_blocks=5)
    assert adapter.get_fragmentation_score() == 1.0 # free_count is 0

    # 3. Scattered blocks (Line 71)
    adapter = PagedKVCacheAdapter(num_blocks=10, block_size=16, block_byte_size=1024)
    # Allocate [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] reversed
    # adapter.free_physical_blocks = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    adapter.allocate(sequence_id=1, num_blocks=1) # pops 0
    adapter.allocate(sequence_id=2, num_blocks=1) # pops 1
    # Free [0, 1] means they are at end: [9, 8, 7, 6, 5, 4, 3, 2, 0, 1]
    # To create gaps, we need to allocate and free specific ones.
    adapter = PagedKVCacheAdapter(num_blocks=5, block_size=16, block_byte_size=1024)
    # Free blocks: [0, 1, 2, 3, 4]
    adapter.free_physical_blocks = [0, 2, 4] # Artificially create gaps
    # Gaps: 2 > 0+1 (YES), 4 > 2+1 (YES). Gaps = 2.
    assert adapter.get_fragmentation_score() > 0.0

def test_kv_cache_compact_callback():
    """Verify physical compaction callback is triggered (Line 83)."""
    adapter = PagedKVCacheAdapter(num_blocks=5, block_size=16, block_byte_size=1024)
    mock_cb = MagicMock()
    adapter.compact_cache(engine_callback=mock_cb)
    assert mock_cb.called
    assert len(adapter.free_physical_blocks) == 5

def test_kv_cache_sync_defragmenter():
    """Verify sync logic with telemetry defragmenter (Line 94-96)."""
    adapter = PagedKVCacheAdapter(num_blocks=5, block_size=16, block_byte_size=1024)
    mock_defrag = MagicMock()
    mock_defrag._history = []
    
    with patch("torch.cuda.is_available", return_value=False):
        adapter.sync_with_defragmenter(mock_defrag)
        assert len(mock_defrag._history) == 1
        assert mock_defrag._history[0]["reason"] == "kv_cache_sync"
        assert "kv_fragmentation" in mock_defrag._history[0]

def test_kv_cache_metadata_map():
    """Verify AeroGrid physical block map generation."""
    adapter = PagedKVCacheAdapter(num_blocks=4, block_size=16, block_byte_size=1024)
    adapter.allocate(sequence_id=1, num_blocks=2) # Uses blocks 3, 2 (reversed pop)
    meta = adapter.get_metadata()
    assert meta["physical_block_map"] == [0, 0, 1, 1]
    assert meta["allocated_blocks"] == 2
