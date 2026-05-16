from rtx_oom_guard.llm_system.kv_cache_manager import PagedKVCacheAdapter

def test_paged_kv_cache_adapter():
    adapter = PagedKVCacheAdapter(num_blocks=100, block_size=16, block_byte_size=1048576)

    # Test initial state
    meta = adapter.get_metadata()
    assert meta["total_blocks"] == 100
    assert meta["free_blocks"] == 100
    assert meta["allocated_blocks"] == 0
    assert meta["fragmentation_score"] == 0.0

    # Simulate sequence allocation
    success = adapter.allocate(sequence_id=1, num_blocks=5)
    assert success

    meta = adapter.get_metadata()
    assert meta["free_blocks"] == 95
    assert meta["fragmentation_score"] >= 0.0

    # Test free
    adapter.free(sequence_id=1)
    assert len(adapter.free_physical_blocks) == 100

    # Test contiguous compaction reset
    adapter.compact_cache()
    assert len(adapter.free_physical_blocks) == 100
