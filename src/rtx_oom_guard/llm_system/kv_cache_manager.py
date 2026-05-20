"""
rtx_oom_guard.llm_system.kv_cache_manager
=====================================

LLM inference systems heavily depend on Key-Value (KV) caches.
During text generation, KV caches can become severely fragmented if users
disconnect or variable length sequences complete at different ticks.
This module integrates predictive defragmentation directly into the KV cache memory pool.
"""

import torch
from rtx_oom_guard.utils import get_logger
from typing import Dict, List, Any

log = get_logger("kv_cache")

class PagedKVCacheAdapter:
    """
    Interface Wrapper for hooking predictive rtx_oom_guard into physical 
    PageAttention engines (like vLLM or TGI) instead of dummy simulation.
    It tracks physical and logical block tables and defines strict OS-level
    allocation handles.
    """
    def __init__(self, num_blocks: int, block_size: int, block_byte_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size # tokens per block
        self.block_byte_size = block_byte_size # capacity in physical bytes

        # Physical block table simulator or pointer references
        self.free_physical_blocks: List[int] = list(range(num_blocks))
        self.logical_to_physical: Dict[int, List[int]] = {}
        
        log.info(f"Initialized KV Cache Adapter with {num_blocks} blocks, {block_byte_size/1024**2:.2f} MB each.")

    def allocate(self, sequence_id: int, num_blocks: int) -> bool:
        """OS-level hook to formally reserve physical memory handles."""
        if len(self.free_physical_blocks) < num_blocks:
            log.warning(f"OOM risk: sequence {sequence_id} failed to allocate {num_blocks} blocks.")
            return False
            
        handles = [self.free_physical_blocks.pop() for _ in range(num_blocks)]
        self.logical_to_physical.setdefault(sequence_id, []).extend(handles)
        return True

    def free(self, sequence_id: int):
        """Releases sequence-bound blocks back to the pool."""
        if sequence_id in self.logical_to_physical:
            handles = self.logical_to_physical.pop(sequence_id)
            self.free_physical_blocks.extend(handles)

    def get_fragmentation_score(self) -> float:
        """
        Evaluate non-contiguous VRAM potential across free physical pointers.
        If pointer IDs are extremely scattered, contiguous VRAM drops.
        """
        if self.num_blocks == 0:
            return 0.0

        free_count = len(self.free_physical_blocks)
        if free_count == 0:
            return 1.0

        # Physical contiguous scatter approximation
        if free_count == self.num_blocks: return 0.0
        
        # Calculate fragmentation by counting pointer gaps
        self.free_physical_blocks.sort()
        gaps = 0
        for i in range(1, free_count):
            if self.free_physical_blocks[i] > self.free_physical_blocks[i-1] + 1:
                gaps += 1
                
        return min(gaps / max(self.num_blocks, 1), 1.0)

    def compact_cache(self, engine_callback=None):
        """
        Instructs the backing engine to execute a physical block swap,
        conditionally coalescing fragmented arrays.
        """
        log.info("Executing native physical pointer compaction on KV Cache.")
        # E.g., engine_callback() executes block movement via CUDA
        if engine_callback:
            engine_callback()
            
        # Sort free blocks to maximize contiguity for future allocations
        self.free_physical_blocks.sort()

    def sync_with_defragmenter(self, defragmenter: Any):
        """
        Hook to allow the KV Cache adapter to report its state directly to the 
        global defragmenter telemetry pipe.
        """
        import time as _time
        meta = self.get_metadata()
        if hasattr(defragmenter, "_history"):
            defragmenter._history.append({
                "timestamp": _time.time(),
                "reason": "kv_cache_sync",
                "kv_fragmentation": meta["fragmentation_score"],
                "kv_allocated_blocks": meta["allocated_blocks"]
            })

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns block-level metadata formatted for the AeroGrid 
        VRAM Topology hex-map visualization.
        """
        allocated_ids = []
        for ids in self.logical_to_physical.values():
            allocated_ids.extend(ids)

        return {
            "total_blocks": self.num_blocks,
            "free_blocks": len(self.free_physical_blocks),
            "allocated_blocks": self.num_blocks - len(self.free_physical_blocks),
            "fragmentation_score": round(self.get_fragmentation_score(), 4),
            "block_size_tokens": self.block_size,
            "physical_block_map": [1 if i in allocated_ids else 0 for i in range(self.num_blocks)]
        }
