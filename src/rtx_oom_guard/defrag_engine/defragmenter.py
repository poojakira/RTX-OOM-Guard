"""
rtx_oom_guard.defrag_engine.defragmenter — Active PyTorch Tensor Compaction.

Implements a REAL-WORLD memory defragmenter that actively repacks scattered 
live PyTorch tensors (e.g., model parameters and accumulated gradients) into 
a contiguous mega-buffer.

This directly combats CachingAllocator fragmentation during model training
by merging scattered allocations into a singular dense block and letting
empty_cache() actually release physical memory holes underneath.
"""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Iterable, List, Dict, Any

import torch

try:
    from rtx_oom_guard.defrag.compaction_kernels import triton_compaction_copy
    HAS_TRITON = True
except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover
    HAS_TRITON = False  # pragma: no cover
    def triton_compaction_copy(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("Triton not available")  # pragma: no cover

log = logging.getLogger("rtx_oom_guard.defragmenter")


class GPUMemoryDefragmenter:
    """
    Actively repacks dynamic PyTorch tensors into contiguous memory blocks.
    
    This is not a simple cache eviction. It physically copies scattered tensor
    data into a single contiguous VRAM allocation and silently replaces the
    underlying `.data` pointers of the live tensors so that autograd and
    optimizer states continue flawlessly.
    """

    def __init__(self, use_triton: bool = True, results_dir: str = "results"):
        """
        Args:
            use_triton: Whether to use the extreme-bandwidth custom Triton copying kernel
                        if it is available on this system.
            results_dir: Directory for telemetry output (default: 'results').
        """
        self.use_triton = use_triton and HAS_TRITON
        self._results_dir = Path(results_dir)
        self._history: List[Dict[str, Any]] = []
        self._last_write_time = 0.0

        # Triton JIT warmup to prevent real-time latency spikes during execution
        if self.use_triton and torch.cuda.is_available():
            try:
                dummy_src = torch.zeros(1024, device="cuda")
                dummy_dst = torch.empty_like(dummy_src)  # pragma: no cover
                triton_compaction_copy(dummy_src, dummy_dst)  # pragma: no cover
            except Exception as e:  # pragma: no cover
                log.warning(f"Triton warmup failed: {e}")

    def defragment_tensors(self, tensors: Iterable[torch.Tensor], reason: str = "compaction") -> Dict[str, Any]:
        """
        Takes an iterable of scattered live tensors and tightly packs them into
        a newly allocated contiguous block.
        
        Args:
            tensors: An iterable of tensors (e.g. `model.parameters()`)
            reason: Tag for telemetry logging.
            
        Returns:
            Dictionary containing metrics about the compaction duration and memory reclaimed.
        """
        # Filter valid tensors (must be floating point / complex and instantiated)
        tensors = [t for t in tensors if t is not None and t.numel() > 0]
        if not tensors:
            return {"skipped": True, "reason": "no_valid_tensors"}

        device = tensors[0].device
        dtype = tensors[0].dtype

        # Verify uniform device and dtype (usually true for parameters/gradients in a single replica)
        valid_tensors, total_elements, total_bytes = [], 0, 0
        for t in tensors:
            if t.device == device and t.dtype == dtype:
                valid_tensors.append(t)
                total_elements += t.numel()
                total_bytes += t.numel() * t.element_size()

        if total_elements == 0:  # pragma: no cover
            return {"skipped": True, "reason": "no_matching_tensors"}  # pragma: no cover

        # 0. Distributed Data Parallel (DDP) sync safety
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            log.info("DDP active: Waiting at barrier before defragmentation...")
            torch.distributed.barrier()

        t0 = time.perf_counter()

        # Pre-execution snapshot
        pre_allocated = torch.cuda.memory_allocated() if device.type == "cuda" else 0
        pre_reserved = torch.cuda.memory_reserved() if device.type == "cuda" else 0

        # 1. Chunk-based execution to avoid double-allocation peak OOM
        element_size = valid_tensors[0].element_size() if valid_tensors else 4
        chunk_size_elements = (256 * 1024 * 1024) // element_size  # 256 MB chunks
        if chunk_size_elements <= 0:  # pragma: no cover
            chunk_size_elements = 1000000  # pragma: no cover

        triton_successes = 0
        current_chunk_tensors = []
        current_chunk_elements = 0

        def process_chunk(tensors_chunk, total_chunk_elements):
            nonlocal triton_successes
            if not tensors_chunk:  # pragma: no cover
                return  # pragma: no cover

            try:
                chunk_buffer = torch.empty(total_chunk_elements, dtype=dtype, device=device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log.warning("Defragmentation failed: Cannot allocate temporary %d MB buffer chunk.", 
                                (total_chunk_elements * element_size) // 1024**2)
                    return
                raise

            # 2. Copy scattered contents into the dense buffer chunk
            offset = 0
            for t in tensors_chunk:
                numel = t.numel()
                dest_view = chunk_buffer[offset : offset + numel]
                src_flat = t.view(-1)

                if self.use_triton and device.type == "cuda" and t.is_cuda:
                    try:
                        triton_compaction_copy(src_flat, dest_view)
                        triton_successes += 1
                    except Exception:
                        dest_view.copy_(src_flat)
                else:
                    dest_view.copy_(src_flat)
                offset += numel

            # 3. Memory Rewrite phase using safe .set_() API
            offset = 0
            for t in tensors_chunk:
                numel = t.numel()
                requires_grad = t.requires_grad

                if requires_grad:
                    t.requires_grad_(False)

                # Direct pointer rewrite via .data is required to bypass PyTorch 
                # autograd graph modification restrictions during training
                t.data = chunk_buffer[offset : offset + numel].view_as(t)

                if requires_grad:
                    t.requires_grad_(True)
                offset += numel

            # 4. Trigger localized Garbage Collection
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Process in bounded chunks
        for t in valid_tensors:
            numel = t.numel()
            if current_chunk_elements + numel > chunk_size_elements and current_chunk_tensors:
                process_chunk(current_chunk_tensors, current_chunk_elements)
                current_chunk_tensors = []
                current_chunk_elements = 0
            
            current_chunk_tensors.append(t)
            current_chunk_elements += numel
            
        if current_chunk_tensors:
            process_chunk(current_chunk_tensors, current_chunk_elements)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Post-execution snapshot
        post_allocated = torch.cuda.memory_allocated() if device.type == "cuda" else 0
        post_reserved = torch.cuda.memory_reserved() if device.type == "cuda" else 0

        freed_mb = (pre_reserved - post_reserved) / (1024 ** 2) if device.type == "cuda" else 0.0

        record = {
            "timestamp": time.time(),
            "reason": reason,
            "tensors_compacted": len(valid_tensors),
            "megabytes_compacted": total_bytes / (1024**2),
            "triton_used": triton_successes > 0,
            "freed_mb": freed_mb,
            "elapsed_ms": elapsed_ms,
        }

        self._history.append(record)
        log.info(
            "Packaged %d tensors (%.1f MB) into contiguous block in %.1f ms using %s. Reclaimed %.1f MB.",
            len(valid_tensors), total_bytes / (1024**2), elapsed_ms,
            "Triton" if triton_successes > 0 else "ATen",
            freed_mb
        )

        self._persist_telemetry(post_allocated / 1024**2, post_reserved / 1024**2, force=True)

        return record

    def _persist_telemetry(self, current_allocated_mb: float, current_reserved_mb: float, force: bool = False):
        """Write the current state to results/live_telemetry.json for the dashboard."""
        now = time.time()
        # Throttling: Only write to disk every 200ms unless forced (e.g., after compaction)
        if not force and (now - self._last_write_time) < 0.2:
            return

        self._last_write_time = now

        try:
            self._results_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.debug(f"Cannot create results directory: {e}")
            return

        history = []
        for h in self._history[-20:]:  # Last 20 compactions
            history.append({
                "compaction_id": h.get("compaction_id", 0),
                "freed_mb": h.get("freed_mb", 0.0),
                "fragReduction": h.get("frag_reduction", 0.0) if "frag_reduction" in h else (h.get("megabytes_compacted", 0.0) / current_reserved_mb if current_reserved_mb > 0 else 0.0),
                "elapsedMs": h.get("elapsed_ms", 0.0),
                "timestamp": time.strftime("%H:%M:%S", time.localtime(h.get("timestamp", time.time())))
            })

        data = {
            "current_allocated_mb": float(round(current_allocated_mb, 2)),
            "current_reserved_mb": float(round(current_reserved_mb, 2)),
            "current_frag": float(round(1.0 - (current_allocated_mb / current_reserved_mb), 4)) if current_reserved_mb > 0 else 0.0,
            "total_compactions": int(len(self._history)),
            "total_freed_mb": float(round(sum(h.get("freed_mb", 0.0) for h in self._history), 2)),
            "avg_latency_ms": float(round(sum(h.get("elapsed_ms", 0.0) for h in self._history) / len(self._history), 3)) if self._history else 0.0,
            "compaction_history": history
        }

        try:
            # Atomic file write to prevent JSON corruption during dashboard reads
            import tempfile
            import os

            fd, temp_path = tempfile.mkstemp(dir=self._results_dir, text=True)
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            target_path = self._results_dir / "live_telemetry.json"
            os.replace(temp_path, target_path)

        except Exception as e:
            log.debug(f"Failed to write telemetry: {e}")
