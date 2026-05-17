"""
scripts/workload_simulator.py
==============================
High-fidelity GPU memory workload simulator.

Models the memory lifecycle of real DL training steps:
  Forward  →  activations allocated (quadratic in seq_len for Transformers)
  Backward →  gradients allocated, activations freed
  Optimizer → state buffers (Adam m/v) allocated
  Cleanup  →  gradient zeroing, periodic cache clears

Fragmentation emerges naturally from interleaved alloc/free patterns
with 2 MB block quantization (matching PyTorch's caching allocator).

Usage::

    from scripts.workload_simulator import GPUWorkload, TransformerSpec
    wl = GPUWorkload(TransformerSpec.gpt2(), vram_mb=8192)
    events = wl.run(steps=500)
"""

from __future__ import annotations

import math
import time
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import numpy as np
import logging

log = logging.getLogger("workload_simulator")
if not log.handlers:
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("[SIM] %(message)s"))
    log.addHandler(sh)
    log.setLevel(logging.INFO)


# Architecture specifications

@dataclass
class TransformerSpec:
    """Memory geometry for a Transformer model."""
    name: str
    layers: int
    hidden: int
    heads: int
    vocab: int
    seq_len: int
    batch_size: int

    # Derived sizes (MB) — computed in __post_init__
    param_mb: float = 0.0
    activation_per_layer_mb: float = 0.0
    gradient_mb: float = 0.0
    optimizer_state_mb: float = 0.0

    def __post_init__(self):
        # Parameters: embeddings + layers*(QKV + FFN + norms)
        embed_params = self.vocab * self.hidden + 1024 * self.hidden
        layer_params = (4 * self.hidden * self.hidden +   # QKV + out proj
                        2 * 4 * self.hidden * self.hidden + # FFN up/down
                        4 * self.hidden)                     # layer norms
        total_params = embed_params + self.layers * layer_params
        self.param_mb = total_params * 4 / (1024 ** 2)  # fp32

        # Activations per layer: attention scores + FFN intermediates
        # Attention: B * H * S * S (quadratic in seq_len)
        attn_mb = (self.batch_size * self.heads * self.seq_len * self.seq_len * 4) / (1024 ** 2)
        ffn_mb = (self.batch_size * self.seq_len * 4 * self.hidden * 4) / (1024 ** 2)
        self.activation_per_layer_mb = attn_mb + ffn_mb

        # Gradients mirror params
        self.gradient_mb = self.param_mb

        # Adam: 2x param size (momentum + variance)
        self.optimizer_state_mb = 2 * self.param_mb

    @classmethod
    def gpt2(cls, batch_size: int = 6, seq_len: int = 512) -> TransformerSpec:
        return cls("GPT-2", layers=12, hidden=768, heads=12,
                   vocab=50257, seq_len=seq_len, batch_size=batch_size)

    @classmethod
    def gpt2_medium(cls, batch_size: int = 4, seq_len: int = 512) -> TransformerSpec:
        return cls("GPT-2-Medium", layers=24, hidden=1024, heads=16,
                   vocab=50257, seq_len=seq_len, batch_size=batch_size)

    @classmethod
    def bert_base(cls, batch_size: int = 16, seq_len: int = 256) -> TransformerSpec:
        return cls("BERT-Base", layers=12, hidden=768, heads=12,
                   vocab=30522, seq_len=seq_len, batch_size=batch_size)

    @classmethod
    def bert_large(cls, batch_size: int = 8, seq_len: int = 256) -> TransformerSpec:
        return cls("BERT-Large", layers=24, hidden=1024, heads=16,
                   vocab=30522, seq_len=seq_len, batch_size=batch_size)

    @classmethod
    def vit_large(cls, batch_size: int = 8, seq_len: int = 197) -> TransformerSpec:
        return cls("ViT-Large", layers=24, hidden=1024, heads=16,
                   vocab=1000, seq_len=seq_len, batch_size=batch_size)

    @classmethod
    def llama_7b(cls, batch_size: int = 2, seq_len: int = 2048) -> TransformerSpec:
        return cls("Llama-7B", layers=32, hidden=4096, heads=32,
                   vocab=32000, seq_len=seq_len, batch_size=batch_size)


@dataclass
class CNNSpec:
    """Memory geometry for a CNN model."""
    name: str
    layers: int
    base_channels: int
    input_hw: int
    batch_size: int

    param_mb: float = 0.0
    activation_per_layer_mb: float = 0.0
    gradient_mb: float = 0.0
    optimizer_state_mb: float = 0.0

    def __post_init__(self):
        total_params = 0
        hw = self.input_hw
        cin = 3
        for i in range(self.layers):
            cout = self.base_channels * (2 ** min(i // 3, 4))
            total_params += cin * cout * 9 + cout  # 3x3 conv + bias
            cin = cout
            if i % 2 == 1:
                hw = max(hw // 2, 1)

        self.param_mb = total_params * 4 / (1024 ** 2)
        # Activations: feature map at mid-resolution
        mid_hw = self.input_hw // 4
        mid_ch = self.base_channels * 4
        self.activation_per_layer_mb = (
            self.batch_size * mid_ch * mid_hw * mid_hw * 4 / (1024 ** 2)
        )
        self.gradient_mb = self.param_mb
        self.optimizer_state_mb = 2 * self.param_mb

    @classmethod
    def resnet50(cls, batch_size: int = 32) -> CNNSpec:
        return cls("ResNet-50", layers=50, base_channels=64,
                   input_hw=224, batch_size=batch_size)

    @classmethod
    def resnet101(cls, batch_size: int = 16) -> CNNSpec:
        return cls("ResNet-101", layers=101, base_channels=64,
                   input_hw=224, batch_size=batch_size)

    @classmethod
    def efficientnet(cls, batch_size: int = 24) -> CNNSpec:
        return cls("EfficientNet-B4", layers=40, base_channels=48,
                   input_hw=380, batch_size=batch_size)


# Block-level allocator model

BLOCK_SIZE_MB = 2  # PyTorch caching allocator quantizes to 2 MB blocks


def _quantize(mb: float) -> float:
    """Round up to nearest BLOCK_SIZE_MB."""
    return math.ceil(mb / BLOCK_SIZE_MB) * BLOCK_SIZE_MB


@dataclass
class _Block:
    """Represents an allocated memory block."""
    block_id: int
    size_mb: float
    tag: str        # "param", "activation", "gradient", "optimizer", "temp"
    step: int
    timestamp_ns: int


class CachingAllocator:
    """
    Simulates PyTorch's CUDA caching allocator at the block level.

    Tracks allocated and free blocks, computes fragmentation as the ratio
    of unusable free memory (blocks too small to satisfy typical requests)
    to total reserved memory.
    """

    def __init__(self, vram_mb: float = 8192.0, noise_std: float = 0.02):
        self.vram_mb = vram_mb
        self.noise_std = noise_std
        self._blocks: Dict[int, _Block] = {}
        self._next_id = 0
        self._reserved_mb = 0.0
        self._allocated_mb = 0.0
        # Simplified address space modeling
        self._block_map: List[tuple[int, int, int]] = [] # (start, size, bid) - sorted by start

    def alloc(self, size_mb: float, tag: str, step: int) -> Optional[int]:
        """Allocate a block. Returns block_id or None on OOM."""
        quantized = _quantize(size_mb + abs(np.random.normal(0, self.noise_std * size_mb)))

        # 1. First-fit search in reserved segments (holes)
        self._block_map.sort()
        curr_addr = 0
        best_gap_start = -1
        
        for start, size, bid in self._block_map:
            gap = start - curr_addr
            if gap >= quantized:
                best_gap_start = curr_addr
                break
            curr_addr = start + size
            
        if best_gap_start != -1:
            # Found a hole!
            bid = self._next_id
            self._next_id += 1
            self._block_map.append((best_gap_start, quantized, bid))
            self._allocated_mb += quantized
            self._blocks[bid] = _Block(bid, quantized, tag, step, int(time.perf_counter() * 1e9))
            return bid

        # 2. No hole found, try to expand reserved memory at the end
        if curr_addr + quantized > self.vram_mb:
            log.debug("  OOM: Requested %.1f MB, but Max Contiguous Gap = %.1f MB", quantized, self.vram_mb - curr_addr)
            return None # OOM from topological fragmentation

        bid = self._next_id
        self._next_id += 1
        self._block_map.append((curr_addr, quantized, bid))
        self._reserved_mb = max(self._reserved_mb, curr_addr + quantized)
        self._allocated_mb += quantized
        self._blocks[bid] = _Block(bid, quantized, tag, step, int(time.perf_counter() * 1e9))
        return bid

    def free(self, block_id: int) -> float:
        """Free a block and remove it from the address map."""
        if block_id not in self._blocks:
            return 0.0
        blk = self._blocks.pop(block_id)
        self._allocated_mb -= blk.size_mb
        # Remove from block map
        self._block_map = [b for b in self._block_map if b[2] != block_id]
        return blk.size_mb

    def empty_cache(self):
        """Release all tailing reserved memory back to OS."""
        if not self._block_map:
            freed = self._reserved_mb
            self._reserved_mb = 0
            return freed
        
        self._block_map.sort()
        last_start, last_size, _ = self._block_map[-1]
        end_addr = last_start + last_size
        freed = self._reserved_mb - end_addr
        self._reserved_mb = end_addr
        return max(0.0, freed)

    @property
    def allocated_mb(self) -> float:
        return max(0.0, self._allocated_mb)

    @property
    def reserved_mb(self) -> float:
        return max(BLOCK_SIZE_MB, self._reserved_mb)

    @property
    def fragmentation(self) -> float:
        """Topological fragmentation: (1 - max_hole / total_abs_free)."""
        if self.vram_mb <= 0:
            return 0.0
        
        # Calculate holes
        self._block_map.sort()
        curr_addr = 0
        holes = []
        for start, size, bid in self._block_map:
            gap = start - curr_addr
            if gap > 0:
                holes.append(gap)
            curr_addr = start + size
        
        # Add the big hole at the end
        if curr_addr < self.vram_mb:
            holes.append(self.vram_mb - curr_addr)
            
        if not holes:
            return 0.0
        
        max_hole = max(holes)
        total_free = sum(holes)
        return 1.0 - (max_hole / total_free) if total_free > 0 else 0.0

    @property
    def utilization(self) -> float:
        return self._allocated_mb / self.vram_mb if self.vram_mb > 0 else 0.0

    def snapshot(self) -> Dict[str, float]:
        return {
            "allocated_mb": round(self.allocated_mb, 2),
            "reserved_mb": round(self.reserved_mb, 2),
            "free_cached_mb": round(self.reserved_mb - self.allocated_mb, 2),
            "fragmentation": round(self.fragmentation, 6),
            "utilization": round(self.utilization, 6),
            "num_blocks": len(self._blocks),
            "num_free_blocks": 0, # In topological model, holes are dynamic
        }

    def defragment(self):
        """Physical compaction: repack all live blocks to start of address space."""
        live_blocks = sorted(self._block_map, key=lambda x: x[0])
        new_map = []
        curr_addr = 0
        for _, size, bid in live_blocks:
            new_map.append((curr_addr, size, bid))
            curr_addr += size
        self._block_map = new_map
        self._reserved_mb = curr_addr
        return True


# Workload runner

@dataclass
class TraceEvent:
    """Single memory event in a trace."""
    timestamp_ns: int
    step: int
    phase: str          # forward / backward / optimizer / cleanup
    action: int         # 1=alloc, 0=free
    delta_bytes: int    # positive=alloc, negative=free
    abs_allocated: float
    abs_reserved: float
    fragmentation: float
    utilization: float
    tag: str
    oom: bool = False


class GPUWorkload:
    """
    Simulates a full DL training workload on a virtual GPU.

    Parameters
    ----------
    spec : TransformerSpec | CNNSpec
        Architecture specification.
    vram_mb : float
        Total VRAM in MB (default 8192 = 8 GB).
    noise_std : float
        Gaussian noise on allocation sizes (0.02 = 2%).
    cache_clear_interval : int
        Steps between full cache clears (simulates empty_cache calls).
    """

    def __init__(
        self,
        spec: TransformerSpec | CNNSpec,
        vram_mb: float = 8192.0,
        noise_std: float = 0.02,
        cache_clear_interval: int = 50,
        defrag_strategy: Optional[str] = None, # None or "predictive"
        defrag_threshold: float = 0.7,
        defrag_overhead_ms: float = 15.0,
    ):
        self.spec = spec
        self.allocator = CachingAllocator(vram_mb=vram_mb, noise_std=noise_std)
        self.cache_clear_interval = cache_clear_interval
        self.defrag_strategy = defrag_strategy
        self.defrag_threshold = defrag_threshold
        self.defrag_overhead_ms = defrag_overhead_ms
        self.events: List[TraceEvent] = []
        self._param_blocks: List[int] = []
        self._optimizer_blocks: List[int] = []
        self._total_defrag_overhead_ns: int = 0

    def _emit(self, step: int, phase: str, action: int,
              delta_mb: float, tag: str, oom: bool = False):
        snap = self.allocator.snapshot()
        self.events.append(TraceEvent(
            timestamp_ns=int(time.perf_counter() * 1e9),
            step=step,
            phase=phase,
            action=action,
            delta_bytes=int(delta_mb * 1024 * 1024) * (1 if action == 1 else -1),
            abs_allocated=snap["allocated_mb"],
            abs_reserved=snap["reserved_mb"],
            fragmentation=snap["fragmentation"],
            utilization=snap["utilization"],
            tag=tag,
            oom=oom,
        ))

    def _alloc_or_oom(self, size_mb: float, tag: str, step: int, phase: str) -> Optional[int]:
        bid = self.allocator.alloc(size_mb, tag, step)
        if bid is None:
            self._emit(step, phase, 1, size_mb, tag, oom=True)
            # Emergency: clear cache and retry
            self.allocator.empty_cache()
            bid = self.allocator.alloc(size_mb, tag, step)
            if bid is None:
                return None
        self._emit(step, phase, 1, size_mb, tag, oom=False)
        return bid

    def apply_defragmentation(self, step: int, phase: str):
        """Simulate a proactive compaction event."""
        # 1. Record overhead
        overhead_ns = int(self.defrag_overhead_ms * 1e6)
        self._total_defrag_overhead_ns += overhead_ns
        
        # 2. Topological compaction
        self.allocator.defragment()
        
        # 3. Emit defrag event
        self._emit(step, phase, 0, 0, tag="predictive_defrag")
        log.info("Step %d - Predictive Defrag repacked memory positions.", step)

    def run(self, steps: int = 500, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Execute the workload for `steps` training iterations.

        Returns a list of event dicts suitable for Parquet export.
        """
        rng = np.random.RandomState(seed)
        random.seed(seed)

        spec = self.spec

        # Step 0: Allocate model parameters (persist for lifetime)
        if isinstance(spec, TransformerSpec):
            per_layer = spec.param_mb / max(spec.layers, 1)
            for layer_i in range(spec.layers):
                bid = self._alloc_or_oom(per_layer, "param", 0, "init")
                if bid is None:
                    return [asdict(e) for e in self.events]
                self._param_blocks.append(bid)
        else:
            bid = self._alloc_or_oom(spec.param_mb, "param", 0, "init")
            if bid is None:
                return [asdict(e) for e in self.events]
            self._param_blocks.append(bid)

        # Allocate optimizer state (Adam m + v, persist)
        opt_bid = self._alloc_or_oom(spec.optimizer_state_mb, "optimizer", 0, "init")
        if opt_bid is None:
            return [asdict(e) for e in self.events]
        self._optimizer_blocks.append(opt_bid)

        # Training loop
        for step in range(1, steps + 1):
            activation_ids = []
            gradient_ids = []

            # ── Forward pass ──
            n_act_allocs = max(1, spec.layers // 2)
            for i in range(n_act_allocs):
                act_size = spec.activation_per_layer_mb * (1 + rng.normal(0, 0.05))
                bid = self._alloc_or_oom(max(act_size, 0.5), "activation", step, "forward")
                if bid is None:
                    return [asdict(e) for e in self.events]
                activation_ids.append(bid)

            # Small temporary buffers (attention masks, scaling tensors)
            n_temps = rng.randint(2, 8)
            temp_ids = []
            for _ in range(n_temps):
                temp_size = rng.uniform(0.5, 4.0)  # 0.5–4 MB
                bid = self._alloc_or_oom(temp_size, "temp", step, "forward")
                if bid is None:
                    return [asdict(e) for e in self.events]
                temp_ids.append(bid)

            # Free temps immediately (creates fragmentation gaps)
            for tid in temp_ids:
                freed = self.allocator.free(tid)
                self._emit(step, "forward", 0, freed, "temp")

            # ── Backward pass ──
            grad_size = spec.gradient_mb / max(spec.layers, 1)
            for i in range(max(1, spec.layers // 3)):
                bid = self._alloc_or_oom(
                    grad_size * (1 + rng.normal(0, 0.03)), "gradient", step, "backward"
                )
                if bid is None:
                    return [asdict(e) for e in self.events]
                gradient_ids.append(bid)

            # Free activations during backward (as gradients are computed)
            for aid in activation_ids:
                freed = self.allocator.free(aid)
                self._emit(step, "backward", 0, freed, "activation")

            # ── Optimizer step ──
            # Small temp allocs for Adam updates
            for _ in range(rng.randint(1, 4)):
                temp_size = rng.uniform(1.0, 6.0)
                bid = self._alloc_or_oom(temp_size, "optimizer_temp", step, "optimizer")
                if bid is None:
                    return [asdict(e) for e in self.events]
                freed = self.allocator.free(bid)
                self._emit(step, "optimizer", 0, freed, "optimizer_temp")

            # ── Cleanup: free gradients (zero_grad) ──
            for gid in gradient_ids:
                freed = self.allocator.free(gid)
                self._emit(step, "cleanup", 0, freed, "gradient")

            # ── Predictive Defragmentation Check ──
            if self.defrag_strategy == "predictive":
                current_frag = self.allocator.fragmentation
                if current_frag > self.defrag_threshold:
                    self.apply_defragmentation(step, "cleanup")

            # Periodic cache clear (Baseline mechanism)
            if self.cache_clear_interval > 0 and step % self.cache_clear_interval == 0:
                cleared = self.allocator.empty_cache()
                if cleared > 0:
                    self._emit(step, "cleanup", 0, cleared, "cache_clear")

        return [asdict(e) for e in self.events]


# CLI verification

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run workload simulator verification")
    ap.add_argument("--verify", action="store_true", help="Run quick verification")
    ap.add_argument("--steps", type=int, default=200, help="Training steps")
    args = ap.parse_args()

    specs = [
        TransformerSpec.gpt2(),
        TransformerSpec.bert_base(),
        CNNSpec.resnet50(),
    ]

    for spec in specs:
        wl = GPUWorkload(spec, vram_mb=8192, noise_std=0.02)
        events = wl.run(steps=args.steps, seed=42)
        frags = [e["fragmentation"] for e in events]
        ooms = sum(1 for e in events if e["oom"])
        print(f"{spec.name:20s}  events={len(events):5d}  "
              f"frag=[{min(frags):.3f}, {max(frags):.3f}]  "
              f"mean={np.mean(frags):.3f}  OOMs={ooms}")

    if args.verify:
        # Verify non-trivial fragmentation
        wl = GPUWorkload(TransformerSpec.gpt2(), vram_mb=8192)
        events = wl.run(steps=100)
        frags = [e["fragmentation"] for e in events]
        assert max(frags) > 0.05, f"Max frag too low: {max(frags)}"
        assert len(events) > 500, f"Too few events: {len(events)}"
        print("\n✓ Verification passed")
