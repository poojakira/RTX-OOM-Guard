"""
rtx_oom_guard.kernels — Custom Triton kernels for memory operations and analysis.

Includes low-level tensor operations that demonstrate GPU memory hierarchy
understanding by optimizing for contiguous memory access and caching.
"""

import torch

try:
    import triton  # pragma: no cover
    import triton.language as tl  # pragma: no cover
    HAS_TRITON = True  # pragma: no cover
except ImportError:
    HAS_TRITON = False
    class DummyTriton:
        def jit(self, func): return func
        def cdiv(self, a, b): return (a + b - 1) // b
        
    class DummyLanguage:
        def __getattr__(self, name): return lambda *args, **kwargs: None
        @property
        def constexpr(self): return int
    
    triton = DummyTriton()
    tl = DummyLanguage()

@triton.jit
def _compaction_copy_kernel(
    src_ptr, dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """
    A custom Triton kernel used for fast contiguous copying, simulating low-level
    defragmentation operations where fragmented tensors are compacted into
    a contiguous block of memory. It leverages Triton's block mechanics to
    maximize memory bandwidth utilization.
    """
    # Compute the starting offset of this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the offsets for the elements in this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to avoid out-of-bounds memory access
    mask = offsets < n_elements

    # Load from source, bypassing cache where possible, but here we just do normal load
    x = tl.load(src_ptr + offsets, mask=mask)

    # Store to destination
    tl.store(dst_ptr + offsets, x, mask=mask)

def triton_compaction_copy(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Uses a custom Triton kernel to execute a fast contiguous copy from src to dst.
    This simulates the physical movement of memory blocks during defragmentation.

    Design Tradeoff:
    Prefer slightly over-allocating scratch space here to avoid fragmentation 
    in long-running jobs, even if it marginally increases transient VRAM peak.
    """
    assert src.is_cuda and dst.is_cuda, "Tensors must be on CUDA."
    assert src.numel() == dst.numel(), "Source and destination must have the same number of elements."

    # Ensure source is contiguous for the 1D view, though in a real defrag we might handle striding
    src_flat = src.flatten()
    dst_flat = dst.flatten()
    n_elements = src_flat.numel()

    if n_elements == 0:
        return dst

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _compaction_copy_kernel[grid](
        src_flat,
        dst_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return dst

@triton.jit
def _fragmentation_scan_kernel(
    mem_blocks_ptr,  # array of block sizes (positive for allocated, negative for free)
    frag_scores_ptr, # output array
    n_blocks,
    BLOCK_SIZE: tl.constexpr
):
    """
    A parallel scan kernel that analyzes memory blocks to compute local fragmentation 
    metrics at the block level on the GPU, avoiding CPU transfer overhead.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_blocks

    sizes = tl.load(mem_blocks_ptr + offsets, mask=mask, other=0)

    # Simple heuristic: heavily fragmented if block is small and free (negative)
    # We output a score: 1.0 if highly fragmented free block, 0.0 otherwise.
    # In reality, fragmentation is a global property, but this simulates a parallel local pass.
    is_free = sizes < 0
    abs_size = tl.where(is_free, -sizes, sizes)

    # If the free block is smaller than 1MB (1024*1024 bytes), consider it fragmented
    is_small = abs_size < 1048576

    score = tl.where(is_free & is_small, 1.0, 0.0)
    tl.store(frag_scores_ptr + offsets, score, mask=mask)

def analyze_fragmentation_triton(block_sizes: torch.Tensor) -> float:
    """
    Analyze block sizes directly on the GPU using Triton.
    Returns a fragmentation score based on small free blocks.
    
    Args:
        block_sizes: 1D tensor of block sizes in bytes (negative implies free).
    """
    if not block_sizes.is_cuda:
        block_sizes = block_sizes.cuda()

    n_blocks = block_sizes.numel()
    if n_blocks == 0:
        return 0.0

    scores = torch.zeros_like(block_sizes, dtype=torch.float32)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_blocks, meta['BLOCK_SIZE']),)

    _fragmentation_scan_kernel[grid](
        block_sizes,
        scores,
        n_blocks,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Aggregate on GPU and return CPU scalar
    return scores.mean().item()
