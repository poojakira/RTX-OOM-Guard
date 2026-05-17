# What This Project Actually Does (And Doesn't Do)

This is a **RESEARCH PROTOTYPE**, not a production tool. It demonstrates the concept of proactive memory management for PyTorch training, but has significant limitations.

## What it CAN do

- Monitor CUDA memory allocation patterns
- Predict when fragmentation will cause OOM based on historical patterns
- Trigger garbage collection and cache clearing before OOM hits

## What it CANNOT do

Actually defragment CUDA memory. PyTorch's caching allocator owns the memory pool. You cannot move allocated tensors to new addresses without breaking autograd graph references. The "compaction" strategy works by releasing cached (freed) blocks back to CUDA and letting the allocator re-coalesce them on next allocation.

## Why the transformer predictor is overkill

The transformer-based predictor is overkill for this problem. A simple linear regression on `(allocated_bytes, num_segments, time_since_last_gc)` would probably work just as well. I used a transformer because I wanted to learn the architecture, not because the problem requires it.

## Hardware limitations

Tested only on RTX 4060 (8GB). Behavior on multi-GPU setups or A100s with different memory architectures is unknown.

## The tiered policy is less sophisticated than it sounds

The "tiered policy" (compact → evict → emergency) sounds sophisticated but in practice, `torch.cuda.empty_cache()` does most of the work. The eviction policy (moving tensors to CPU) breaks if those tensors are needed for backward pass within the same step.

## Biggest lesson

The real solution to OOM in production is gradient checkpointing + mixed precision + smaller batch size. This tool is useful for understanding **WHY** you're running out of memory, not for magically fixing it.

## Performance overhead

The monitoring thread adds ~2-3% training time overhead due to CUDA synchronization needed to read accurate memory stats.
