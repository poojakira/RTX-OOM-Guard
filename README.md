# gpu-memory-optimizer

Research prototype: proactive CUDA memory defragmentation for PyTorch training loops.

## What This Does

Repacks scattered model parameter tensors into a contiguous VRAM buffer by replacing `.data` pointers, then calls `empty_cache()` to release the fragmented blocks back to the caching allocator.

```python
from rtx_oom_guard import auto_instrument

model, optimizer = auto_instrument(model, optimizer)
# Training loop runs normally. Monitor triggers compaction when fragmentation exceeds threshold.
```

## What This Does NOT Do

- **Does not migrate optimizer state.** Adam's `exp_avg` and `exp_avg_sq` remain in their original scattered allocations. After compaction, parameters are contiguous but optimizer state is still fragmented. A complete solution would walk `optimizer.state_dict()` and include state tensors.
- **Does not migrate gradients.** `p.grad` tensors are not repacked.
- **Does not use the Transformer predictor in practice.** The `FragPredictor` model has no training script, no labeled dataset, and no validated weights. The system actually uses `OOMRiskModel` — a rule-based sigmoid heuristic on `[utilization, fragmentation_ratio, allocation_rate]`.
- **Not yet validated on a real GPU.** Previous synthetic benchmarks were deleted. Run `notebooks/colab_t4_validation.ipynb` on a Colab T4 to produce real numbers. See `results/colab_t4_results.json` for latest run output (if present).

## How It Works

1. `DefragMonitor` runs in a background daemon thread, polling memory state every 50ms
2. `OOMRiskModel` scores OOM probability from current fragmentation ratio
3. If score > threshold (default 0.7), triggers `GPUMemoryDefragmenter.defragment_tensors()`
4. Defragmenter allocates a contiguous buffer, copies parameters into it via `tensor.copy_()` (or Triton if available), then rebinds `.data` pointers
5. `gc.collect()` + `torch.cuda.empty_cache()` releases the old scattered blocks

## Known Issues

- **DDP barrier from daemon thread.** The defragmenter checks `threading.current_thread()` and skips `barrier()` if not on main thread (would deadlock NCCL). For DDP, use `pending_compaction=True` and trigger from the training loop.
- **Kill switch is permanent.** If prediction latency exceeds 5ms, `self._killed = True` and the monitor exits. No recovery. This prevents the monitor from blocking training under GIL contention, but means any CPU spike permanently disables it.
- **`chunk_buffer` lifetime.** After compaction, parameter `.data` tensors are views into `chunk_buffer` (a local variable). The storage stays alive via PyTorch's reference counting, but correctness depends on storage-aliasing semantics surviving `gc.collect()` + `empty_cache()`.
- **Triton kernel is a copy loop.** `_compaction_copy_kernel` is a textbook `load → store` — functionally identical to `tensor.copy_()`. It exercises the Triton JIT path but provides no bandwidth advantage over ATen.

## Development Notes

**The thread leak.** `_persist_telemetry` originally spawned a new daemon thread on every write (every 200ms under active monitoring). Under sustained operation, this leaked hundreds of threads. Fixed with synchronous atomic writes (tempfile + `os.replace`).

**The namespace collision.** This repo originally used `apex_aegis` as its package name — same as the Aerospace-Trajectory-Simulator repo. Renamed to `rtx_oom_guard` across 134 files.

**The empty tensor list.** The monitor's `_predict_and_act` originally called `defragment_tensors([])` — an empty list. The defragmenter returned `{"skipped": True}` every time. The monitor was running but never actually defragmenting. Fixed by adding `register_tensors()` so the monitor knows which tensors to compact.

## Validation

Run the Colab notebook on a free T4:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/poojakira/RTX-OOM-Guard/blob/main/notebooks/colab_t4_validation.ipynb)

```bash
pip install git+https://github.com/poojakira/RTX-OOM-Guard.git
```

The notebook trains a 12-layer Transformer (d_model=1024, batch=16, seq=256) under deliberate memory fragmentation, with and without the guard. Workload is sized to push T4 to ~12-14GB — close enough to the 15.6GB ceiling that fragmentation can trigger OOM.

Results are committed to `results/colab_t4_results.json` after each run. If the guard shows no improvement, that's documented too — the likely cause is that optimizer state (which isn't compacted) dominates the fragmentation pattern.

## Structure

- `src/rtx_oom_guard/defrag_engine/` — Core compaction logic
- `src/rtx_oom_guard/scheduler/` — Monitor, risk model, dataset
- `src/rtx_oom_guard/predictor/` — Transformer model (unused in practice)
- `src/rtx_oom_guard/profiler/` — Allocation event collector
- `src/rtx_oom_guard/trainer/` — Auto-instrumentation hooks

## Author

Pooja Kiran — [@poojakira](https://github.com/poojakira)
