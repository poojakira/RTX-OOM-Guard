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
- **Has never been benchmarked on a real GPU.** All previously published numbers came from synthetic numpy curves, not `torch.cuda.memory_stats()`. Those docs have been deleted.

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

## To Actually Validate This

```bash
# On a Colab T4 (free):
pip install -e .
python -c "
import torch
from rtx_oom_guard import auto_instrument

model = torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6).cuda()
optimizer = torch.optim.Adam(model.parameters())
model, optimizer = auto_instrument(model, optimizer)

# Print memory stats before/after training
print(torch.cuda.memory_stats()['allocated_bytes.all.peak'])
"
```

Until someone runs this and publishes the output, the system is unvalidated.

## Structure

- `src/rtx_oom_guard/defrag_engine/` — Core compaction logic
- `src/rtx_oom_guard/scheduler/` — Monitor, risk model, dataset
- `src/rtx_oom_guard/predictor/` — Transformer model (unused in practice)
- `src/rtx_oom_guard/profiler/` — Allocation event collector
- `src/rtx_oom_guard/trainer/` — Auto-instrumentation hooks

## Author

Pooja Kiran — [@poojakira](https://github.com/poojakira)
