# RTX-OOM-Guard

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![CI](https://github.com/poojakira/RTX-OOM-Guard/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/RTX-OOM-Guard/actions)

**Proactive CUDA memory defragmenter for PyTorch that predicts and prevents GPU out-of-memory (OOM) crashes by actively compacting VRAM during training.**

---

## Problem

PyTorch's `CachingAllocator` leaves VRAM fragmented during long training runs, causing OOM crashes even when total free memory appears sufficient. Gradient checkpointing and reduced batch sizes sacrifice throughput. RTX-OOM-Guard predicts fragmentation before it causes a crash and proactively compacts live tensors into contiguous blocks.

---

## Key Features

- **`GPUMemoryDefragmenter`** — Repacks scattered live PyTorch tensors into a single contiguous VRAM allocation; physically replaces `.data` pointers so autograd and optimizer states continue uninterrupted
- **Triton Compaction Kernels** — High-bandwidth custom Triton copy kernel (`compaction_kernels.py`); falls back to PyTorch if Triton unavailable
- **`FragPredictor`** — ML model that predicts fragmentation score from the allocation event stream
- **`DefragMonitor`** — Background daemon thread polling at 50 ms intervals; auto-triggers compaction when predicted fragmentation exceeds threshold (default 0.7)
- **`OOMRiskModel`** — Risk model scoring OOM probability from memory traces
- **`AllocationCollector`** — Hooks into PyTorch allocator to log per-step allocation/free events
- **`auto_instrument`** — Zero-code-change wrapper: `model, optimizer = auto_instrument(model, optimizer)`
- **`DDPSyncManager`** — Distributed Data Parallel (DDP) integration for multi-GPU training
- **FastAPI REST API** — Exposes defrag endpoints for remote monitoring
- **React Dashboard** — Vite+React frontend with 13 panels: VRAM map, fragmentation chart, DDP choreography, Triton trace, latency graphs
- **KV Cache Manager** — LLM-specific KV cache memory optimization

---

## Benchmark Results

Benchmarked across BERT-base, BERT-large, GPT-2, GPT-2-medium, ResNet-50, ResNet-101, EfficientNet-B4, ViT-Large with batch sizes 2–16 and VRAM configs 6 GB/8 GB/12 GB.

| Metric | Baseline | With RTX-OOM-Guard |
|---|---|---|
| OOM crashes (100-step run) | 23 | 0 |
| Peak VRAM utilization | 94% | 87% |
| Iteration time overhead | — | < 2% |
| Fragmentation ratio | 0.61 avg | 0.18 avg |

---

## Quick Start

### Install

```bash
git clone https://github.com/poojakira/RTX-OOM-Guard.git
cd RTX-OOM-Guard
pip install -e .
```

### Zero-Code-Change Integration

```python
from rtx_oom_guard import auto_instrument
model, optimizer = auto_instrument(model, optimizer)
# ... standard training loop, no other changes needed
```

### Manual Monitor

```python
from rtx_oom_guard import DefragMonitor
monitor = DefragMonitor(threshold=0.7)
monitor.start()
for batch in dataloader:
    monitor.record_alloc(tensor.numel() * tensor.element_size())
    output = model(batch)
    loss.backward()
    optimizer.step()
monitor.stop()
print(monitor.stats())
```

### Docker

```bash
docker build -t rtx-oom-guard .
docker run --gpus all rtx-oom-guard
```

### React Dashboard

```bash
cd dashboard
npm install
npm run dev  # http://localhost:5173
```

---

## Configuration

Edit `configs/config.yaml`:

```yaml
defrag:
  threshold: 0.7       # Fragmentation score to trigger compaction
  interval_ms: 50      # Monitor polling interval
  cooldown_steps: 10   # Steps between compaction runs
  use_triton: true     # Use Triton kernels if available
logging:
  results_dir: results
```

---

## Project Structure

```
.
├── src/rtx_oom_guard/
│   ├── defrag_engine/     # GPUMemoryDefragmenter, compactor, policy
│   ├── defrag/            # Custom Triton copy kernel
│   ├── scheduler/         # DefragMonitor, OOMRiskModel
│   ├── predictor/         # FragPredictor ML model
│   ├── profiler/          # AllocationCollector, AllocatorLogger
│   ├── trainer/           # auto_instrument, DefragCallback, DDPSyncManager
│   ├── llm_system/        # KV cache manager
│   └── api/main.py        # FastAPI REST API
├── dashboard/            # React + Vite frontend (13 panels)
├── benchmarks/           # OOM benchmarks, model fragmentation tests
├── data/traces/          # 100+ Parquet memory trace files
├── results/              # Benchmark results
├── tests/                # 50+ test files
├── configs/config.yaml
├── Dockerfile
└── run_benchmark.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Roadmap

- [ ] Automatic Triton kernel tuning per GPU model
- [ ] Integration with HuggingFace Trainer as a callback
- [ ] Support for FSDP (Fully Sharded Data Parallel)
- [ ] Live memory visualization via WebSocket
- [ ] PyPI package release

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

Built by [Pooja Kiran](https://github.com/poojakira) — M.S. student at Arizona State University.
