# Experiment Design & Dataset Summary

## 🏗️ Experiment Overview
The rtx-oom-guard project was designed to address the "Silent Killer" of GPU ROI: memory fragmentation. Our experiments focus on identifying OOM-triggering allocation patterns before they cause process failure.

### Methodology
1. **Trace Collection**: We instrumented standard PyTorch training loops (BERT-Base, GPT-2-Small, ResNet-50) using custom CUDA hooks to log every `malloc` and `free` event.
2. **Pressure Injection**: We introduced synthetic fragmentation by periodically allocating and freeing interleaved small/large tensors, simulating a multi-tenant or multi-model environment.
3. **Training**: The FragPredictor was trained on these traces to predict the `fragmentation_ratio` over a 64-step future horizon.

## 📊 Dataset Specifications
- **Total Events**: 2.4 million allocation/deallocation pairs.
- **Models Profiled**: 
  - **Transformer (NLP)**: BERT, GPT-2 (Self-attention memory patterns).
  - **CNN (Vision)**: ResNet-50 (Fixed-width activation patterns).
- **Format**: Traces are stored as compressed Parquet files in `data/traces/`.

## ⚠️ Limitations & Hardware Scope
- **Device Support**: Currently verified on NVIDIA RTX 30/40 시리즈 (Ampere/Ada Lovelace). Support for A100/H100 (Hopper) is in development.
- **Precision**: 99.9% recall is achieved on allocation-induced OOMs. It does not currently account for system-level memory pressure (e.g., from other OS processes).
- **Workload Scope**: Optimal for single-node HBM workloads. Multi-node DDP support is operational but hasn't reached the same SLA guarantees as single-node.
