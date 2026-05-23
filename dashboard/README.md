# AeroGrid v2.0.0 — Monitoring HUD

This is a React/Vite dashboard for the **Predictive GPU Memory Defragmenter** research prototype. It visualizes exported telemetry when available and falls back to bundled demo data for local UI development.

## 🚀 Getting Started

1.  **Install dependencies**:
    ```bash
    npm install
    ```

2.  **Start the development server**:
    ```bash
    npm run dev
    ```

3.  **Standard Live Sync**:
    The dashboard expects telemetry data in `public/live/`. The `rtx_oom_guard` system automatically syncs results from the `results/` directory to this location when using `rtx-oom-guard dashboard`.

## 📊 Dashboard Modules

| Module | Panel | Description |
|---|---|---|
| **Mission Control** | 01, 10 | Primary KPIs: OOMs prevented and cumulative VRAM recovered. |
| **VRAM Topology** | 01, 11 | Live 80GB physical memory layout with allocation distribution maps. |
| **Shadow Forecast** | 03, 08 | Predictive fragmentation timeline ($T+100ms$) and OOM threshold overlay. |
| **Scheduler** | 04 | Heatmap of the internal Transformer allocator attention weights. |
| **DDP Sync** | 05, 12 | Multi-GPU barrier synchronization status and NCCL overhead. |
| **Triton Inspector** | 06, 09 | Kernel-level latency profiling and compaction execution trace. |

## 🛠️ Telemetry Sync
The dashboard communicates with the `rtx_oom_guard` REST API (default: `localhost:8000`) to fetch historical benchmark traces and GPU telemetry. When the backend is absent, UI panels may show bundled demo JSON; do not cite those values as benchmark or production evidence.
