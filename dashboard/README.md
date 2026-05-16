# AeroGrid v2.0.0 — Monitoring HUD

This is the production-grade React/Vite dashboard for the **Predictive GPU Memory Defragmenter**. It provides real-time visibility into VRAM topology, fragmentation forecasts, and Triton kernel execution traces.

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
The dashboard communicates with the `rtx_oom_guard` REST API (default: `localhost:8000`) to fetch historical benchmarks and real-time GPU statistics. Field names are standardized to `camelCase` (e.g., `elapsedMs`, `fragReduction`) for seamless React state integration.
