"""
rtx_oom_guard.cli — Enterprise Command-Line Interface.

CLI entry point for the Predictive GPU Memory Defragmenter.
Provides a unified interface for profiling, training, benchmarking, and serving.

Usage::

    rtx-oom-guard profile --model gpt2
    rtx-oom-guard train --epochs 20
    rtx-oom-guard benchmark --runs 5
    rtx-oom-guard server --port 8000
"""

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Optional

from rtx_oom_guard.utils import get_logger

log = get_logger("cli")

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console()
    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False  # pragma: no cover
    console = None  # pragma: no cover


def print_banner() -> None:
    """Prints a startup banner."""
    if HAS_RICH:
        banner = Text("▶ rtx_oom_guard: Predictive GPU Memory Defragmenter", style="bold #76b900")
        console.print(Panel(banner, border_style="#76b900"))
    else:
        print("=" * 60)
        print("▶ rtx_oom_guard: Predictive GPU Memory Defragmenter")
        print("=" * 60)


def _print(msg: str, style: Optional[str] = None) -> None:
    """Print with optional Rich styling."""
    if HAS_RICH and style:
        console.print(f"[{style}]{msg}[/]")
    else:
        print(msg)


# These are registered in pyproject.toml as console_scripts.
# Each must parse its own args when called standalone.


def collect_cmd() -> None:
    """Entry point for `rtx_oom_guard-collect` — collect CUDA allocation traces."""
    parser = argparse.ArgumentParser(
        description="Collect CUDA allocation traces from reference models."
    )
    parser.add_argument(
        "--model", choices=["gpt2", "resnet50", "bert", "all"], default="all",
        help="Model to profile (default: all)"
    )
    parser.add_argument(
        "--iterations", type=int, default=200,
        help="Training iterations per model (default: 200)"
    )
    args = parser.parse_args()
    print_banner()
    _print(f"▶ Starting telemetry collection for {args.model}...", "bold cyan")

    from rtx_oom_guard.profiler.collector import collect_from_model

    models = ["gpt2", "resnet50", "bert"] if args.model == "all" else [args.model]
    for model_name in models:
        try:
            count = collect_from_model(model_name, iterations=args.iterations)
            _print(f"  ✓ {model_name}: {count} events collected", "bold green")
        except Exception as e:
            _print(f"  ✗ {model_name}: {e}", "bold red")


def train_cmd() -> None:
    """Entry point for `rtx_oom_guard-train` — train the FragPredictor."""
    parser = argparse.ArgumentParser(
        description="Train the Transformer-based fragmentation predictor."
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--trace-dir", default="data/traces", help="Trace directory")
    args = parser.parse_args()
    print_banner()
    _print("▶ Starting FragPredictor training...", "bold cyan")

    from rtx_oom_guard.utils import DefragConfig
    from rtx_oom_guard.trainer.trainer import train

    config = DefragConfig(
        train_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        trace_dir=args.trace_dir,
    )
    metrics = train(config=config, verbose=True)
    _print(f"  ✓ Training complete. Test MAE: {metrics.get('test_mae', 'N/A')}", "bold green")


def benchmark_cmd() -> None:
    """Entry point for `rtx_oom_guard-benchmark` — run the local benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Run the local GPU defragmentation benchmark."
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--steps", type=int, default=100, help="Steps per run")
    args = parser.parse_args()
    print_banner()
    _print(f"▶ Launching benchmark ({args.runs} runs, {args.steps} steps)...", "bold #76b900")

    # Import and run the benchmark directly instead of shelling out
    sys.argv = [
        "run_local_benchmark.py",
        "--runs", str(args.runs),
        "--steps", str(args.steps),
    ]
    from benchmarks.run_local_benchmark import main as benchmark_main
    benchmark_main()


def serve_cmd() -> None:
    """Entry point for `rtx_oom_guard-serve` — launch the REST API server."""
    parser = argparse.ArgumentParser(
        description="Launch the rtx_oom_guard REST API server."
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    args = parser.parse_args()
    print_banner()
    _print(f"▶ Starting rtx_oom_guard REST API on {args.host}:{args.port}...", "bold cyan")

    import uvicorn
    uvicorn.run("rtx_oom_guard.api:app", host=args.host, port=args.port)


def dashboard_cmd() -> None:
    """Entry point for `rtx_oom_guard-dashboard` — launch the standalone monitoring dashboard."""
    parser = argparse.ArgumentParser(
        description="Launch the AEON CORE monitoring dashboard."
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to run the dashboard on")
    args = parser.parse_args()
    print_banner()
    _print("▶ Starting Standalone AEON CORE Dashboard...", "bold #10b981")

    import subprocess
    import time
    import webbrowser

    # Add current directory to PYTHONPATH so uvicorn can find the module
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "rtx_oom_guard.api:app", "--host", "127.0.0.1", "--port", str(args.port)],
        stdout=subprocess.PIPE, # Keep logs available
        stderr=subprocess.STDOUT,
        env=env,
        text=True
    )
    
    # Simple background thread to drain uvicorn output
    def drain_api():
        for line in api_proc.stdout:
            if "error" in line.lower():
                log.error(f"[API] {line.strip()}")
            else:
                log.debug(f"[API] {line.strip()}")
                
    threading.Thread(target=drain_api, daemon=True).start()

    _print(f"▶ Dashboard available at http://127.0.0.1:{args.port}", "bold cyan")
    _print("▶ Opening browser...", "bold yellow")
    
    time.sleep(1.5) # Wait for uvicorn to bind
    webbrowser.open(f"http://127.0.0.1:{args.port}")
    
    try:
        while True:
            time.sleep(1)
            if api_proc.poll() is not None:
                _print("✗ Dashboard Server died unexpectedly.", "bold red")
                break
    except KeyboardInterrupt:
        _print("\nStopping Dashboard Service...", "bold yellow")
        api_proc.terminate()
        api_proc.wait(timeout=5)
        _print("✔ System offline. Have a productive day!", "bold green")




def main() -> None:
    """Unified CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="rtx_oom_guard: Predictive Memory Defragmenter Engine"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")

    # 1. Profile command
    profile_p = subparsers.add_parser("profile", help="Collect raw VRAM telemetry from reference models")
    profile_p.add_argument("--model", choices=["gpt2", "resnet50", "bert", "all"], default="all")
    profile_p.add_argument("--iterations", type=int, default=200)

    # 2. Train command
    train_p = subparsers.add_parser("train", help="Train the FragPredictor model")
    train_p.add_argument("--epochs", type=int, default=20)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--trace-dir", default="data/traces")

    # 3. Simulate command
    sim_p = subparsers.add_parser("simulate", help="Run the benchmark simulation")
    sim_p.add_argument("--runs", type=int, default=5)
    sim_p.add_argument("--steps", type=int, default=100)

    # 4. Server command
    serve_p = subparsers.add_parser("server", help="Launch the live Telemetry API server")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--host", default="0.0.0.0")

    # 5. Dashboard command
    dash_p = subparsers.add_parser("dashboard", help="Launch the AEON CORE monitoring dashboard")
    dash_p.add_argument("--root", default=".")

    # 6. Mock Telemetry command
    mock_p = subparsers.add_parser("mock-telemetry", help="Generate synthetic telemetry for dashboard testing")
    mock_p.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds")

    # 7. Status command
    status_p = subparsers.add_parser("status", help="Check the health of all rtx_oom_guard components")

    args = parser.parse_args()
    print_banner()

    if args.command == "mock-telemetry":
        _print(f"▶ Generating synthetic telemetry (interval={args.interval}s)...", "bold yellow")
        from rtx_oom_guard.defrag_engine.defragmenter import GPUMemoryDefragmenter
        import time
        import random
        
        engine = GPUMemoryDefragmenter()
        try:
            step = 0
            while True:
                alloc = random.uniform(2000, 4000)
                resv = random.uniform(4000, 6000)
                
                # Periodically simulate a compaction event
                if step % 5 == 0:
                    engine._history.append({
                        "timestamp": time.time(),
                        "reason": "predictive_sweep",
                        "freed_mb": random.uniform(100, 300),
                        "elapsed_ms": random.uniform(8, 15),
                        "megabytes_compacted": random.uniform(500, 1500)
                    })
                
                engine._persist_telemetry(alloc, resv, force=True)
                _print(f"  → Heartbeat: {alloc:.1f}MB allocated / {resv:.1f}MB reserved", "dim")
                time.sleep(args.interval)
                step += 1
        except KeyboardInterrupt:
            _print("\n▶ Mock telemetry stopped.", "bold red")

    elif args.command == "status":
        _print("▶ Checking system health...", "bold cyan")
        import torch
        from rtx_oom_guard.utils import DefragConfig
        
        # 1. GPU Check
        if torch.cuda.is_available():
            _print(f"  ✓ GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})", "bold green")
        else:
            _print("  ! GPU: CUDA not available. Running in simulation mode.", "bold yellow")
            
        # 2. Predictor Check
        config = DefragConfig()
        if os.path.exists(config.checkpoint_path):
            _print(f"  ✓ Predictor: Checkpoint found at {config.checkpoint_path}", "bold green")
        else:
            _print("  ! Predictor: No local checkpoint. Using default pre-trained weights.", "bold blue")
            
        # 3. Dashboard Check
        dist_path = Path(__file__).parent.parent / "dashboard" / "dist"
        if dist_path.exists():
            _print("  ✓ Dashboard: Production build found (AeroGrid v2.0.0)", "bold green")
        else:
            _print("  ! Dashboard: Production build missing. Run 'npm run build' in dashboard dir.", "bold red")  # pragma: no cover
            
        _print("\n▶ System status: READY", "bold green")

    elif args.command == "profile":
        _print(f"▶ Starting telemetry collection for {args.model}...", "bold cyan")
        from rtx_oom_guard.profiler.collector import collect_from_model
        models = ["gpt2", "resnet50", "bert"] if args.model == "all" else [args.model]
        for model_name in models:
            try:
                count = collect_from_model(model_name, iterations=args.iterations)
                _print(f"  ✓ {model_name}: {count} events collected", "bold green")
            except Exception as e:
                _print(f"  ✗ {model_name}: {e}", "bold red")

    elif args.command == "train":
        _print("▶ Starting FragPredictor training...", "bold cyan")
        from rtx_oom_guard.utils import DefragConfig
        from rtx_oom_guard.trainer.trainer import train
        config = DefragConfig(
            train_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            trace_dir=args.trace_dir,
        )
        train(config=config, verbose=True)

    elif args.command == "simulate":
        _print(f"▶ Launching benchmark ({args.runs} runs, {args.steps} steps)...", "bold #76b900")
        sys.argv = [
            "run_local_benchmark.py",
            "--runs", str(args.runs),
            "--steps", str(args.steps),
        ]
        from benchmarks.run_local_benchmark import main as benchmark_main
        benchmark_main()

    elif args.command == "server":
        _print(f"▶ Starting rtx_oom_guard REST API on {args.host}:{args.port}...", "bold cyan")
        import uvicorn
        uvicorn.run("rtx_oom_guard.api:app", host=args.host, port=args.port)

    elif args.command == "dashboard":
        _print("▶ Starting AEON CORE Dashboard & Telemetry Sync...", "bold #10b981")
        from rtx_oom_guard.dashboard import DashboardManager
        mgr = DashboardManager(root_dir=args.root)
        mgr.start_sync()
        mgr.start_dashboard()
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            _print("\n▶ Shutting down dashboard...", "bold yellow")
            mgr.stop_sync()


if __name__ == "__main__":  # pragma: no cover
    main()  # pragma: no cover
