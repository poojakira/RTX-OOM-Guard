"""
rtx_oom_guard.collector — High-frequency CUDA allocation trace collector.

Captures allocation events by polling torch.cuda.memory_allocated() at
sub-millisecond intervals during model training. Each event records the
memory delta, direction (alloc/free), absolute memory, and timestamp.

Usage::

    collector = AllocationCollector()
    collector.start()
    # ... training loop ...
    collector.stop()
    df = collector.to_dataframe()
    collector.save("data/traces/gpt2_run1.parquet")
"""

import time
import threading
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
from rtx_oom_guard.utils import get_logger, DefragConfig, ensure_cuda

log = get_logger("collector")


class AllocationCollector:
    """
    Collects CUDA memory allocation traces at high frequency.

    This collector uses two complementary strategies:
    1. **Polling mode**: A background thread polls memory_allocated() at configurable intervals.
    2. **Hook mode**: Manual instrumentation points record events at specific training steps.
    """

    def __init__(self, config: Optional[DefragConfig] = None):
        self.config = config or DefragConfig()
        self._events: List[Dict] = []
        self._lock = threading.Lock()
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self._last_mem: int = 0
        self._start_time: float = 0.0

    # ── Manual Hook API ────────────────────────────────────────────────────

    def record(self) -> None:
        """Record a single allocation event at the current memory state."""
        if not torch.cuda.is_available():
            return
        current = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        ts = time.perf_counter()

        delta = current - self._last_mem
        if delta == 0:
            return

        event = {
            "timestamp_ns": int(ts * 1e9),
            "delta_bytes": delta,
            "action": 1 if delta > 0 else 0,  # 1=alloc, 0=free
            "abs_allocated": current,
            "abs_reserved": reserved,
            "fragmentation": 1.0 - (current / reserved) if reserved > 0 else 0.0,
        }

        with self._lock:
            self._events.append(event)
            if len(self._events) > self.config.max_events:
                self._events.pop(0)

        self._last_mem = current

    # ── Background Polling API ─────────────────────────────────────────────

    def start(self) -> None:
        """Start background polling thread."""
        ensure_cuda()
        self._last_mem = torch.cuda.memory_allocated()
        self._start_time = time.perf_counter()
        self._active = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="rtx_oom_guard-collector")
        self._thread.start()
        log.info("Collector started (polling every %.1fms)", self.config.poll_interval_ms)

    def stop(self) -> None:
        """Stop background polling thread."""
        self._active = False
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info("Collector stopped. %d events recorded.", len(self._events))

    def _poll_loop(self) -> None:
        interval = self.config.poll_interval_ms / 1000.0
        while self._active:
            self.record()
            time.sleep(interval)

    # ── Data Export ────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Export events to a pandas DataFrame with computed features."""
        with self._lock:
            if not self._events:
                return pd.DataFrame()
            df = pd.DataFrame(self._events)

        # Compute derived features
        df["size_gb"] = df["delta_bytes"].abs() / (1024**3)
        df["time_delta_ms"] = df["timestamp_ns"].diff().fillna(0) / 1e6
        return df

    def save(self, path: str) -> None:
        """Save traces to Parquet format."""
        df = self.to_dataframe()
        if df.empty:
            log.warning("No events to save.")
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        log.info("Saved %d events to %s (%.1f KB)", len(df), path, Path(path).stat().st_size / 1024)

    @property
    def event_count(self) -> int:
        return len(self._events)

    def clear(self) -> None:
        """Clear all collected events."""
        with self._lock:
            self._events.clear()
        self._last_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0


def collect_from_model(model_name: str, iterations: int = 200, config: Optional[DefragConfig] = None) -> int:
    """
    Run a model training loop and collect allocation traces.

    Args:
        model_name: One of 'gpt2', 'resnet50', 'bert'
        iterations: Number of training iterations
        config: Optional configuration

    Returns:
        Number of events collected
    """
    ensure_cuda()
    config = config or DefragConfig()
    device = "cuda"

    log.info("Collecting traces for %s (%d iterations)...", model_name, iterations)

    # Build model
    if model_name == "gpt2":
        from rtx_oom_guard.trainer._models import build_gpt2
        model, inputs = build_gpt2(device)
    elif model_name == "resnet50":
        from rtx_oom_guard.trainer._models import build_resnet50
        model, inputs = build_resnet50(device)
    elif model_name == "bert":
        from rtx_oom_guard.trainer._models import build_bert
        model, inputs = build_bert(device)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: gpt2, resnet50, bert")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    collector = AllocationCollector(config)

    # Warmup (5 iterations, no recording)
    for _ in range(5):
        optimizer.zero_grad()
        out = model(inputs)
        loss = out.sum() if isinstance(out, torch.Tensor) else out[0].sum()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    collector.start()

    for i in range(iterations):
        optimizer.zero_grad()
        collector.record()  # Extra hook at step boundary

        out = model(inputs)
        collector.record()

        loss = out.sum() if isinstance(out, torch.Tensor) else out[0].sum()
        loss.backward()
        collector.record()

        optimizer.step()
        collector.record()

        if (i + 1) % 50 == 0:
            log.info("  [%s] Iteration %d/%d — %d events", model_name, i + 1, iterations, collector.event_count)

    collector.stop()

    save_path = f"{config.trace_dir}/{model_name}_trace.parquet"
    collector.save(save_path)

    return collector.event_count
