"""
rtx_oom_guard.monitor — Real-time background defragmentation monitor.

Runs the FragPredictor in a daemon thread, analyzing the allocation event
stream and triggering the Compactor when predicted fragmentation exceeds
the configured threshold.

Features:
    - Configurable prediction interval (default: 50ms)
    - Automatic kill switch if prediction latency exceeds 5ms
    - Cooldown timer to prevent excessive compaction
    - Thread-safe event recording API
    - Full telemetry for post-analysis
"""

import time
import threading
import torch
import numpy as np
from typing import Optional, Dict, List
from rtx_oom_guard.predictor.model import FragPredictor
from rtx_oom_guard.defrag_engine.defragmenter import GPUMemoryDefragmenter
from rtx_oom_guard.utils import get_logger, DefragConfig, parse_memory_snapshot

log = get_logger("monitor")


class DefragMonitor:
    """
    Background thread that predicts fragmentation and triggers proactive compaction.

    Usage::

        monitor = DefragMonitor(threshold=0.7)
        monitor.start()

        # In your training loop:
        for batch in dataloader:
            monitor.record_alloc(tensor.numel() * tensor.element_size())
            output = model(batch)
            loss.backward()
            optimizer.step()

        monitor.stop()
        print(monitor.stats())
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.7,
        config: Optional[DefragConfig] = None,
        compactor: Optional[GPUMemoryDefragmenter] = None,
        predictor: Optional[FragPredictor] = None,
    ):
        self.config = config or DefragConfig()
        self.config.frag_threshold = threshold

        # Components
        self.compactor = compactor or GPUMemoryDefragmenter()
        self._model = predictor
        self._model_path = model_path or self.config.checkpoint_path

        # Thread state
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Event ring buffer
        self._buffer = np.zeros((self.config.seq_len, self.config.input_dim), dtype=np.float32)
        self._buffer_idx = 0
        self._buffer_full = False

        # Tracked tensors for defragmentation
        self._tracked_tensors: List[torch.Tensor] = []

        # Telemetry
        self._predictions: List[float] = []
        self._prediction_latencies: List[float] = []
        self._killed = False
        self._last_defrag_time = 0.0
        self._last_mem = 0
        self.pending_compaction = False
        self.last_predicted_score = 0.0

    def _load_model(self) -> None:
        """Load the predictor model."""
        import os
        if self._model is not None:
            return  # Already provided via DI

        if os.path.exists(self._model_path):
            self._model = FragPredictor.load(self._model_path, self.config)
            self._model.eval()
            log.info("Loaded predictor from %s (%d params)", self._model_path, self._model.count_parameters())
        else:
            try:
                import base64
                import io
                from rtx_oom_guard.scheduler.default_weights import DEFAULT_WEIGHTS_B64
                
                log.info("No checkpoint at %s — initializing with default pre-trained weights.", self._model_path)
                
                self._model = FragPredictor.from_config(self.config)
                weights_bytes = base64.b64decode(DEFAULT_WEIGHTS_B64)
                state_dict = torch.load(io.BytesIO(weights_bytes), map_location="cpu", weights_only=True)
                self._model.load_state_dict(state_dict)
                self._model.eval()
            except ImportError:
                log.warning("No checkpoint at %s and default weights not found — using untrained model.", self._model_path)
                self._model = FragPredictor.from_config(self.config)
                self._model.eval()
            except Exception as e:
                log.error("Failed to load default weights: %s. Using untrained model.", e)
                self._model = FragPredictor.from_config(self.config)
                self._model.eval()

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background monitor thread."""
        if self._active:
            log.warning("Monitor already running.")
            return

        self._load_model()
        self._active = True
        self._last_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="rtx_oom_guard-monitor")
        self._thread.start()
        log.info("Monitor started (threshold=%.2f, interval=%dms)", self.config.frag_threshold, self.config.monitor_interval_ms)

    def stop(self) -> "DefragMonitor":
        """Stop the background monitor thread. Returns self for chaining."""
        self._active = False
        if self._thread:
            self._thread.join(timeout=3.0)
        log.info("Monitor stopped. %d predictions, %d compactions.", len(self._predictions), len(self.compactor._history))
        return self

    def record_alloc(self, size_bytes: int, is_alloc: bool = True) -> None:
        """Record an allocation event (thread-safe)."""
        ts = time.perf_counter() * 1000
        with self._lock:
            reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 1
            allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            frag = 1.0 - (allocated / reserved) if reserved > 0 else 0.0

            self._buffer[self._buffer_idx] = [
                1.0 if is_alloc else 0.0,
                abs(size_bytes) / (1024**3),
                ts - getattr(self, "_last_ts", ts),
                frag,
            ]
            self._last_ts = ts
            self._buffer_idx = (self._buffer_idx + 1) % self.config.seq_len
            if self._buffer_idx == 0:
                self._buffer_full = True

    def auto_record(self) -> None:
        """Automatically record based on memory_allocated() delta."""
        if not torch.cuda.is_available():
            return
        current = torch.cuda.memory_allocated()
        delta = current - self._last_mem
        if delta != 0:
            self.record_alloc(abs(delta), is_alloc=(delta > 0))
            self._last_mem = current

    def register_tensors(self, tensors: List[torch.Tensor]) -> None:
        """Register tensors (e.g. model parameters) for defragmentation."""
        with self._lock:
            self._tracked_tensors = list(tensors)

    def _get_tracked_tensors(self) -> List[torch.Tensor]:
        """Return the currently tracked tensors."""
        with self._lock:
            return list(self._tracked_tensors)

    # ── Background Loop ───────────────────────────────────────────────────

    def _run_loop(self) -> None:
        interval = self.config.monitor_interval_ms / 1000.0

        while self._active:
            self.auto_record()

            if self._buffer_full and self._model is not None:
                self._predict_and_act()  # pragma: no cover

            time.sleep(interval)

    def _predict_and_act(self) -> None:
        # Cooldown check
        now = time.time()
        if now - self._last_defrag_time < self.config.cooldown_seconds:
            return

        # Snapshot parsing for deeper profiling (low-level allocator tracing)
        if hasattr(self.config, 'enable_snapshots') and self.config.enable_snapshots:
            snapshot_info = parse_memory_snapshot()
            real_frag = snapshot_info['frag_score']
            if real_frag > 0.8:
                log.debug("Snapshot indicates high real fragmentation: %.3f", real_frag)  # pragma: no cover

        # Build input tensor from ring buffer (in correct order)
        t0 = time.perf_counter()
        with self._lock:
            if self._buffer_idx == 0:
                seq = self._buffer.copy()
            else:
                seq = np.concatenate([  # pragma: no cover
                    self._buffer[self._buffer_idx:],
                    self._buffer[:self._buffer_idx],
                ])

        x = torch.from_numpy(seq).unsqueeze(0)  # (1, seq_len, input_dim)

        with torch.no_grad():
            score = self._model(x).item()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._predictions.append(score)
        self._prediction_latencies.append(elapsed_ms)

        # Kill switch
        if elapsed_ms > self.config.max_prediction_latency_ms:
            log.warning("Prediction latency %.1fms exceeds limit. Disabling monitor.", elapsed_ms)
            self._killed = True
            self._active = False
            return

        # Trigger compaction if above threshold
        if score > self.config.frag_threshold:
            self.last_predicted_score = score
            if self.config.ddp_sync or self.config.async_compaction:
                self.pending_compaction = True
            else:
                tracked = self._get_tracked_tensors()
                self.compactor.defragment_tensors(tracked, reason=f"predicted_frag={score:.3f}")
                self._last_defrag_time = time.time()

    # ── Telemetry ─────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return comprehensive monitor statistics."""
        return {
            "total_predictions": len(self._predictions),
            "total_compactions": len(self.compactor._history),
            "total_freed_mb": sum(h.get("freed_mb", 0.0) for h in self.compactor._history),
            "avg_prediction_score": np.mean(self._predictions) if self._predictions else 0,
            "max_prediction_score": max(self._predictions) if self._predictions else 0,
            "avg_latency_ms": np.mean(self._prediction_latencies) if self._prediction_latencies else 0,
            "max_latency_ms": max(self._prediction_latencies) if self._prediction_latencies else 0,
            "killed": self._killed,
            "compaction_history": self.compactor._history,
        }
