"""
rtx_oom_guard.utils — Shared utilities, logging, and configuration.
"""

import logging
import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a consistently formatted logger."""
    logger = logging.getLogger(f"rtx_oom_guard.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger



@dataclass
class DefragConfig:
    """Central configuration for the defrag system."""

    # Collector
    poll_interval_ms: float = 0.1          # Sub-ms polling
    max_events: int = 200_000              # Circular buffer cap

    # Predictor
    seq_len: int = 64                       # Input sequence length
    hidden_dim: int = 128                   # Transformer hidden dim
    n_layers: int = 4                       # Transformer layers
    n_heads: int = 4                        # Attention heads
    input_dim: int = 4                      # Features per event

    # Monitor
    monitor_interval_ms: float = 50.0       # Prediction frequency
    frag_threshold: float = 0.7             # Trigger threshold
    cooldown_seconds: float = 1.0           # Min time between defrags
    max_prediction_latency_ms: float = 5.0  # Kill switch

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64
    train_epochs: int = 20
    train_split: float = 0.8

    # Systems & Infra
    ddp_sync: bool = False                  # Coordinate compactions across DDP ranks
    async_compaction: bool = False          # Use async memory operations where supported

    # Paths
    trace_dir: str = "data/traces"
    checkpoint_path: str = "checkpoints/predictor.pt"
    results_dir: str = "results"

    def save(self, path: str) -> None:
        """Save config to JSON or YAML."""
        ext = Path(path).suffix.lower()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            if ext == ".yaml" or ext == ".yml":
                yaml.safe_dump(asdict(self), f, sort_keys=False)
            else:
                json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | None = None) -> "DefragConfig":
        """Load config from file (JSON/YAML) or return defaults."""
        if path is None or not Path(path).exists():
            return cls()

        try:
            ext = Path(path).suffix.lower()
            with open(path) as f:
                if ext == ".yaml" or ext == ".yml":
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            # Filter keys to match dataclass fields
            from dataclasses import fields
            valid_keys = {f.name for f in fields(cls)}
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            return cls(**filtered)
        except Exception as e:
            get_logger("config").warning(f"Failed to load config from {path}: {e}. Using defaults.")
            return cls()



class Timer:
    """High-resolution timer for benchmarking."""

    def __init__(self):
        self._start: float = 0
        self._elapsed: float = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._elapsed = time.perf_counter() - self._start

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed * 1000

    @property
    def elapsed_s(self) -> float:
        return self._elapsed



def get_cuda_info() -> dict:
    """Return comprehensive CUDA info or None if unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_mem / (1024**3),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def ensure_cuda():
    """Raise RuntimeError if CUDA is not available."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. rtx_oom_guard requires an NVIDIA GPU with CUDA support. "
            f"torch.cuda.is_available() = False, torch version = {torch.__version__}"
        )

def parse_memory_snapshot() -> dict:
    """
    Parses PyTorch's memory snapshot to extract low-level block information.
    Returns a list of segment sizes and a boolean indicating whether it is free or allocated.
    """
    import torch
    if not torch.cuda.is_available():
        return {"blocks": [], "frag_score": 0.0}

    try:
        snapshot = torch.cuda.memory_snapshot()
    except Exception:
        return {"blocks": [], "frag_score": 0.0}

    blocks = []
    total_free = 0
    total_allocated = 0
    largest_free = 0

    for segment in snapshot:
        for block in segment['blocks']:
            size = block['size']
            state = block['state']
            if state == 'active_allocated':
                blocks.append({'size': size, 'free': False})
                total_allocated += size
            elif state == 'inactive': # free block in the allocator pool
                blocks.append({'size': size, 'free': True})
                total_free += size
                largest_free = max(largest_free, size)

    if total_free == 0:
        frag_score = 0.0  # pragma: no cover
    else:
        frag_score = 1.0 - (largest_free / total_free)

    return {
        "blocks": blocks,
        "frag_score": frag_score,
        "total_free": total_free,
        "total_allocated": total_allocated
    }
