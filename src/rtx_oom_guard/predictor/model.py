"""
rtx_oom_guard.predictor — Transformer-based GPU memory fragmentation predictor.

Architecture:
    - Input projection: (input_dim) → (hidden_dim)
    - Learnable positional encoding: (seq_len, hidden_dim)
    - N-layer Transformer Encoder with pre-norm
    - Regression head: (hidden_dim) → 1 (sigmoid → [0,1] frag score)

The model processes a sliding window of the last `seq_len` allocation events
and outputs a scalar fragmentation score predicting how fragmented memory
will be in the near future.
"""

import torch
import torch.nn as nn
from typing import Optional
from rtx_oom_guard.utils import DefragConfig


class FragPredictor(nn.Module):
    """
    Lightweight Transformer encoder for memory fragmentation prediction.

    Input:  (batch, seq_len, input_dim) — allocation event features
    Output: (batch, 1)                  — predicted fragmentation score ∈ [0, 1]
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        seq_len: int = 64,
        dropout: float = 0.1,
    ):
        """
        Initialization for the FragPredictor.

        Design Choice:
        Prefer Transformer Encoder over LSTM to capture non-linear, multi-scale 
        allocation patterns from long-tail history, as global attention is key 
        to identifying periodic OOM-triggering "bursts" that recurrence often misses.
        """
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers, 
            enable_nested_tensor=False
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        # Regression head with residual connection
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for better convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) — allocation event features

        Returns:
            (batch, 1) — predicted fragmentation score
        """
        B, S, _ = x.shape
        assert S == self.seq_len, f"Expected seq_len={self.seq_len}, got {S}"

        x = self.input_proj(x) + self.pos_encoding[:, :S, :]
        x = self.encoder(x)
        x = self.encoder_norm(x)

        # Utilize Global Average Pooling across the temporal axis to capture steady historical trends
        global_avg = x.mean(dim=1)
        return self.head(global_avg)

    @classmethod
    def from_config(cls, config: DefragConfig) -> "FragPredictor":
        """Create a predictor from a DefragConfig."""
        return cls(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            seq_len=config.seq_len,
        )

    @classmethod
    def load(cls, path: str, config: Optional[DefragConfig] = None, device: str = "cpu") -> "FragPredictor":
        """Load a trained model from checkpoint.

        Security note: ``weights_only=True`` is set to prevent arbitrary code
        execution via pickle deserialization (semgrep: python.lang.security.audit.pickle).
        Only load checkpoints from trusted sources.
        """
        config = config or DefragConfig()
        model = cls.from_config(config)
        # weights_only=True prevents pickle-based code execution (CVE-class: insecure deserialization)
        state = torch.load(path, map_location=device, weights_only=True)  # noqa: S614
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    def save(self, path: str) -> None:
        """Save model state-dict checkpoint.

        Security note: ``torch.save`` uses pickle internally. Only load the
        resulting file with ``weights_only=True`` (see :meth:`load`).
        """
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Saves state_dict only (not the full model object) to limit pickle surface
        torch.save(self.state_dict(), path)  # noqa: S614

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
