"""
rtx_oom_guard._models — Model factories for trace collection and benchmarking.

These are lightweight model builders used internally for data collection
and benchmarking. They create standard architectures (GPT-2, ResNet-50, BERT)
with appropriate input tensors.
"""

import torch
import torch.nn as nn
from typing import Tuple


class SimpleGPT2(nn.Module):
    """Lightweight GPT-2-style Transformer for benchmarking."""

    def __init__(self, vocab_size: int = 50257, d_model: int = 768, n_layers: int = 12, n_heads: int = 12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.head(x)


def build_gpt2(device: str = "cuda", n_layers: int = 6) -> Tuple[nn.Module, torch.Tensor]:
    """Build a GPT-2 model and sample input."""
    model = SimpleGPT2(n_layers=n_layers).to(device)
    inputs = torch.randint(0, 50257, (8, 512), device=device)
    return model, inputs


def build_resnet50(device: str = "cuda") -> Tuple[nn.Module, torch.Tensor]:
    """Build ResNet-50 and sample input."""
    try:
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=None).to(device)  # pragma: no cover
    except ImportError:
        # Fallback: simple CNN if torchvision not installed
        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1000),
        ).to(device)
    inputs = torch.randn(16, 3, 224, 224, device=device)
    return model, inputs


def build_bert(device: str = "cuda") -> Tuple[nn.Module, torch.Tensor]:
    """Build BERT-like encoder and sample input."""
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        ),
        num_layers=6,
    ).to(device)
    inputs = torch.randn(8, 512, 768, device=device)
    return model, inputs
