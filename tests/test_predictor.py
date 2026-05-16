"""Unit tests for the FragPredictor model."""

import torch

from rtx_oom_guard.predictor.model import FragPredictor
from rtx_oom_guard.utils import DefragConfig


class TestFragPredictor:
    def test_output_shape(self):
        model = FragPredictor(input_dim=4, hidden_dim=64, n_layers=2, n_heads=2, seq_len=64)
        x = torch.randn(8, 64, 4)
        out = model(x)
        assert out.shape == (8, 1)

    def test_output_range(self):
        model = FragPredictor(input_dim=4, hidden_dim=64, n_layers=2, n_heads=2, seq_len=64)
        x = torch.randn(16, 64, 4)
        out = model(x)
        assert (out >= 0).all() and (out <= 1).all(), "Output must be in [0, 1]"

    def test_from_config(self):
        config = DefragConfig(input_dim=4, hidden_dim=64, n_layers=2, n_heads=2, seq_len=32)
        model = FragPredictor.from_config(config)
        x = torch.randn(4, 32, 4)
        out = model(x)
        assert out.shape == (4, 1)

    def test_save_load(self, tmp_path):
        model = FragPredictor(input_dim=4, hidden_dim=64, n_layers=2, n_heads=2, seq_len=64)
        model.eval()
        path = str(tmp_path / "test_model.pt")
        model.save(path)
        loaded = FragPredictor.load(path, DefragConfig(hidden_dim=64, n_layers=2, n_heads=2))
        loaded.eval()
        x = torch.randn(1, 64, 4)
        with torch.no_grad():
            assert torch.allclose(model(x), loaded(x), atol=1e-3)

    def test_parameter_count(self):
        model = FragPredictor(input_dim=4, hidden_dim=128, n_layers=4, n_heads=4, seq_len=64)
        assert model.count_parameters() > 0
        assert model.count_parameters() < 5_000_000  # Reasonable size

    def test_gradient_flow(self):
        model = FragPredictor(input_dim=4, hidden_dim=64, n_layers=2, n_heads=2, seq_len=64)
        x = torch.randn(4, 64, 4, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
