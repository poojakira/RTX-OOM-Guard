import torch
import torch.nn as nn
from rtx_oom_guard.optimization.quantization import apply_gpu_quantization, get_model_size_mb

class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        return self.fc(x)

def test_gpu_quantization():
    model = SimpleLinear()
    orig_size = get_model_size_mb(model)

    quantized_model = apply_gpu_quantization(model, dtype=torch.float16)
    quantized_size = get_model_size_mb(quantized_model)

    assert quantized_model is not None
    assert isinstance(quantized_model, nn.Module)
    
    # Assert model parameter memory footprint actually dropped
    assert quantized_size < orig_size

    # Run an FP16 forward pass
    x = torch.randn(2, 512, dtype=torch.float16)
    out = quantized_model(x)
    assert out.shape == (2, 512)

def test_model_size_computation():
    model = SimpleLinear()
    size_mb = get_model_size_mb(model)
    # 512 * 512 * 4 bytes + 512 * 4 bytes = 1.05 MB
    assert 1.0 < size_mb < 1.1
