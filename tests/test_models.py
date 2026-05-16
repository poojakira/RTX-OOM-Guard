import torch
import torch.nn as nn
from rtx_oom_guard.trainer._models import SimpleGPT2, build_gpt2, build_resnet50, build_bert

def test_simple_gpt2():
    """Verify SimpleGPT2 forward pass and output shape."""
    model = SimpleGPT2(vocab_size=100, d_model=32, n_layers=2, n_heads=4)
    x = torch.randint(0, 100, (2, 8))
    output = model(x)
    assert output.shape == (2, 8, 100)

def test_build_gpt2():
    """Verify GPT-2 builder."""
    model, inputs = build_gpt2(device='cpu', n_layers=2)
    assert isinstance(model, SimpleGPT2)
    assert inputs.device.type == 'cpu'
    output = model(inputs)
    assert output.shape[0] == inputs.shape[0]

def test_build_resnet50():
    """Verify ResNet-50 builder (handles torchvision fallback)."""
    model, inputs = build_resnet50(device='cpu')
    assert isinstance(model, nn.Module)
    assert inputs.shape == (16, 3, 224, 224)
    output = model(inputs)
    assert output.shape == (16, 1000)

def test_build_bert():
    """Verify BERT builder."""
    model, inputs = build_bert(device='cpu')
    assert isinstance(model, nn.Module)
    assert inputs.shape == (8, 512, 768)
    output = model(inputs)
    assert output.shape == (8, 512, 768)
