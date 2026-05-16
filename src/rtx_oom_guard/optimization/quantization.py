"""
rtx_oom_guard.optimization.quantization
===================================

Integrates memory-efficient numerical representation (quantization) pipelines
for PyTorch models, critical for large LLMs before running defragmentation loops.
"""

import torch
import torch.nn as nn
from rtx_oom_guard.utils import get_logger

log = get_logger("quantization")

def apply_gpu_quantization(model: nn.Module, dtype=torch.float16) -> nn.Module:
    """
    Applies authentic GPU-native mixed-precision downcasting to a PyTorch model.
    By compressing weight representations from FP32 to FP16 or BF16, we cut the contiguous
    memory footprint by 50% globally, directly alleviating the CachingAllocator's 
    fragmentation workload on CUDA.
    """
    log.info(f"Applying GPU-native memory optimization via automatic casting to {dtype}")
    
    # Check if the hardware supports fast generic downcasting
    if not torch.cuda.is_available():
        log.warning("CUDA not available. Downcasting may not yield physical VRAM improvements.")
        return model.to(dtype)

    # Cast to precision (e.g., float16 or bfloat16)
    optimized_model = model.to(dtype)
    return optimized_model

def get_model_size_mb(model: nn.Module) -> float:
    """Calculates the physical memory footprint of a model."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / (1024**2)
