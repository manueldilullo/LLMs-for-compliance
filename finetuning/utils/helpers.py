"""
Utility helper functions for fine-tuning.
"""

import gc
import torch


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_gpu_info() -> None:
    """Print GPU information for debugging."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
