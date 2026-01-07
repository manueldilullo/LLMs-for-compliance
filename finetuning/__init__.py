"""
Fine-tuning LLM with Unsloth

This module provides functionality for fine-tuning language models using the Unsloth
library with LoRA adapters for efficient training on legal Q&A datasets.

Usage:
    python -m finetuning

Submodules:
    - config: Configuration settings and constants
    - models: Model loading and LoRA setup utilities
    - datasets: Dataset loading and preparation
    - training: Training utilities and trainer creation
    - evaluation: Evaluation metrics and prediction extraction
    - utils: Utility functions (GPU helpers, etc.)
"""

# Re-export from submodules for convenient access
from .config import (
    AI_ACT_QA_DATASET,
    GDPR_QA_DATASET,
    FOURBIT_MODELS,
    TARGET_MODULES_LLAMA,
)

from .models import (
    load_model_and_tokenizer,
    setup_lora_adapters,
    load_model_for_inference,
)

from .datasets import (
    to_chat_example,
    load_qa_dataset,
    formatting_prompts_func,
    prepare_datasets,
)

from .training import (
    create_trainer,
    save_model,
)

from .evaluation import (
    safe_extract_predictions,
    compute_metrics,
)

from .utils import (
    clear_gpu_memory,
    print_gpu_info,
)

__all__ = [
    # Config
    "AI_ACT_QA_DATASET",
    "GDPR_QA_DATASET",
    "FOURBIT_MODELS",
    "TARGET_MODULES_LLAMA",
    # Models
    "load_model_and_tokenizer",
    "setup_lora_adapters",
    "load_model_for_inference",
    # Datasets
    "to_chat_example",
    "load_qa_dataset",
    "formatting_prompts_func",
    "prepare_datasets",
    # Training
    "create_trainer",
    "save_model",
    # Evaluation
    "safe_extract_predictions",
    "compute_metrics",
    # Utils
    "clear_gpu_memory",
    "print_gpu_info",
]
