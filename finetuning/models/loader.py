"""
Model loading and LoRA adapter setup utilities.
"""

from typing import List, Any, Optional, Tuple

import torch
from unsloth import FastLanguageModel
from peft import PeftModel

from ..config import TARGET_MODULES_LLAMA


def load_model_and_tokenizer(
    model_name: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length: int = 2048,
    dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = True,
) -> Tuple[Any, Any]:
    """
    Load a pre-trained model and tokenizer using Unsloth's FastLanguageModel.

    Args:
        model_name: Name or path of the model to load
        max_seq_length: Maximum sequence length for the model
        dtype: Data type for model weights (None for auto-detection)
        load_in_4bit: Whether to load model in 4-bit quantization

    Returns:
        Tuple of (model, tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Configure tokenizer
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def setup_lora_adapters(
    model: Any,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0,
    target_modules: Optional[List[str]] = None,
    use_rslora: bool = True,
    random_state: int = 2704,
) -> Any:
    """
    Add LoRA adapters to the model for efficient fine-tuning.

    Args:
        model: The base model to add adapters to
        r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout rate for LoRA layers
        target_modules: List of module names to apply LoRA to
        use_rslora: Whether to use rsLoRA (recommended by Unsloth)
        random_state: Random seed for reproducibility

    Returns:
        Model with LoRA adapters attached
    """
    if target_modules is None:
        target_modules = TARGET_MODULES_LLAMA

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=random_state,
        use_rslora=use_rslora,
        loftq_config=None,
    )

    model.print_trainable_parameters()
    return model


def load_model_for_inference(
    base_model_name: str,
    lora_adapter_path: str,
) -> Tuple[Any, Any]:
    """
    Load a model with LoRA adapters for inference.

    Args:
        base_model_name: Name of the base model
        lora_adapter_path: Path to the LoRA adapters

    Returns:
        Tuple of (model, tokenizer)
    """
    base_model, tokenizer = FastLanguageModel.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
