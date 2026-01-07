"""
Configuration settings for fine-tuning module.
"""

# Try to import psutil for CPU count
try:
    import psutil
    DATASET_NUM_PROC = max(psutil.cpu_count() + 4, 2)
except ImportError:
    psutil = None
    DATASET_NUM_PROC = 2

# =============================================================================
# Dataset Paths
# =============================================================================

AI_ACT_QA_DATASET = "data/AI ACT/datasets/dataset_truncated.json"
GDPR_QA_DATASET = "data/GDPR/datasets/dataset_truncated.jsonl"

# =============================================================================
# Model Constants
# =============================================================================

# 4bit pre-quantized models supported by Unsloth
FOURBIT_MODELS = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",
]

# LoRA target modules by model architecture
TARGET_MODULES_LLAMA = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
