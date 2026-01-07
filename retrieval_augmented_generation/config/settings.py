"""
Configuration settings for RAG system.
"""

import os
from dataclasses import dataclass


# =============================================================================
# Constants
# =============================================================================

GRAPH_FILENAMES = {
    "GDPR": "gdpr_w_annexes.json",
    "AI ACT": "ai-act.json"
}

EMBEDDING_MODELS = {
    "fast": "sentence-transformers/all-mpnet-base-v2",
    "accurate": "intfloat/e5-base-v2",
    "best": "intfloat/multilingual-e5-large-instruct",
    "bge-large": "BAAI/bge-base-en-v1.5",
    "bge": "BAAI/llm-embedder"
}

MODEL_DICT = {
    "DeepSeek-R1-Distill-Qwen-1.5B-GGUF": {
        "HF_REPO_NAME": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "HF_MODEL_NAME": "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"
    },
    "Qwen2.5-3B": {
        "HF_REPO_NAME": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "HF_MODEL_NAME": "qwen2.5-3b-instruct-q8_0.gguf"
    },
    "gpt-oss-20b": {
        "HF_REPO_NAME": "unsloth/gpt-oss-20b-GGUF",
        "HF_MODEL_NAME": "gpt-oss-20b-F16.gguf"
    },
    "Llama-3.1-8B-Instruct": {
        "HF_REPO_NAME": "unsloth/Llama-3.1-8B-Instruct-GGUF",
        "HF_MODEL_NAME": "Llama-3.1-8B-Instruct-BF16.gguf"
    },
    "Llama-3.2-3B-Instruct": {
        "HF_REPO_NAME": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "HF_MODEL_NAME": "Llama-3.2-3B-Instruct-f16.gguf"
    }
}


# =============================================================================
# Configuration Class
# =============================================================================

@dataclass
class Config:
    """Configuration for RAG system."""
    DOMAIN: str
    GRAPH_FILENAME: str
    GRAPH_FILEPATH: str
    DATASET_FILENAME: str
    DATASET_JSONL: str
    SHUFFLED_DATASET: str
    EMBEDDING_MODEL: str
    EMBEDDINGS_DIR: str
    RETRIEVAL_RESULTS_DIR: str
    TOPK: int
    USE_CACHE: bool
    MAX_TOKENS: int
    GENERATOR_RESULTS_DIR: str

    def __init__(
        self,
        domain: str,
        selected_emb: str,
        graph_filename: str,
        max_tokens: int = 256,
        use_cache: bool = False,
        topk: int = 20
    ):
        self.DOMAIN = domain
        self.GRAPH_FILENAME = graph_filename
        self.GRAPH_FILEPATH = os.path.join(domain, "datasets", graph_filename)

        self.DATASET_FILENAME = "dataset_truncated.jsonl" if domain == "GDPR" else "dataset_truncated.json"
        self.DATASET_JSONL = os.path.join(domain, "datasets", self.DATASET_FILENAME)
        self.SHUFFLED_DATASET = os.path.join(domain, "datasets", "shuffled_" + self.DATASET_FILENAME)

        self.EMBEDDING_MODEL = EMBEDDING_MODELS.get(selected_emb, selected_emb)
        self.EMBEDDINGS_DIR = os.path.join(domain, "embeddings", selected_emb)
        self.RETRIEVAL_RESULTS_DIR = os.path.join(domain, "retrieval_results", selected_emb)

        self.TOPK = topk
        self.USE_CACHE = use_cache
        self.MAX_TOKENS = max_tokens

        self.GENERATOR_RESULTS_DIR = os.path.join(domain, "generator_results")
