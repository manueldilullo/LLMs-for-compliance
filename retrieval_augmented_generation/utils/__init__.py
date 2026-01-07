"""Utility functions for RAG system."""

from .helpers import (
    get_timestamp,
    load_dataset,
    sort_index_related_lists,
    chunk_text,
    load_documents_json,
)

from .caching import (
    compute_query_hash,
    compute_rag_hash,
    smart_retrieve,
    get_base_generation,
    load_checkpoint,
    save_checkpoint,
    compute_config_hash,
)

from .json_utils import (
    extract_json_from_text,
    _parse_json_with_repair,
)

from .llm import (
    call_llm_with_retry,
    askgenerator_raw,
    _join_context,
    init_llm,
)

from .initialization import (
    init_rag,
)

__all__ = [
    # Helpers
    "get_timestamp",
    "load_dataset",
    "sort_index_related_lists",
    "chunk_text",
    "load_documents_json",
    # Caching
    "compute_query_hash",
    "compute_rag_hash",
    "smart_retrieve",
    "get_base_generation",
    "load_checkpoint",
    "save_checkpoint",
    "compute_config_hash",
    # JSON Utils
    "extract_json_from_text",
    "_parse_json_with_repair",
    # LLM
    "call_llm_with_retry",
    "askgenerator_raw",
    "_join_context",
    "init_llm",
    # Initialization
    "init_rag",
]
