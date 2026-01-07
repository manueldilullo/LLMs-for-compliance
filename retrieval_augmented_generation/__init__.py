"""
Retrieval Augmented Generation (RAG) module.

This module provides RAG systems for legal Q&A including:
- Basic FAISS-based RAG
- Graph-enhanced RAG
- Agentic patterns (routing, collaboration, self-refinement)
- Evaluation metrics and pipelines

Usage:
    python -m retrieval_augmented_generation

Submodules:
    - config: Configuration settings (Config, GRAPH_FILENAMES, EMBEDDING_MODELS)
    - prompts: Prompt templates for RAG patterns
    - schemas: Pydantic validation schemas
    - rag: RAG implementations (AbstractRAG, BasicRAG, RAGFactory)
    - patterns: Agentic patterns (baseline, rag, graph, routing, etc.)
    - evaluation: Evaluation metrics and retrieval evaluation
    - utils: Utility functions (caching, JSON parsing, LLM calls, etc.)
"""

# Re-export from submodules
from .config import (
    GRAPH_FILENAMES,
    EMBEDDING_MODELS,
    MODEL_DICT,
    Config,
)

from .prompts import (
    INSTRUCTIONS_ANSWER_JSON,
    RAG_PROMPT,
    BASELINE_PROMPT,
    SELF_REFINEMENT_PROMPT,
    REFINE_FROM_ISSUES_PROMPT,
    INSTRUCTIONS_CRITIQUE_JSON,
    CRITIC_PROMPT,
)

from .schemas import (
    BaseAnswer,
)

# Import RAG implementations
try:
    from .rag import (
        AbstractRAG,
        BasicRAG,
        DatapizzaRAG,
        RAGFactory,
    )
    HAS_DATAPIZZA_RAG = True
except ImportError:
    from .rag import (
        AbstractRAG,
        BasicRAG,
        RAGFactory,
    )
    HAS_DATAPIZZA_RAG = False

from .patterns import (
    baseline_pattern,
    rag_pattern,
    rag_with_graph_pattern,
    routing_rag_pattern,
    collaboration_rag_pattern,
    self_refinement_rag_pattern,
)

from .evaluation import (
    compute_em,
    compute_f1,
    compute_all_metrics,
    parse_refid_to_ground_truth,
    extract_id_from_doc,
    create_retrieval_cache,
    calculate_retrieval,
)

from .utils import (
    get_timestamp,
    load_dataset,
    sort_index_related_lists,
    chunk_text,
    load_documents_json,
    compute_query_hash,
    compute_rag_hash,
    smart_retrieve,
    get_base_generation,
    load_checkpoint,
    save_checkpoint,
    compute_config_hash,
    extract_json_from_text,
    call_llm_with_retry,
    askgenerator_raw,
    init_rag,
)

# Build __all__ list dynamically
__all__ = [
    # Config
    "GRAPH_FILENAMES",
    "EMBEDDING_MODELS",
    "MODEL_DICT",
    "Config",
    # Prompts
    "INSTRUCTIONS_ANSWER_JSON",
    "RAG_PROMPT",
    "BASELINE_PROMPT",
    "SELF_REFINEMENT_PROMPT",
    "REFINE_FROM_ISSUES_PROMPT",
    "INSTRUCTIONS_CRITIQUE_JSON",
    "CRITIC_PROMPT",
    # Schemas
    "BaseAnswer",
    # RAG
    "AbstractRAG",
    "BasicRAG",
    "RAGFactory",
    # Patterns
    "baseline_pattern",
    "rag_pattern",
    "rag_with_graph_pattern",
    "routing_rag_pattern",
    "collaboration_rag_pattern",
    "self_refinement_rag_pattern",
    # Evaluation
    "compute_em",
    "compute_f1",
    "compute_all_metrics",
    "parse_refid_to_ground_truth",
    "extract_id_from_doc",
    "create_retrieval_cache",
    "calculate_retrieval",
    # Utils
    "get_timestamp",
    "load_dataset",
    "sort_index_related_lists",
    "chunk_text",
    "load_documents_json",
    "compute_query_hash",
    "compute_rag_hash",
    "smart_retrieve",
    "get_base_generation",
    "load_checkpoint",
    "save_checkpoint",
    "compute_config_hash",
    "extract_json_from_text",
    "call_llm_with_retry",
    "askgenerator_raw",
    "init_rag",
]

# Add DatapizzaRAG to exports if available
if HAS_DATAPIZZA_RAG:
    __all__.append("DatapizzaRAG")
