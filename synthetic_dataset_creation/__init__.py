"""
Synthetic Dataset Creation module.

This module provides functionality for generating synthetic Q&A datasets
for legal regulations (GDPR, AI Act) using LLMs.

Usage:
    python -m synthetic_dataset_creation

Submodules:
    - prompts: Prompt templates for Q&A generation
    - schemas: Pydantic validation schemas
    - config: Configuration dataclasses
    - pipeline: Async Q&A generation pipeline
    - utils: Utility functions
    - gdpr_to_markdown: GDPR markdown parsing utilities
"""

# Re-export from submodules
from .prompts import (
    PROMPT_ARTICLE_UNITY_Q,
    PROMPT_ARTICLE_UNITY_A,
    PROMPT_ARTICLE_RECITAL_BINDING_Q,
    PROMPT_ARTICLE_RECITAL_BINDING_A,
    PROMPT_RECITAL_UNITY_Q,
    PROMPT_RECITAL_UNITY_A,
    PROMPT_ANNEX_UNITY_Q,
    PROMPT_ANNEX_UNITY_A,
    PROMPT_ANNEX_RECITAL_BINDING_Q,
    PROMPT_ANNEX_RECITAL_BINDING_A,
    PROMPT_ANNEX_ARTICLE_BINDING_Q,
    PROMPT_ANNEX_ARTICLE_BINDING_A,
    PROMPT_AUGMENTATION_Q,
    ALL_PROMPTS,
)

from .schemas import (
    BaseQuestion,
    BaseAnswer,
    ArticleUnityQuestion,
    ArticleUnityAnswer,
    RecitalUnityQuestion,
    RecitalUnityAnswer,
    AnnexUnityQuestion,
    AnnexUnityAnswer,
    BindingArticleRecitalQuestion,
    BindingArticleRecitalAnswer,
    BindingAnnexArticleQuestion,
    BindingAnnexArticleAnswer,
    BindingAnnexRecitalQuestion,
    BindingAnnexRecitalAnswer,
    AugmentedVariants,
)

from .config import (
    UnityConfig,
    BindingConfig,
)

from .pipeline import (
    AsyncQAGDPRPipeline,
)

from .utils import (
    load_gdpr_graph,
    create_llama_cpp_func,
    truncate_dataset,
)

from .gdpr_to_markdown.build_gdpr_json import (
    parse_article,
    parse_recital,
    build_gdpr_json,
)

__all__ = [
    # Prompts
    "PROMPT_ARTICLE_UNITY_Q",
    "PROMPT_ARTICLE_UNITY_A",
    "PROMPT_ARTICLE_RECITAL_BINDING_Q",
    "PROMPT_ARTICLE_RECITAL_BINDING_A",
    "PROMPT_RECITAL_UNITY_Q",
    "PROMPT_RECITAL_UNITY_A",
    "PROMPT_ANNEX_UNITY_Q",
    "PROMPT_ANNEX_UNITY_A",
    "PROMPT_ANNEX_RECITAL_BINDING_Q",
    "PROMPT_ANNEX_RECITAL_BINDING_A",
    "PROMPT_ANNEX_ARTICLE_BINDING_Q",
    "PROMPT_ANNEX_ARTICLE_BINDING_A",
    "PROMPT_AUGMENTATION_Q",
    "ALL_PROMPTS",
    # Schemas
    "BaseQuestion",
    "BaseAnswer",
    "ArticleUnityQuestion",
    "ArticleUnityAnswer",
    "RecitalUnityQuestion",
    "RecitalUnityAnswer",
    "AnnexUnityQuestion",
    "AnnexUnityAnswer",
    "BindingArticleRecitalQuestion",
    "BindingArticleRecitalAnswer",
    "BindingAnnexArticleQuestion",
    "BindingAnnexArticleAnswer",
    "BindingAnnexRecitalQuestion",
    "BindingAnnexRecitalAnswer",
    "AugmentedVariants",
    # Config
    "UnityConfig",
    "BindingConfig",
    # Pipeline
    "AsyncQAGDPRPipeline",
    # Utils
    "load_gdpr_graph",
    "create_llama_cpp_func",
    "truncate_dataset",
    # GDPR Markdown
    "parse_article",
    "parse_recital",
    "build_gdpr_json",
]
