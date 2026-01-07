"""Pydantic schemas for synthetic data generation."""

from .validation import (
    # Base classes
    BaseQuestion,
    BaseAnswer,
    # Unity schemas
    ArticleUnityQuestion,
    ArticleUnityAnswer,
    RecitalUnityQuestion,
    RecitalUnityAnswer,
    AnnexUnityQuestion,
    AnnexUnityAnswer,
    # Binding schemas
    BindingArticleRecitalQuestion,
    BindingArticleRecitalAnswer,
    BindingAnnexArticleQuestion,
    BindingAnnexArticleAnswer,
    BindingAnnexRecitalQuestion,
    BindingAnnexRecitalAnswer,
    # Augmentation
    AugmentedVariants,
)

__all__ = [
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
]
