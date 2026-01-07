"""
Pydantic validation schemas for synthetic data generation.
"""

from typing import List
from pydantic import BaseModel, Field


# =============================================================================
# Base Classes
# =============================================================================

class BaseQuestion(BaseModel):
    """Base class for all question schemas."""
    question: str


class BaseAnswer(BaseModel):
    """Base class for all answer schemas."""
    answer: str


# =============================================================================
# Unity Schemas (Single Entity)
# =============================================================================

class ArticleUnityQuestion(BaseQuestion):
    """Question schema for article unity generation."""
    article_number: int


class ArticleUnityAnswer(BaseAnswer):
    """Answer schema for article unity generation."""
    article_number: int


class RecitalUnityQuestion(BaseQuestion):
    """Question schema for recital unity generation."""
    recital_number: int


class RecitalUnityAnswer(BaseAnswer):
    """Answer schema for recital unity generation."""
    recital_number: int


class AnnexUnityQuestion(BaseQuestion):
    """Question schema for annex unity generation."""
    annex_number: int


class AnnexUnityAnswer(BaseAnswer):
    """Answer schema for annex unity generation."""
    annex_number: int


# =============================================================================
# Binding Schemas (Two Entities)
# =============================================================================

class BindingArticleRecitalQuestion(BaseQuestion):
    """Question schema for article-recital binding generation."""
    recital_number: int
    article_number: int


class BindingArticleRecitalAnswer(BaseAnswer):
    """Answer schema for article-recital binding generation."""
    recital_number: int
    article_number: int


class BindingAnnexArticleQuestion(BaseQuestion):
    """Question schema for annex-article binding generation."""
    annex_number: int
    article_number: int


class BindingAnnexArticleAnswer(BaseAnswer):
    """Answer schema for annex-article binding generation."""
    annex_number: int
    article_number: int


class BindingAnnexRecitalQuestion(BaseQuestion):
    """Question schema for annex-recital binding generation."""
    annex_number: int
    recital_number: int


class BindingAnnexRecitalAnswer(BaseAnswer):
    """Answer schema for annex-recital binding generation."""
    annex_number: int
    recital_number: int


# =============================================================================
# Augmentation Schema
# =============================================================================

class AugmentedVariants(BaseModel):
    """Schema for augmented question variants."""
    questions: List[str] = Field(default_factory=list)
