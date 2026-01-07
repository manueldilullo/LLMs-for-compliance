"""
Pydantic validation schemas for RAG system.
"""

from pydantic import BaseModel


class BaseAnswer(BaseModel):
    """Base class for answer schemas."""
    answer: str
