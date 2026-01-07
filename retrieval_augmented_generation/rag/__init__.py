"""RAG implementations and factory."""

from .base import AbstractRAG
from .basic import BasicRAG
from .factory import RAGFactory

# Try to import DatapizzaRAG (optional dependency)
try:
    from .datapizza import DatapizzaRAG
    __all__ = [
        "AbstractRAG",
        "BasicRAG",
        "DatapizzaRAG",
        "RAGFactory",
    ]
except ImportError:
    __all__ = [
        "AbstractRAG",
        "BasicRAG",
        "RAGFactory",
    ]
