"""
Abstract base class for RAG implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class AbstractRAG(ABC):
    """Abstract base class for all RAG implementations."""

    @abstractmethod
    def load_documents(self, json_file: str, max_tokens: int = 128) -> None:
        """Load documents from JSON file."""
        pass

    @abstractmethod
    def embed_documents(self, savepath: Optional[str] = None) -> None:
        """Generate embeddings for all documents."""
        pass

    @abstractmethod
    def load_embeddings(self, *args, **kwargs) -> None:
        """Load pre-computed embeddings."""
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        topk: int = 20,
        threshold: float = 0.0,
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        pass

    @abstractmethod
    def build_graph(self) -> None:
        """Build knowledge graph from document relationships."""
        pass

    @abstractmethod
    def retrieve_with_graph(
        self,
        query: str,
        topk: int = 5,
        threshold: float = 0.6,
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve documents with graph-based expansion."""
        pass
