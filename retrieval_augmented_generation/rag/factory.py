"""
Factory pattern for creating RAG instances.
"""

import logging
from typing import Any

from .base import AbstractRAG
from .basic import BasicRAG

logger = logging.getLogger(__name__)

# Try to import DatapizzaRAG
try:
    from .datapizza import DatapizzaRAG
    HAS_DATAPIZZA = True
except ImportError:
    HAS_DATAPIZZA = False
    logger.warning("DatapizzaRAG not available. Install datapizza-ai to use it.")


class RAGFactory:
    """Factory pattern for creating RAG instances."""

    @staticmethod
    def create(
        rag_type: str = "basic",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_cache: bool = False,
        embeddings_dir: str = "./embeddings",
        vectorstore_path: str = "./qdrant_db",
        domain: str = "generic",
        collection_name: str = "mydocuments",
        **kwargs
    ) -> AbstractRAG:
        """
        Factory method to create RAG instances.

        Args:
            rag_type: 'basic' for FAISS-based RAG, 'datapizza' for Qdrant-based RAG
            embedding_model: Model name for embeddings
            use_cache: Load cached embeddings if available
            embeddings_dir: Directory for cached embeddings
            vectorstore_path: Path for vector store (Datapizza only)
            domain: Domain name (GDPR, AI ACT, etc.)
            collection_name: Qdrant collection name (Datapizza only)

        Returns:
            RAG instance
            
        Raises:
            ValueError: If rag_type is unknown or dependencies are missing
        """
        if rag_type == "basic":
            return BasicRAG(embedding_model=embedding_model, domain=domain)
        
        elif rag_type == "datapizza":
            if not HAS_DATAPIZZA:
                raise ValueError(
                    "DatapizzaRAG is not available. "
                    "Install with: pip install datapizza-ai datapizza-ai-parsers-docling "
                    "datapizza-ai-embedders-fastembedder"
                )
            return DatapizzaRAG(
                embedding_model=embedding_model,
                vectorstore_path=vectorstore_path,
                domain=domain,
                collection_name=collection_name
            )
        
        else:
            available_types = ["basic"]
            if HAS_DATAPIZZA:
                available_types.append("datapizza")
            raise ValueError(
                f"Unknown RAG type: {rag_type}. "
                f"Available types: {', '.join(available_types)}"
            )
