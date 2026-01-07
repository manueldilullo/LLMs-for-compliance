"""
Datapizza RAG implementation using Qdrant vector store.

This module provides a RAG implementation using the Datapizza AI framework
with Qdrant as the vector store backend.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional

import numpy as np

from .base import AbstractRAG

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False
    logger.warning("fastembed not installed. Install with: pip install fastembed")

try:
    from datapizza.core.vectorstore import VectorConfig
    from datapizza.embedders.fastembedder import FastEmbedder
    from datapizza.vectorstores.qdrant import QdrantVectorstore
    from datapizza.type import Chunk, DenseEmbedding, SparseEmbedding
    from qdrant_client import QdrantClient
    HAS_DATAPIZZA = True
except ImportError:
    HAS_DATAPIZZA = False
    logger.warning(
        "Datapizza libraries not installed. "
        "Install with: pip install datapizza-ai datapizza-ai-parsers-docling datapizza-ai-embedders-fastembedder"
    )

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.warning("NetworkX not installed. Graph features will be disabled.")


class DatapizzaRAG(AbstractRAG):
    """
    Datapizza implementation with Qdrant vector store.
    
    This RAG uses the Datapizza AI framework with Qdrant for vector storage,
    providing efficient semantic search with optional graph-based enhancement.
    
    Args:
        embedding_model: Name of the embedding model to use
        vectorstore_path: Path for Qdrant storage (use ":memory:" for in-memory)
        domain: Domain identifier (e.g., "GDPR", "AI ACT")
        collection_name: Name of the Qdrant collection
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        vectorstore_path: str = "./qdrant_db",
        domain: str = "generic",
        collection_name: str = "mydocuments"
    ):
        if not HAS_DATAPIZZA:
            raise ImportError(
                "Datapizza libraries not installed. "
                "Install with: pip install datapizza-ai datapizza-ai-parsers-docling "
                "datapizza-ai-embedders-fastembedder"
            )
        
        if not HAS_FASTEMBED:
            raise ImportError(
                "FastEmbed not installed. Install with: pip install fastembed"
            )

        self.embedding_model = embedding_model
        self.vectorstore_path = vectorstore_path
        self.collection_name = collection_name
        self.documents: List[Dict[str, Any]] = []
        self.model = None
        self.vectorstore = None
        self.graph = None if HAS_NETWORKX else None
        self.domain = domain
        
        self._setup_datapizza()

    def _setup_datapizza(self):
        """Initialize Datapizza components."""
        logger.info(f"Initializing Datapizza with model: {self.embedding_model}")
        
        # Initialize FastEmbed model
        self.model = TextEmbedding(model_name=self.embedding_model)
        
        # Initialize Qdrant vectorstore
        # Use in-memory for faster access, or persist with path
        if self.vectorstore_path == ":memory:":
            self.vectorstore = QdrantVectorstore(location=":memory:")
        else:
            self.vectorstore = QdrantVectorstore(path=self.vectorstore_path)
        
        # Create collection with proper vector configuration
        self.vectorstore.create_collection(
            self.collection_name,
            vector_config=[VectorConfig(name="dense_embeddings", dimensions=768)]
        )
        
        logger.info(f"Datapizza initialized with collection: {self.collection_name}")

    def load_documents(self, json_file: str, max_tokens: int = 128) -> None:
        """
        Load documents from JSON file.
        
        Args:
            json_file: Path to JSON file containing documents
            max_tokens: Maximum tokens per document (not used in Datapizza)
        """
        from ..utils import load_documents_json
        
        self.documents = load_documents_json(json_file)
        logger.info(f"Loaded {len(self.documents)} documents from {json_file}")

    def embed_documents(self, savepath: Optional[str] = None) -> None:
        """
        Embed documents and add to Qdrant vectorstore.
        
        Args:
            savepath: Not used for Datapizza (uses Qdrant persistence)
        """
        if not self.documents:
            raise ValueError("Load documents first!")

        logger.info("Embedding documents with Datapizza...")
        chunks = []
        
        for doc in self.documents:
            if not doc['text'].strip():
                continue
            
            # Generate embeddings
            embeddings = list(self.model.embed([doc['text']]))
            embedding_array = embeddings[0]  # numpy.ndarray
            
            # Create dense embedding
            dense_embedding = DenseEmbedding(
                name="dense_embeddings",
                vector=embedding_array
            )

            # Create chunk with metadata
            chunk = Chunk(
                id=str(uuid.uuid4()),
                text=doc['text'],
                embeddings=[dense_embedding],
                metadata={
                    'id': doc['id'],
                    'type': doc['type'],
                    'source': self.domain,
                    'number': doc['number'],
                    'title': doc.get('title', ''),
                    'related_annexes': doc.get('relatedAnnexes', []),
                    'related_articles': doc.get('relatedArticles', []),
                    'related_recitals': doc.get('relatedRecitals', [])
                }
            )
            chunks.append(chunk)

        # Add to vectorstore
        self.vectorstore.add(chunks, collection_name=self.collection_name)
        logger.info(f"Added {len(chunks)} chunks to Datapizza vectorstore")

    def load_embeddings(self, *args, **kwargs) -> None:
        """
        Load embeddings from persistent storage.
        
        For Datapizza with Qdrant, embeddings are loaded automatically
        from the persistent storage path.
        """
        logger.info("Datapizza embeddings loaded from persistent storage")

    def retrieve(
        self,
        query: str,
        topk: int = 20,
        threshold: float = 0.0,
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using semantic search.
        
        Args:
            query: Search query
            topk: Number of documents to retrieve
            threshold: Minimum similarity score
            query_embedding: Pre-computed query embedding (optional)
            
        Returns:
            List of retrieved documents with scores
        """
        if query_embedding is None:
            query_embeddings = list(self.model.embed([query]))
            query_embedding = query_embeddings[0]

        # Get native Qdrant client from Datapizza
        qdrant_client = self.vectorstore.client

        # Native search WITH SCORES
        hits = qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            using="dense_embeddings",
            limit=topk,
            with_payload=True,
            with_vectors=False
        )

        results = []
        for hit in hits.points:
            score = hit.score
            if score < threshold:
                continue

            results.append({
                'id': hit.payload.get('id', 'unknown'),
                'text': hit.payload.get('text', ''),
                'score': float(score),
                'type': hit.payload.get('type', 'unknown')
            })
        
        # Pad results if needed
        while len(results) < topk:
            results.append({
                'id': 'PAD',
                'text': '',
                'score': -1.0,
                'type': 'pad'
            })
        
        return results[:topk]

    def build_graph(self) -> None:
        """
        Build knowledge graph from document relationships.
        
        For Datapizza, graph relationships are stored in metadata.
        """
        if not HAS_NETWORKX:
            logger.warning("NetworkX not available, skipping graph construction")
            return
        
        logger.info("Building graph from document metadata...")
        self.graph = nx.DiGraph()
        
        # Add nodes
        for doc in self.documents:
            self.graph.add_node(
                doc['id'],
                type=doc['type'],
                text=doc['text']
            )
        
        # Add edges based on relationships
        for doc in self.documents:
            source_id = doc['id']
            
            if doc['type'] == 'article':
                related_recitals = doc.get('relatedRecitals', [])
                for rnum in related_recitals:
                    target_id = f"Recital_{rnum}_chunk0"
                    if self.graph.has_node(target_id):
                        self.graph.add_edge(
                            source_id, target_id,
                            relation='related_recital'
                        )
            
            elif doc['type'] == 'annex':
                related_articles = doc.get('relatedArticles', [])
                for anum in related_articles:
                    target_id = f"Article_{anum}_chunk0"
                    if self.graph.has_node(target_id):
                        self.graph.add_edge(
                            source_id, target_id,
                            relation='related_article'
                        )
                
                related_recitals = doc.get('relatedRecitals', [])
                for rnum in related_recitals:
                    target_id = f"Recital_{rnum}_chunk0"
                    if self.graph.has_node(target_id):
                        self.graph.add_edge(
                            source_id, target_id,
                            relation='related_recital'
                        )
        
        logger.info(
            f"Built graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    def retrieve_with_graph(
        self,
        query: str,
        topk: int = 5,
        threshold: float = 0.6,
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with graph-based enhancement.
        
        For Datapizza, this retrieves more documents and filters based on
        metadata relationships.
        
        Args:
            query: Search query
            topk: Number of documents to retrieve
            threshold: Minimum similarity score
            query_embedding: Pre-computed query embedding (optional)
            
        Returns:
            List of retrieved documents with graph relationships
        """
        # Retrieve more documents than needed for graph expansion
        initial_topk = topk * 2
        retrieved_docs = self.retrieve(
            query=query,
            topk=initial_topk,
            threshold=threshold,
            query_embedding=query_embedding
        )
        
        # Filter out padding
        retrieved_docs = [d for d in retrieved_docs if d.get("id") != "PAD"]
        
        final_results = []
        seen_ids = set()

        def add_doc(doc_obj: Dict[str, Any], is_graph_result: bool = False):
            """Helper to add document avoiding duplicates."""
            if doc_obj['id'] not in seen_ids:
                seen_ids.add(doc_obj['id'])
                if is_graph_result:
                    doc_obj['graph_related'] = True
                final_results.append(doc_obj)

        # Add retrieved documents and their graph neighbors
        for doc in retrieved_docs:
            add_doc(doc)
            
            # Expand via graph if available
            if doc['type'] in ['annex', 'article'] and self.graph and self.graph.has_node(doc['id']):
                neighbors = list(self.graph.neighbors(doc['id']))
                for neighbor_id in neighbors:
                    neighbor_doc = next(
                        (d for d in self.documents if d['id'] == neighbor_id),
                        None
                    )
                    if neighbor_doc:
                        neighbor_result = {
                            'id': neighbor_doc['id'],
                            'text': neighbor_doc['text'],
                            'score': doc['score'],  # Inherit score from parent
                            'type': neighbor_doc['type']
                        }
                        add_doc(neighbor_result, is_graph_result=True)

        # Return top-k results or pad if needed
        if not final_results:
            return [{'id': 'PAD', 'text': '', 'score': -1.0, 'type': 'pad'}] * topk
        
        return final_results[:topk]
