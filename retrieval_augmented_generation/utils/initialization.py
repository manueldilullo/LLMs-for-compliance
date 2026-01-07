"""
RAG initialization utilities.
"""

from ..config import Config
from ..rag import AbstractRAG, RAGFactory


def init_rag(config: Config, rag_type: str = "basic") -> AbstractRAG:
    """Initialize RAG system from configuration."""
    rag = RAGFactory.create(
        rag_type=rag_type,
        embedding_model=config.EMBEDDING_MODEL,
        use_cache=config.USE_CACHE,
        embeddings_dir=config.EMBEDDINGS_DIR,
        domain=config.DOMAIN
    )

    rag.load_documents(config.GRAPH_FILEPATH, max_tokens=config.MAX_TOKENS)
    rag.embed_documents(config.EMBEDDINGS_DIR)
    rag.build_graph()

    print(f"Graph nodes: {rag.graph.number_of_nodes()}")
    print(f"Graph edges: {rag.graph.number_of_edges()}")
    for d in rag.documents[:3]:
        print(f"Sample node: {d['id']}")
        print(f"Neighbors: {list(rag.graph.neighbors(d['id']))}")

    return rag
