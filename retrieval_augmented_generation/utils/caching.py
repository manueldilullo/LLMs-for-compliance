"""
Caching utilities for RAG system.
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Callable

from ..rag import AbstractRAG
from ..prompts import RAG_PROMPT

# Global caches
base_generation_cache: Dict[str, str] = {}
base_retrieval_cache: Dict[str, List[Dict]] = {}


def compute_query_hash(
    model_name: str,
    query: str,
    topk: int = 5,
    threshold: float = 0.0,
    use_graph: bool = False
) -> str:
    """Compute hash for query caching."""
    query_str = f"{model_name}|{query}|{topk}|{threshold}|{use_graph}"
    return hashlib.md5(query_str.encode()).hexdigest()[:24]


def compute_rag_hash(
    query: str,
    topk_max: int = 10,
    threshold: float = 0.0,
    use_graph: bool = False
) -> str:
    """Compute hash for retrieval caching."""
    retrieval_str = f"{query}|{topk_max}|{threshold}|{str(use_graph)}"
    return hashlib.md5(retrieval_str.encode()).hexdigest()[:12]


async def smart_retrieve(
    rag: AbstractRAG,
    query: str,
    topk_max: int = 10,
    threshold: float = 0.0,
    use_graph: bool = False
) -> List[Dict[str, Any]]:
    """Smart retrieve with caching."""
    query_hash = compute_rag_hash(query, topk_max, threshold, use_graph)

    if query_hash in base_retrieval_cache:
        return base_retrieval_cache[query_hash]

    if use_graph:
        result = rag.retrieve_with_graph(query, topk=topk_max, threshold=threshold)
    else:
        result = rag.retrieve(query, topk=topk_max, threshold=threshold)

    base_retrieval_cache[query_hash] = result
    return result


def _join_context(docs: List[Dict[str, Any]]) -> str:
    """Join document texts into context string."""
    return "\n".join(d["text"] for d in docs if d.get("text"))


async def get_base_generation(
    model_name: str,
    rag: AbstractRAG,
    query: str,
    llm_func: Callable,
    topk: int = 5,
    topk_max: int = 10,
    threshold: float = 0.0,
    use_graph: bool = False,
    temperature: float = 0.3
) -> str:
    """Get base generation with caching."""
    from .llm import askgenerator_raw
    
    query_hash = compute_query_hash(
        model_name=model_name,
        query=query,
        topk=topk,
        threshold=threshold,
        use_graph=use_graph
    )

    if query_hash in base_generation_cache:
        return base_generation_cache[query_hash]

    retrieved_docs = await smart_retrieve(
        rag, query, topk_max=topk_max, threshold=threshold, use_graph=use_graph
    )
    context_docs = retrieved_docs[:topk]
    context_text = _join_context(context_docs)

    prompt = RAG_PROMPT.format(query=query, context=context_text)
    response = await askgenerator_raw(llm_func, prompt, temperature)
    answer = response.get("answer", "") if isinstance(response, dict) else str(response)

    base_generation_cache[query_hash] = answer
    return answer


def load_checkpoint(output_dir: str) -> Dict[str, bool]:
    """Load checkpoint for resuming evaluation."""
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {}


def save_checkpoint(output_dir: str, checkpoint: Dict[str, bool]) -> None:
    """Save checkpoint for resuming evaluation."""
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def compute_config_hash(
    model_name: str,
    pattern_name: str,
    topk: int,
    limit: int
) -> str:
    """Compute configuration hash for checkpointing."""
    config_str = f"{model_name}|{pattern_name}|{topk}|{limit}"
    return hashlib.md5(config_str.encode()).hexdigest()[:8]
