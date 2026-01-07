"""
Retrieval evaluation utilities.
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional

from ..rag import AbstractRAG
from ..utils import sort_index_related_lists


def parse_refid_to_ground_truth(ref_id: str, question_type: str) -> Dict[str, Any]:
    """Parse reference ID to ground truth document types."""
    gt = {"article": False, "annex": False, "recital": False}

    nums = [t for t in (ref_id or "").split("_") if t.isdigit()]

    if "unity" in (question_type or ""):
        if "article" in question_type and nums:
            gt["article"] = nums[0]
        if "annex" in question_type and nums:
            gt["annex"] = nums[0]
        if "recital" in question_type and nums:
            gt["recital"] = nums[0]

    elif "binding" in (question_type or ""):
        if len(nums) >= 2:
            a, b = nums[0], nums[1]
            if "article_recital" in question_type:
                gt["article"], gt["recital"] = a, b
            elif "annex_article" in question_type:
                gt["annex"], gt["article"] = a, b
            elif "annex_recital" in question_type:
                gt["annex"], gt["recital"] = a, b

    return gt


def extract_id_from_doc(doc_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract document type and number from chunk ID."""
    if doc_id.startswith('Article_'):
        parts = doc_id.split('_')
        return ('article', parts[1])
    elif doc_id.startswith('Annex_'):
        parts = doc_id.split('_')
        return ('annex', parts[1])
    elif doc_id.startswith('Recital_'):
        parts = doc_id.split('_')
        return ('recital', parts[1])
    return (None, None)


def create_retrieval_cache(
    rag_system: AbstractRAG,
    qa_pairs: List[Dict],
    top_k: int = 20,
    use_graph: bool = False,
    output_file: str = "cache.jsonl",
    threshold: float = 0.6
) -> List[Dict]:
    """Generate retrieval cache for evaluation."""
    cache = []
    for pair in qa_pairs:
        question = pair['question']
        refid = pair.get('refid', '')
        question_type = pair.get('type', '')

        if not refid:
            recital_number = pair.get("recital_number", False)
            article_number = pair.get("article_number", False)
            annex_number = pair.get("annex_number", False)
            if recital_number or article_number or annex_number:
                gt = {"article": article_number, "annex": annex_number, "recital": recital_number}
            else:
                continue
        else:
            gt = parse_refid_to_ground_truth(refid, question_type)

        if use_graph:
            context = rag_system.retrieve_with_graph(question, top_k, threshold=threshold)
        else:
            context = rag_system.retrieve(question, top_k, threshold=threshold)

        predictions, similarities = [], []
        for doc in context:
            doc_type, doc_num = extract_id_from_doc(doc['id'])
            graph_related = doc.get('graph_related', False)
            if doc_type:
                predictions.append({doc_type: doc_num, "graph_related": graph_related})
                similarities.append(doc['score'])

        if not similarities or not predictions:
            print(f"Empty predictions or similarity")
            continue
        similarities, predictions = sort_index_related_lists(similarities, predictions)

        cache.append({
            "question": question,
            "ground_truth": gt,
            "predictions": predictions,
            "similarity": similarities,
            "threshold": threshold
        })

    with open(output_file, 'w') as f:
        for entry in cache:
            f.write(json.dumps(entry) + '\n')
    return cache


def calculate_retrieval(
    rag_system: AbstractRAG,
    qa_pairs: List[Dict],
    top_k: int = 20,
    use_graph: bool = False,
    cache_file: str = None,
    load_cache: bool = False,
    threshold: float = float("-inf")
) -> List[Dict]:
    """Calculate retrieval results with optional caching."""
    if load_cache and cache_file and os.path.exists(cache_file):
        print(f"Loading existing cache: {cache_file}")
        cache = []
        with open(cache_file) as f:
            for line in f:
                cache.append(json.loads(line))
    else:
        cache = create_retrieval_cache(
            rag_system, qa_pairs, top_k, use_graph, cache_file, threshold
        )
    return cache
