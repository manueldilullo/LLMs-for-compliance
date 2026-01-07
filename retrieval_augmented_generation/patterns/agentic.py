"""
Agentic RAG patterns implementation.
"""

import json
from typing import Callable, Dict, Any

from ..rag import AbstractRAG
from ..prompts import (
    RAG_PROMPT,
    BASELINE_PROMPT,
    SELF_REFINEMENT_PROMPT,
    REFINE_FROM_ISSUES_PROMPT,
    CRITIC_PROMPT,
)
from ..utils import (
    smart_retrieve,
    get_base_generation,
    askgenerator_raw,
    _join_context,
)


async def baseline_pattern(query: str, llm_func: Callable) -> str:
    """Pattern 0: Baseline LLM without RAG."""
    prompt = BASELINE_PROMPT.format(query=query)
    response = await askgenerator_raw(llm_func, prompt)
    return response.get("answer", "") if isinstance(response, dict) else str(response)


async def rag_pattern(
    model_name: str,
    query: str,
    rag: AbstractRAG,
    llm_func: Callable,
    topk: int = 5,
    topk_max: int = 20,
    threshold: float = 0.0
) -> str:
    """Pattern 1: LLM + RAG."""
    return await get_base_generation(
        model_name=model_name,
        rag=rag,
        query=query,
        llm_func=llm_func,
        topk=topk,
        topk_max=topk_max,
        threshold=threshold,
        use_graph=False
    )


async def rag_with_graph_pattern(
    model_name: str,
    query: str,
    rag: AbstractRAG,
    llm_func: Callable,
    topk: int = 5,
    topk_max: int = 20,
    threshold: float = 0.0
) -> str:
    """Pattern 1.5: LLM + RAG + Graph."""
    return await get_base_generation(
        model_name=model_name,
        rag=rag,
        query=query,
        llm_func=llm_func,
        topk=topk,
        topk_max=topk_max,
        threshold=threshold,
        use_graph=True
    )


async def routing_rag_pattern(
    model_name: str,
    query: str,
    rags_dict: Dict[str, AbstractRAG],
    llm_func: Callable,
    topk: int = 5,
    topk_max: int = 20,
    threshold: float = 0.0,
    use_graph: bool = False
) -> str:
    """Pattern 2: LLM + RAG + Routing (domain classification)."""
    route_prompt = f"Classify this query as 'GDPR', 'AIACT', or 'GENERAL': {query}. Critical: Output only the domain as a single word string!"
    domain_raw = await llm_func(route_prompt, max_tokens=10, stop=[".", "\n"])
    domain = (domain_raw.get("choices", [{}])[0].get("text", "GENERAL")).strip().upper()

    domain = "GDPR" if "GDPR" in domain else "AIACT" if "AIACT" in domain else "GENERAL"

    threshold_map = {"GDPR": 0.4, "AIACT": 0.5, "GENERAL": 0.3}
    threshold = threshold_map.get(domain, 0.3)
    rag = rags_dict.get(domain, None)

    if rag is None:
        return "Empty answer. Couldn't detect domain"

    return await get_base_generation(
        model_name=model_name,
        rag=rag,
        query=query,
        llm_func=llm_func,
        topk=topk,
        topk_max=topk_max,
        threshold=threshold,
        use_graph=use_graph
    )


async def collaboration_rag_pattern(
    model_name: str,
    query: str,
    rag: AbstractRAG,
    llm_func: Callable,
    topk: int = 5,
    topk_max: int = 20,
    threshold: float = 0.0,
    use_graph: bool = False,
    max_rounds: int = 2
) -> str:
    """Pattern 3: LLM + RAG + Collaboration (Generator-Critic loop)."""
    retrieved_docs = await smart_retrieve(
        rag, query, topk_max=topk_max, threshold=threshold, use_graph=use_graph
    )
    context_docs = retrieved_docs[:topk]
    context_text = _join_context(context_docs)

    answer_text = await get_base_generation(
        model_name=model_name,
        rag=rag,
        query=query,
        llm_func=llm_func,
        topk=topk,
        topk_max=topk_max,
        threshold=threshold,
        use_graph=use_graph
    )

    for _ in range(max_rounds):
        critic_prompt = CRITIC_PROMPT.format(
            query=query,
            draft=answer_text,
            context=context_text
        )

        critique = await askgenerator_raw(
            llm_func=llm_func,
            prompt=critic_prompt,
            max_tokens=300,
            temperature=0.0,
            expected_keys=["verdict", "issues"],
            allow_arrays=True
        )

        payload = critique.get("answer")
        if not isinstance(payload, dict):
            payload = {}

        verdict = payload.get("verdict", "REVISE")
        issues = payload.get("issues", payload)

        if verdict == "APPROVE":
            break

        refine_prompt = REFINE_FROM_ISSUES_PROMPT.format(
            query=query,
            draft=answer_text,
            issues_json=json.dumps(issues, ensure_ascii=False),
            context=context_text
        )
        refined = await askgenerator_raw(llm_func, refine_prompt)
        answer_text = refined.get("answer", "") if isinstance(refined, dict) else str(refined)

    return answer_text


async def self_refinement_rag_pattern(
    model_name: str,
    query: str,
    rag: AbstractRAG,
    llm_func: Callable,
    topk: int = 5,
    max_iters: int = 2,
    topk_max: int = 20,
    threshold: float = 0.0,
    use_graph: bool = True
) -> str:
    """Pattern 4: LLM + RAG + Self-Refinement (iterative improvement)."""
    retrieved_docs = await smart_retrieve(
        rag, query, topk_max=topk_max, threshold=0.0, use_graph=True
    )
    context_docs = retrieved_docs[:topk]
    context_text = "\n".join([d["text"] for d in context_docs])

    answer_text = await get_base_generation(
        model_name, rag, query, llm_func,
        topk=topk, topk_max=topk_max, threshold=threshold, use_graph=use_graph
    )

    for _ in range(max_iters):
        refine_prompt = SELF_REFINEMENT_PROMPT.format(
            query=query, answer_text=answer_text, context=context_text
        )
        refined = await askgenerator_raw(llm_func, refine_prompt, temperature=0.3)
        new_answer = refined.get("answer", "") if isinstance(refined, dict) else str(refined)

        if new_answer.strip() == answer_text.strip():
            break
        answer_text = new_answer

    return answer_text
