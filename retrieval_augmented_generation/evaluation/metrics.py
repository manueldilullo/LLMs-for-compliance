"""
Evaluation metrics for RAG generation.
"""

import logging
from typing import List, Dict
from collections import defaultdict

import evaluate

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Basic text normalization."""
    return text.lower().strip()


def compute_em(pred: str, ref: str) -> float:
    """Compute exact match score."""
    return 1.0 if normalize_text(pred) == normalize_text(ref) else 0.0


def compute_f1(pred: str, ref: str) -> float:
    """Compute F1 score based on word overlap."""
    pt = normalize_text(pred).split()
    rt = normalize_text(ref).split()
    if not pt or not rt:
        return 0.0
    common = set(pt) & set(rt)
    if not common:
        return 0.0
    precision = len(common) / len(pt)
    recall = len(common) / len(rt)
    return 2 * precision * recall / (precision + recall)


def compute_all_metrics(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    results = defaultdict(float)
    n = len(predictions)

    if n == 0:
        return dict(results)

    # EM and F1
    try:
        em_scores = [compute_em(p, r) for p, r in zip(predictions, references)]
        f1_scores = [compute_f1(p, r) for p, r in zip(predictions, references)]
        results["em"] = sum(em_scores) / n
        results["f1"] = sum(f1_scores) / n
    except Exception as e:
        logger.warning(f"EM/F1 computation failed: {e}")
        results["em"] = results["f1"] = 0.0

    # BLEU
    try:
        bleu_metric = evaluate.load("bleu")
        bleu_result = bleu_metric.compute(
            predictions=predictions,
            references=[[r] for r in references]
        )
        results["bleu"] = bleu_result["bleu"]
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        results["bleu"] = 0.0

    # ROUGE
    try:
        rouge_metric = evaluate.load("rouge")
        rouge_result = rouge_metric.compute(
            predictions=predictions,
            references=references
        )
        results["rouge1"] = rouge_result["rouge1"]
        results["rouge2"] = rouge_result["rouge2"]
        results["rougeL"] = rouge_result["rougeL"]
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
        results["rouge1"] = results["rouge2"] = results["rougeL"] = 0.0

    # METEOR
    try:
        meteor_metric = evaluate.load("meteor")
        meteor_result = meteor_metric.compute(
            predictions=predictions,
            references=references
        )
        results["meteor"] = meteor_result["meteor"]
    except Exception as e:
        logger.warning(f"METEOR computation failed: {e}")
        results["meteor"] = 0.0

    # BERTScore
    try:
        bertscore_metric = evaluate.load("bertscore")
        bertscore_result = bertscore_metric.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type="distilbert-base-uncased",
            batch_size=8
        )
        results["bertscore_f1"] = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])
        results["bertscore_p"] = sum(bertscore_result["precision"]) / len(bertscore_result["precision"])
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
        results["bertscore_f1"] = 0.0
        results["bertscore_p"] = 0.0

    return dict(results)
