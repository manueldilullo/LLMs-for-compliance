"""
Evaluation metrics for fine-tuned models.
"""

from typing import List, Dict, Any, Tuple

import torch
from datasets import Dataset
import evaluate


def safe_extract_predictions(
    raw_dataset: Dataset,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 64,
    num_samples: int = 500,
) -> Tuple[List[str], List[str]]:
    """
    Extract predictions using greedy decoding (stable, no sampling crashes).

    Args:
        raw_dataset: Dataset with raw message format
        model: The model for generation
        tokenizer: The tokenizer
        max_new_tokens: Maximum new tokens to generate
        num_samples: Number of samples to process

    Returns:
        Tuple of (predictions, references)
    """
    predictions, references = [], []
    model.eval()

    for i, item in enumerate(raw_dataset.select(range(num_samples))):
        messages = item["messages"]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        print(f"Generating {i+1}/{num_samples}: {messages[0]['content'][:50]}...")

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            pred = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

        ref = messages[1]["content"]
        predictions.append(pred.strip())
        references.append(ref.strip())

    return predictions, references


def compute_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, Any]:
    """
    Compute evaluation metrics (BLEU, ROUGE, METEOR, BERTScore).

    Args:
        predictions: List of predicted strings
        references: List of reference strings

    Returns:
        Dictionary with metric results
    """
    # Load metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    bertscore_metric = evaluate.load("bertscore")

    results = {
        "bleu": bleu_metric.compute(predictions=predictions, references=references),
        "rouge": rouge_metric.compute(predictions=predictions, references=references),
        "bertscore": bertscore_metric.compute(predictions=predictions, references=references, lang="en"),
        "meteor": meteor_metric.compute(predictions=predictions, references=references),
    }

    return results
