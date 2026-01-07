"""Evaluation utilities for fine-tuned models."""

from .metrics import (
    safe_extract_predictions,
    compute_metrics,
)

__all__ = [
    "safe_extract_predictions",
    "compute_metrics",
]
