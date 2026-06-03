"""Evaluation metric helpers for fraud detection."""

from __future__ import annotations

import numpy as np


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Return the fraction of the top-k highest-scored transactions that are actual frauds.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        scores: Continuous fraud probability scores, one per sample.
        k: Number of top-ranked samples to evaluate.

    Returns:
        Precision@k as a float in [0, 1].
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, len(scores))
    top_k_idx = np.argsort(scores)[-k:]
    return float(np.sum(y_true[top_k_idx]) / k)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute the area under the precision-recall curve.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        scores: Continuous fraud probability scores.

    Returns:
        Average precision (AP) as a float in [0, 1]; 0.0 when there are no positives.
    """
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    recall = tp / n_pos
    recall_diff = np.diff(recall, prepend=0.0)
    return float(np.sum(precision * recall_diff))


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the false-positive rate: FP / (FP + TN).

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_pred: Binary predicted labels (0 or 1).

    Returns:
        FPR as a float in [0, 1]; 0.0 when there are no true negatives.
    """
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return 0.0 if (fp + tn) == 0 else fp / (fp + tn)


def detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the detection rate (recall): TP / (TP + FN).

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_pred: Binary predicted labels (0 or 1).

    Returns:
        Detection rate as a float in [0, 1]; 0.0 when there are no true positives.
    """
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return 0.0 if (tp + fn) == 0 else tp / (tp + fn)
