"""Evaluation metric helpers for fraud detection."""

from __future__ import annotations

import numpy as np


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of the top-k scored transactions that are actual frauds."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, len(scores))
    top_k_idx = np.argsort(scores)[-k:]
    return float(np.sum(y_true[top_k_idx]) / k)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve (sklearn-compatible implementation)."""
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
    """FPR = FP / (FP + TN)."""
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return 0.0 if (fp + tn) == 0 else fp / (fp + tn)


def detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall / true-positive rate = TP / (TP + FN)."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return 0.0 if (tp + fn) == 0 else tp / (tp + fn)
