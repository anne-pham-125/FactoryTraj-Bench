"""Generic metric implementations shared across task modules.

These are intentionally generic (treat outputs as labels/rankings/probs).
Once real model output formats are known for a given task, its build_prompt/
parse_output should do the task-specific shaping - compute_metric should
still be able to call into these.
"""
from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    ndcg_score,
)


def macro_f1(y_true: list[Any], y_pred: list[Any]) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def exact_match_accuracy(y_true: list[Any], y_pred: list[Any]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def top_k_accuracy(y_true: list[Any], ranked_preds: list[list[Any]], k: int) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, ranked in zip(y_true, ranked_preds) if t in ranked[:k])
    return correct / len(y_true)


def auprc(y_true: list[int], y_score: list[float]) -> float:
    return float(average_precision_score(y_true, y_score))


def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    if not y_true:
        return 0.0
    return sum((p - t) ** 2 for t, p in zip(y_true, y_prob)) / len(y_true)


def recall_at_k(y_true: list[Any], ranked_preds: list[list[Any]], k: int) -> float:
    return top_k_accuracy(y_true, ranked_preds, k)


def ndcg_at_k(y_true_relevance: list[list[float]], y_score: list[list[float]], k: int) -> float:
    return float(ndcg_score(y_true_relevance, y_score, k=k))
