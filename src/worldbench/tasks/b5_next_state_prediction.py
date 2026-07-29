"""B5: Next-state/event prediction.
Input: observation window and production context.
Output: events/state in next 30-120 seconds.
Primary metric: auprc. Secondary: brier_score, sequence_edit_distance.
"""
from __future__ import annotations

from worldbench.tasks import metrics
from worldbench.types import Prediction, Sample

TASK_ID = "B5"


def build_prompt(sample: Sample) -> str:
    return (
        "Given the following observation window and production context, "
        "predict the events/state in the next 30-120 seconds.\n\n"
        f"Input: {sample.raw_input}"
    )


def parse_output(raw_output: str) -> str:
    return raw_output.strip()


def _sequence_edit_distance(a: list, b: list) -> int:
    # Standard Levenshtein distance over event sequences.
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1]


def compute_metric(predictions: list[Prediction], samples: list[Sample]) -> dict:
    gt_by_id = {s.id: s.ground_truth for s in samples}
    try:
        y_true = [int(gt_by_id[p.sample_id]) for p in predictions]
        y_score = [float(p.output) for p in predictions]
        result = {"auprc": metrics.auprc(y_true, y_score)}
    except (TypeError, ValueError):
        result = {"auprc": float("nan")}
        y_true, y_score = [], []

    edit_distances = [
        _sequence_edit_distance(list(str(gt_by_id[p.sample_id])), list(str(p.output)))
        for p in predictions
    ]
    result["sequence_edit_distance"] = (
        sum(edit_distances) / len(edit_distances) if edit_distances else float("nan")
    )
    return result
