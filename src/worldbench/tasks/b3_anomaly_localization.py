"""B3: Anomaly and fault localization.
Input: signals, alarms and context.
Output: anomaly interval and responsible subsystem.
Primary metric: event_auprc. Secondary: topk_localization.
"""
from __future__ import annotations

from worldbench.tasks import metrics
from worldbench.types import Prediction, Sample

TASK_ID = "B3"


def build_prompt(sample: Sample) -> str:
    return (
        "Given the following signals/alarms and context, identify the anomaly "
        "interval and the responsible subsystem.\n\n"
        f"Input: {sample.raw_input}"
    )


def parse_output(raw_output: str) -> str:
    return raw_output.strip()


def compute_metric(predictions: list[Prediction], samples: list[Sample]) -> dict:
    gt_by_id = {s.id: s.ground_truth for s in samples}
    # Expects prediction.output to be an anomaly score in [0, 1] and
    # ground_truth to be 0/1 for AUPRC. Falls back to NaN if that shape
    # isn't met yet (model wrapper hasn't been wired to emit scores).
    try:
        y_true = [int(gt_by_id[p.sample_id]) for p in predictions]
        y_score = [float(p.output) for p in predictions]
        return {"event_auprc": metrics.auprc(y_true, y_score)}
    except (TypeError, ValueError):
        return {"event_auprc": float("nan")}
