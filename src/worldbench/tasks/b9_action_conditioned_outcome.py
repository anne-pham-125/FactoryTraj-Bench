"""B9: Action-conditioned outcome prediction.
Input: state plus candidate action.
Output: state/event distribution after action.
Primary metric: calibration. Secondary: outcome_accuracy.

This is the task that would need to hold up before calling this a "world
model" benchmark (see README.md section 1) - it requires a real or
simulated environment to generate ground-truth post-action outcomes, which
no current public data source provides (see configs/compat_matrix.yaml -
B9 has no dataset entries yet).
"""
from __future__ import annotations

import logging

from worldbench.tasks import metrics
from worldbench.types import Prediction, Sample

TASK_ID = "B9"
logger = logging.getLogger("worldbench.tasks.b9")


def build_prompt(sample: Sample) -> str:
    return (
        "Given the current state and a candidate action, predict the "
        "resulting state/event distribution.\n\n"
        f"Input: {sample.raw_input}"
    )


def parse_output(raw_output: str) -> str:
    return raw_output.strip()


def compute_metric(predictions: list[Prediction], samples: list[Sample]) -> dict:
    gt_by_id = {s.id: s.ground_truth for s in samples}
    try:
        y_true = [int(gt_by_id[p.sample_id]) for p in predictions]
        y_prob = [float(p.output) for p in predictions]
        return {"calibration": metrics.brier_score(y_true, y_prob)}
    except (TypeError, ValueError):
        logger.warning("B9 has no environment to source ground-truth outcomes from yet.")
        return {"calibration": float("nan")}
