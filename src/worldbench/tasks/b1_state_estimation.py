"""B1: State estimation.
Input: 60-second multimodal window.
Output: machine mode and latent operational state.
Primary metric: macro_f1. Secondary: brier_score.
"""
from __future__ import annotations

from worldbench.tasks import metrics
from worldbench.types import Prediction, Sample

TASK_ID = "B1"


def build_prompt(sample: Sample) -> str:
    return (
        "Given the following 60-second multimodal observation window, "
        "identify the machine mode and latent operational state.\n\n"
        f"Input: {sample.raw_input}"
    )


def parse_output(raw_output: str) -> str:
    return raw_output.strip()


def compute_metric(predictions: list[Prediction], samples: list[Sample]) -> dict:
    gt_by_id = {s.id: s.ground_truth for s in samples}
    y_true = [gt_by_id[p.sample_id] for p in predictions]
    y_pred = [p.output for p in predictions]
    return {"macro_f1": metrics.macro_f1(y_true, y_pred)}
