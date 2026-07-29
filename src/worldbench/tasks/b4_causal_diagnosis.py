"""B4: Causal diagnosis.
Input: trajectory before and during interruption.
Output: ranked causal mechanisms with evidence.
Primary metric: top1_accuracy. Secondary: top3_accuracy.

No dataset currently supplies ground-truth causal mechanisms (see
configs/compat_matrix.yaml - B4 has no dataset entries yet).
"""
from __future__ import annotations

from worldbench.tasks import metrics
from worldbench.types import Prediction, Sample

TASK_ID = "B4"


def build_prompt(sample: Sample) -> str:
    return (
        "Given the following trajectory before and during the interruption, "
        "rank the likely causal mechanisms with supporting evidence.\n\n"
        f"Input: {sample.raw_input}"
    )


def parse_output(raw_output: str) -> list[str]:
    return [line.strip() for line in raw_output.splitlines() if line.strip()]


def compute_metric(predictions: list[Prediction], samples: list[Sample]) -> dict:
    gt_by_id = {s.id: s.ground_truth for s in samples}
    y_true = [gt_by_id[p.sample_id] for p in predictions]
    ranked_preds = [p.output if isinstance(p.output, list) else [p.output] for p in predictions]
    return {
        "top1_accuracy": metrics.top_k_accuracy(y_true, ranked_preds, k=1),
        "top3_accuracy": metrics.top_k_accuracy(y_true, ranked_preds, k=3),
    }
