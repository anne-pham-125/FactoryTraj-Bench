"""B0: Schema and tag understanding.
Input: tag names, samples, units, partial documentation.
Output: type, unit, range, role and relationships.
Primary metric: exact_semantic_accuracy (exact match on the structured fields).
"""
from __future__ import annotations

from worldbench.tasks import metrics
from worldbench.types import Prediction, Sample

TASK_ID = "B0"


def build_prompt(sample: Sample) -> str:
    return (
        "Given the following tag name, sample values, units, and any partial "
        "documentation, identify its type, unit, range, role, and relationships "
        "to other tags.\n\n"
        f"Input: {sample.raw_input}"
    )


def parse_output(raw_output: str) -> str:
    return raw_output.strip()


def compute_metric(predictions: list[Prediction], samples: list[Sample]) -> dict:
    gt_by_id = {s.id: s.ground_truth for s in samples}
    y_true = [gt_by_id[p.sample_id] for p in predictions]
    y_pred = [p.output for p in predictions]
    return {"exact_semantic_accuracy": metrics.exact_match_accuracy(y_true, y_pred)}
