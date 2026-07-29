"""B6: SOP grounding.
Input: current state, alarms and SOP.
Output: applicable step, blocked steps and required checks.
Primary metric: step_accuracy. Secondary: contradiction_rate.

NOTE: the source planning doc's metric text for this task was garbled
("Step acciction rate") - step_accuracy/contradiction_rate below is a
best-effort reconstruction. Confirm the intended secondary metric before
treating it as final (see configs/tasks.yaml TODO on this task).

No dataset currently supplies ground-truth SOP steps (see
configs/compat_matrix.yaml - B6 has no dataset entries yet).
"""
from __future__ import annotations

from worldbench.tasks import metrics
from worldbench.types import Prediction, Sample

TASK_ID = "B6"


def build_prompt(sample: Sample) -> str:
    return (
        "Given the current state, active alarms, and the SOP, identify the "
        "applicable step, any blocked steps, and required checks.\n\n"
        f"Input: {sample.raw_input}"
    )


def parse_output(raw_output: str) -> str:
    return raw_output.strip()


def compute_metric(predictions: list[Prediction], samples: list[Sample]) -> dict:
    gt_by_id = {s.id: s.ground_truth for s in samples}
    y_true = [gt_by_id[p.sample_id] for p in predictions]
    y_pred = [p.output for p in predictions]
    return {"step_accuracy": metrics.exact_match_accuracy(y_true, y_pred)}
