"""B10: OOD recognition.
Input: unseen machine, product or fault.
Output: prediction or abstention.
Primary metric: risk_coverage_curve. Secondary: unsafe_confidence.
"""
from __future__ import annotations

from worldbench.types import Prediction, Sample

TASK_ID = "B10"

ABSTAIN_TOKEN = "ABSTAIN"


def build_prompt(sample: Sample) -> str:
    return (
        "Given the following input, which may be an unseen machine, product, "
        f"or fault type, predict the label or abstain (output '{ABSTAIN_TOKEN}').\n\n"
        f"Input: {sample.raw_input}"
    )


def parse_output(raw_output: str) -> str:
    return raw_output.strip()


def compute_metric(predictions: list[Prediction], samples: list[Sample]) -> dict:
    gt_by_id = {s.id: s.ground_truth for s in samples}
    coverage = sum(1 for p in predictions if p.output != ABSTAIN_TOKEN) / len(predictions) if predictions else 0.0
    answered = [p for p in predictions if p.output != ABSTAIN_TOKEN]
    risk = (
        sum(1 for p in answered if p.output != gt_by_id[p.sample_id]) / len(answered)
        if answered
        else 0.0
    )
    unsafe_confidence = sum(
        1 for p in answered if p.output != gt_by_id[p.sample_id]
    )
    return {
        "coverage": coverage,
        "risk_at_coverage": risk,
        "unsafe_confidence": unsafe_confidence,
    }
