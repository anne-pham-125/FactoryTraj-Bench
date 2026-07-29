"""B2: Cycle and event segmentation.
Input: continuous logs/video.
Output: cycle boundaries, stop onset and recovery.
Primary metric: event_f1. Secondary: timing_error.

No dataset currently supplies ground-truth cycle boundaries (see
configs/compat_matrix.yaml - B2 has no dataset entries yet), so
compute_metric is a placeholder until one does.
"""
from __future__ import annotations

import logging

from worldbench.types import Prediction, Sample

TASK_ID = "B2"
logger = logging.getLogger("worldbench.tasks.b2")


def build_prompt(sample: Sample) -> str:
    return (
        "Given the following continuous log/video segment, identify cycle "
        "boundaries, stop onset, and recovery points.\n\n"
        f"Input: {sample.raw_input}"
    )


def parse_output(raw_output: str) -> str:
    return raw_output.strip()


def compute_metric(predictions: list[Prediction], samples: list[Sample]) -> dict:
    logger.warning(
        "B2 has no dataset with ground-truth event boundaries yet - "
        "event_f1/timing_error are not computable. Returning NaN placeholders."
    )
    return {"event_f1": float("nan"), "timing_error": float("nan")}
