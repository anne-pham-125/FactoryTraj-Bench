"""B7: Workflow execution.
Input: goal, state and available tools.
Output: typed sequence of actions with preconditions.
Primary metric: simulator_success. Secondary: constraint_violations.

Requires a simulator/environment to check preconditions and execute the
action sequence - not available with current public data sources (see
configs/compat_matrix.yaml - B7 has no dataset entries yet).
"""
from __future__ import annotations

import logging

from worldbench.types import Prediction, Sample

TASK_ID = "B7"
logger = logging.getLogger("worldbench.tasks.b7")


def build_prompt(sample: Sample) -> str:
    return (
        "Given the goal, current state, and available tools, produce a typed "
        "sequence of actions with their preconditions.\n\n"
        f"Input: {sample.raw_input}"
    )


def parse_output(raw_output: str) -> list[str]:
    return [line.strip() for line in raw_output.splitlines() if line.strip()]


def compute_metric(predictions: list[Prediction], samples: list[Sample]) -> dict:
    logger.warning(
        "B7 requires a simulator/environment to score action sequences - "
        "none is wired up yet. Returning NaN placeholders."
    )
    return {"simulator_success": float("nan"), "constraint_violations": float("nan")}
