"""task_id -> TaskSpec (from configs/tasks.yaml) and task_id -> handler module.

score_run.py and run_eval.py should go through this module rather than
importing b0_schema_tag.py etc directly, so adding a task only ever means
touching configs/tasks.yaml and one new module here.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from worldbench.tasks import (
    b0_schema_tag,
    b1_state_estimation,
    b2_segmentation,
    b3_anomaly_localization,
    b4_causal_diagnosis,
    b5_next_state_prediction,
    b6_sop_grounding,
    b7_workflow_execution,
    b8_recovery_ranking,
    b9_action_conditioned_outcome,
    b10_ood_recognition,
)
from worldbench.types import TaskSpec

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TASKS_CONFIG = _REPO_ROOT / "configs" / "tasks.yaml"

TASK_HANDLERS = {
    "B0": b0_schema_tag,
    "B1": b1_state_estimation,
    "B2": b2_segmentation,
    "B3": b3_anomaly_localization,
    "B4": b4_causal_diagnosis,
    "B5": b5_next_state_prediction,
    "B6": b6_sop_grounding,
    "B7": b7_workflow_execution,
    "B8": b8_recovery_ranking,
    "B9": b9_action_conditioned_outcome,
    "B10": b10_ood_recognition,
}


def load_task_specs(config_path: Path = _TASKS_CONFIG) -> dict[str, TaskSpec]:
    with open(config_path) as f:
        data = yaml.safe_load(f)
    specs = {}
    for entry in data["tasks"]:
        specs[entry["id"]] = TaskSpec(
            id=entry["id"],
            name=entry["name"],
            input_desc=entry["input"],
            output_desc=entry["output"],
            primary_metric=entry["primary_metric"],
            secondary_metrics=entry.get("secondary_metrics", []),
        )
    return specs


def get_handler(task_id: str):
    if task_id not in TASK_HANDLERS:
        raise KeyError(f"No task handler registered for '{task_id}'. Known tasks: {sorted(TASK_HANDLERS)}")
    return TASK_HANDLERS[task_id]
