"""Rule-based baseline. Rules are inherently task/dataset specific - this is
a placeholder that must be filled in per (dataset, task) before it means
anything. It intentionally does not fall back to majority_class silently,
so a report can't mistake an unfilled rule set for a real baseline."""
from __future__ import annotations

import logging
from typing import Any

from worldbench.models.base import ModelWrapper
from worldbench.types import Sample

logger = logging.getLogger("worldbench.models.rule_based")


class RuleBasedModel(ModelWrapper):
    model_id = "rule_based"
    needs_training = False

    def predict(self, sample: Sample, prompt: str) -> Any:
        raise NotImplementedError(
            f"{self.model_id}: no rule set has been written yet for "
            f"dataset={sample.dataset} task={sample.task_id}. "
            "Add task/dataset-specific logic here before running this baseline."
        )
