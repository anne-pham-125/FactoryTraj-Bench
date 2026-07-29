"""Persistence / majority-class baseline. No training in the ML sense, but
fit() still needs to run once to compute the majority label from the train
split - see configs/models.yaml (baseline_specialized group)."""
from __future__ import annotations

from collections import Counter
from typing import Any

from worldbench.models.base import ModelWrapper
from worldbench.types import Sample


class MajorityClassModel(ModelWrapper):
    model_id = "majority_class"
    needs_training = False

    def __init__(self) -> None:
        self._majority_label: Any = None

    def fit(self, train_samples: list[Sample]) -> None:
        labels = [s.ground_truth for s in train_samples]
        self._majority_label = Counter(labels).most_common(1)[0][0] if labels else None

    def predict(self, sample: Sample, prompt: str) -> Any:
        if self._majority_label is None:
            raise RuntimeError(f"{self.model_id}: fit() must be called before predict()")
        return self._majority_label
