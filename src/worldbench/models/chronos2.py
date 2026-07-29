"""Chronos-2 specialized time-series baseline. needs_training = True.

TODO: wire up the actual Chronos-2 forecasting model (pip package / weights)
once available - this is a structural placeholder, see configs/models.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from worldbench.models.base import ModelWrapper
from worldbench.types import Sample

CHECKPOINT_DIR = Path("checkpoints")


class Chronos2Model(ModelWrapper):
    model_id = "chronos2"
    needs_training = True

    def __init__(self) -> None:
        self._model = None

    def fit(self, train_samples: list[Sample]) -> None:
        raise NotImplementedError(
            f"{self.model_id}.fit(): TODO integrate the Chronos-2 forecasting model."
        )

    def predict(self, sample: Sample, prompt: str) -> Any:
        if self._model is None:
            raise RuntimeError(f"{self.model_id}: fit() (or checkpoint load) must run before predict()")
        raise NotImplementedError(f"{self.model_id}.predict(): not wired up yet.")
