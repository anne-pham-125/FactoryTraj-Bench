"""Cosmos3-Nano Reasoner - MLLM zero-shot group. needs_training = False.

TODO: wire up actual Cosmos3-Nano inference (local weights or API) - this
is a structural placeholder so the pipeline can be exercised end-to-end
before the real model is available. See configs/models.yaml.
"""
from __future__ import annotations

from typing import Any

from worldbench.models.base import ModelWrapper
from worldbench.types import Sample


class Cosmos3NanoModel(ModelWrapper):
    model_id = "cosmos3_nano"
    needs_training = False

    def predict(self, sample: Sample, prompt: str) -> Any:
        raise NotImplementedError(
            f"{self.model_id}.predict(): TODO wire up Cosmos3-Nano Reasoner inference."
        )
