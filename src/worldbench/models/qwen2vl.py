"""Qwen2-VL-2B - MLLM zero-shot group. needs_training = False.

TODO: wire up actual Qwen2-VL-2B inference (transformers / local weights)
- this is a structural placeholder. See configs/models.yaml.
"""
from __future__ import annotations

from typing import Any

from worldbench.models.base import ModelWrapper
from worldbench.types import Sample


class Qwen2VLModel(ModelWrapper):
    model_id = "qwen2vl"
    needs_training = False

    def predict(self, sample: Sample, prompt: str) -> Any:
        raise NotImplementedError(
            f"{self.model_id}.predict(): TODO wire up Qwen2-VL-2B inference."
        )
