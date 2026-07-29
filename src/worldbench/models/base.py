"""Shared model wrapper interface.

Every model in evaluation/examples/ wraps one of these. Zero-shot models
(MLLM, LLM text-only, commercial API) only need predict(); models that need
training (XGBoost/LightGBM, Chronos-2, V-JEPA 2 linear probe) also implement
fit() and load their own checkpoint from checkpoints/ (gitignored).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from worldbench.types import Sample


class ModelWrapper(ABC):
    model_id: str
    needs_training: bool = False

    @abstractmethod
    def predict(self, sample: Sample, prompt: str) -> Any:
        """Return a raw output for one sample.

        `prompt` is the task's build_prompt(sample) output (text instruction);
        `sample.raw_input` carries the raw multimodal payload (video/image
        path, sensor window, etc) for models that need more than text.
        Task-specific parsing into a Prediction happens in worldbench.runner
        via the task's parse_output()."""
        raise NotImplementedError

    def fit(self, train_samples: list[Sample]) -> None:
        """Called by worldbench.runner before every eval run, even for
        zero-shot models - default is a no-op unless needs_training is True.
        Override this for models that need a fit-like step (gradient
        training, or something cheap like a majority-class baseline
        computing the mode) regardless of the needs_training flag."""
        if self.needs_training:
            raise NotImplementedError(f"{self.model_id} needs_training=True but fit() is not implemented")
