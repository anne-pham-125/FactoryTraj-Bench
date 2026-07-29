"""Strong text-only LLM (non-VLM), zero-shot. needs_training = False.

Only ever receives `prompt` (text) - never sample.raw_input media - by
design, to isolate language reasoning from vision. See configs/models.yaml:
the specific model was left unspecified in the planning doc ("1 LLM manh");
record the exact model/version used in the report, not in this file.

TODO: wire up actual inference (local weights or API).
"""
from __future__ import annotations

from typing import Any

from worldbench.models.base import ModelWrapper
from worldbench.types import Sample


class LLMTextOnlyModel(ModelWrapper):
    model_id = "llm_text_only"
    needs_training = False

    def predict(self, sample: Sample, prompt: str) -> Any:
        raise NotImplementedError(
            f"{self.model_id}.predict(): TODO wire up text-only LLM inference. "
            "Must only consume `prompt`, never sample.raw_input media."
        )
