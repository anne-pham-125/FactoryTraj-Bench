"""GPT-4o via API - commercial group, zero-shot. needs_training = False.

Reads OPENAI_API_KEY from the environment (loaded from .env via
python-dotenv) - never hardcode the key.
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from worldbench.models.base import ModelWrapper
from worldbench.types import Sample

load_dotenv()


class GPT4oModel(ModelWrapper):
    model_id = "gpt4o"
    needs_training = False

    def __init__(self, model_name: str = "gpt-4o") -> None:
        self.model_name = model_name

    def predict(self, sample: Sample, prompt: str) -> Any:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"{self.model_id}: OPENAI_API_KEY not set - copy .env.example to .env and fill it in."
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                f"{self.model_id}: the `openai` package is required - add it to requirements.txt "
                "or evaluation/examples/commercial_api/requirements.txt."
            ) from e

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
