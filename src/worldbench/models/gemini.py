"""Gemini via API - commercial group, zero-shot. needs_training = False.

Reads GOOGLE_API_KEY from the environment (loaded from .env via
python-dotenv) - never hardcode the key.
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from worldbench.models.base import ModelWrapper
from worldbench.types import Sample

load_dotenv()


class GeminiModel(ModelWrapper):
    model_id = "gemini"
    needs_training = False

    def __init__(self, model_name: str = "gemini-1.5-pro") -> None:
        self.model_name = model_name

    def predict(self, sample: Sample, prompt: str) -> Any:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"{self.model_id}: GOOGLE_API_KEY not set - copy .env.example to .env and fill it in."
            )
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise RuntimeError(
                f"{self.model_id}: the `google-generativeai` package is required - add it to "
                "requirements.txt or evaluation/examples/commercial_api/requirements.txt."
            ) from e

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text
