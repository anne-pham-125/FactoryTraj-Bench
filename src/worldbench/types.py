"""Shared data schema used by every dataset loader, model wrapper, and scorer.

Nothing downstream should invent its own sample/prediction format - if a
dataset needs an extra field, put it in `metadata`, don't add a new type.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    id: str
    task_id: str
    dataset: str
    raw_input: Any
    ground_truth: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    sample_id: str
    task_id: str
    dataset: str
    model: str
    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    id: str
    name: str
    input_desc: str
    output_desc: str
    primary_metric: str
    secondary_metrics: list[str] = field(default_factory=list)
