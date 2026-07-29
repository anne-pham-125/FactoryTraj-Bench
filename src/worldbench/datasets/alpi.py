"""ALPI/PIADE packaging alarm logs - event/alarm tabular data.

Download: https://zenodo.org/records/7071747 (see data/README.md).

Task coverage: B0 (schema/tag understanding), B5 (next-state/event
prediction) per configs/compat_matrix.yaml.
"""
from __future__ import annotations

from worldbench.datasets import data_root, warn_missing_data
from worldbench.types import Sample

DATASET_ID = "alpi"


def load(split: str) -> list[Sample]:
    raw_dir = data_root() / DATASET_ID
    if not raw_dir.exists():
        warn_missing_data(
            DATASET_ID,
            raw_dir,
            "download from https://zenodo.org/records/7071747 and place it under this path",
        )
        return []

    # TODO: parse ALPI/PIADE alarm event logs into Sample objects for split.
    # ground_truth depends on task_id: tag schema info for B0, next-event/
    # state for B5.
    raise NotImplementedError(
        f"{DATASET_ID}: raw data found at {raw_dir} but the log parser is not written yet."
    )
