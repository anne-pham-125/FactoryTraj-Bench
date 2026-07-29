"""Tennessee Eastman Process (TEP) - tabular / numeric process simulation data.

Download: see data/README.md (TEP row) - no single canonical mirror, pick one
(Kaggle / Harvard Dataverse) and record the exact URL used in your report.

Task coverage: B1 (state estimation) per configs/compat_matrix.yaml.
"""
from __future__ import annotations

from worldbench.datasets import data_root, warn_missing_data
from worldbench.types import Sample

DATASET_ID = "tep"


def load(split: str) -> list[Sample]:
    raw_dir = data_root() / DATASET_ID
    if not raw_dir.exists():
        warn_missing_data(
            DATASET_ID,
            raw_dir,
            "download a TEP mirror per data/README.md and place it under this path",
        )
        return []

    # TODO: parse TEP simulation runs (fault/mode labels) into Sample objects
    # for split (e.g. "train" / "test_locked"). ground_truth = machine mode /
    # fault label for task B1.
    raise NotImplementedError(
        f"{DATASET_ID}: raw data found at {raw_dir} but the CSV/mat parser is not written yet."
    )
