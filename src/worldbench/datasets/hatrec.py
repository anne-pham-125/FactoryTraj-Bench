"""HATREC - assembly video.

Download: internal - see team's shared data location (not a public URL);
record the exact source in data/README.md once known.

Task coverage: B1 (state estimation) per configs/compat_matrix.yaml.

KNOWN ISSUES from prior runs (see configs/compat_matrix.yaml notes): a
static-frame shortcut has been observed, and a train/test leak inflated at
least one prior V-JEPA 2 result to a suspicious 1.0 F1. Always run
scoring/leak_check.py and scoring/shortcut_check.py against this dataset
before trusting a headline number.
"""
from __future__ import annotations

from worldbench.datasets import data_root, warn_missing_data
from worldbench.types import Sample

DATASET_ID = "hatrec"


def load(split: str) -> list[Sample]:
    raw_dir = data_root() / DATASET_ID
    if not raw_dir.exists():
        warn_missing_data(
            DATASET_ID,
            raw_dir,
            "obtain HATREC from the team's shared data location and place it under this path",
        )
        return []

    # TODO: parse HATREC video clips into Sample objects for split.
    # ground_truth = machine mode / operational state for task B1.
    raise NotImplementedError(
        f"{DATASET_ID}: raw data found at {raw_dir} but the video parser is not written yet."
    )
