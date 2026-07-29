"""MMAD - static industrial defect images (8,366 images, 38 product types,
39,672 MCQ questions). Published at ICLR 2025; NOT our data, respect their
license and credit the original authors wherever results are reported.

Repo: https://github.com/jam-cc/MMAD
Dataset: https://huggingface.co/datasets/jiang-cc/MMAD

License: academic per original authors - confirm before any commercial use.
This module only loads it as one of our 7 data sources; it does not
reimplement MMAD's own evaluation code.

Task coverage: B3 (anomaly detection on product images) per
configs/compat_matrix.yaml.
"""
from __future__ import annotations

from worldbench.datasets import data_root, warn_missing_data
from worldbench.types import Sample

DATASET_ID = "mmad"


def load(split: str) -> list[Sample]:
    raw_dir = data_root() / DATASET_ID
    if not raw_dir.exists():
        warn_missing_data(
            DATASET_ID,
            raw_dir,
            "download via https://huggingface.co/datasets/jiang-cc/MMAD and place it under this path",
        )
        return []

    # TODO: load MMAD's MCQ questions + images into Sample objects for split.
    # ground_truth = correct MCQ answer for task B3. Preserve MMAD's own
    # question/sub-task metadata in Sample.metadata so results can be
    # compared against their published per-sub-task breakdown.
    raise NotImplementedError(
        f"{DATASET_ID}: raw data found at {raw_dir} but the loader is not written yet."
    )
