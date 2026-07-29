"""Assembly101 - assembly video with annotated mistake/error events.

Download: https://assembly-101.github.io/ (see data/README.md). Download the
FULL dataset, not the sample subset - the sample subset has only one real
mistake event (n=1), too few to draw conclusions from.

Task coverage: B3 (anomaly/fault localization) per configs/compat_matrix.yaml.
NOTE: this task id is an inference from prose in the planning doc ("anomaly
detection", "mistake event"), not a clean B-code in the source - confirm
before treating it as final.
"""
from __future__ import annotations

from worldbench.datasets import data_root, warn_missing_data
from worldbench.types import Sample

DATASET_ID = "assembly101"


def load(split: str) -> list[Sample]:
    raw_dir = data_root() / DATASET_ID
    if not raw_dir.exists():
        warn_missing_data(
            DATASET_ID,
            raw_dir,
            "download the FULL dataset from https://assembly-101.github.io/ and place it under this path",
        )
        return []

    # TODO: parse Assembly101 video + mistake-event annotations into Sample
    # objects for split. ground_truth = mistake event interval/type for task B3.
    raise NotImplementedError(
        f"{DATASET_ID}: raw data found at {raw_dir} but the video parser is not written yet."
    )
