"""IndustReal - assembly video with procedural errors.

Download: https://timschoonbeek.github.io/industreal (see data/README.md).
Not yet used in any run as of this scaffold; has more annotated errors than
Assembly101, prioritized for Fault-OOD testing.

Task coverage: B3 (anomaly/fault localization), B10 (OOD recognition) per
configs/compat_matrix.yaml. NOTE: these task ids are an inference from prose
in the planning doc ("Fault-OOD test"), not a clean B-code in the source -
confirm before treating them as final.
"""
from __future__ import annotations

from worldbench.datasets import data_root, warn_missing_data
from worldbench.types import Sample

DATASET_ID = "industreal"


def load(split: str) -> list[Sample]:
    raw_dir = data_root() / DATASET_ID
    if not raw_dir.exists():
        warn_missing_data(
            DATASET_ID,
            raw_dir,
            "download from https://timschoonbeek.github.io/industreal and place it under this path",
        )
        return []

    # TODO: parse IndustReal video + error annotations into Sample objects
    # for split. ground_truth = fault/error label for B3, OOD flag for B10.
    raise NotImplementedError(
        f"{DATASET_ID}: raw data found at {raw_dir} but the video parser is not written yet."
    )
