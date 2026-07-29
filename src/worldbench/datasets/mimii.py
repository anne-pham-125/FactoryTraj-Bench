"""MIMII - machine audio, the only audio source in this suite.

Download: https://zenodo.org/record/3384388
Domain-shift variant: https://zenodo.org/records/4740355
(see data/README.md).

Task coverage: B3 (anomaly/fault localization, via machine sound), low
priority per the planning doc.
"""
from __future__ import annotations

from worldbench.datasets import data_root, warn_missing_data
from worldbench.types import Sample

DATASET_ID = "mimii"


def load(split: str) -> list[Sample]:
    raw_dir = data_root() / DATASET_ID
    if not raw_dir.exists():
        warn_missing_data(
            DATASET_ID,
            raw_dir,
            "download from https://zenodo.org/record/3384388 (or the domain-shift "
            "variant at https://zenodo.org/records/4740355) and place it under this path",
        )
        return []

    # TODO: parse MIMII wav files into Sample objects for split. ground_truth
    # = normal/anomalous label (+ machine type/id) for task B3. Models that
    # need spectrogram-as-image input should do that conversion in their own
    # ModelWrapper, not here - raw_input stays as the audio path/array.
    raise NotImplementedError(
        f"{DATASET_ID}: raw data found at {raw_dir} but the audio parser is not written yet."
    )
