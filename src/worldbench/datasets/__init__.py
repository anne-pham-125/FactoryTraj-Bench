"""Shared helpers for dataset loader modules.

Every module in this package exposes one function: `load(split) -> list[Sample]`.
None of them should invent their own path conventions or logging setup - use
these helpers so behavior (env var name, warning format) stays consistent.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("worldbench.datasets")

DATA_ROOT_ENV_VAR = "WORLDBENCH_DATA_ROOT"


def data_root() -> Path:
    return Path(os.environ.get(DATA_ROOT_ENV_VAR, "data"))


def warn_missing_data(dataset_id: str, expected_path: Path, download_hint: str) -> None:
    logger.warning(
        "[%s] raw data not found at %s. TODO: download it first - %s. "
        "Returning an empty sample list until the data is present.",
        dataset_id,
        expected_path,
        download_hint,
    )
