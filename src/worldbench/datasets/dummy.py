"""Synthetic dataset used only to prove the run_eval.py pipeline works
end-to-end (Phase 7 smoke test). Not one of the 7 real data sources - not
listed in configs/compat_matrix.yaml on purpose, so score_run.py's compat
check has to special-case it (see ALLOW_DUMMY in scoring/score_run.py).
"""
from __future__ import annotations

from worldbench.types import Sample

DATASET_ID = "dummy"

_MODES = ["idle", "running", "fault"]


def load(split: str) -> list[Sample]:
    samples = []
    for i in range(12):
        mode = _MODES[i % len(_MODES)]
        samples.append(
            Sample(
                id=f"{split}-{i:03d}",
                task_id="B1",
                dataset=DATASET_ID,
                raw_input={"window_seconds": 60, "sensor_mean": float(i)},
                ground_truth=mode,
                metadata={"split": split},
            )
        )
    return samples
