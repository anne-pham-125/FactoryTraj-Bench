"""dataset_id -> loader module. Add a new source by adding one line here
plus a new module in this package - nothing else needs to know the list."""
from __future__ import annotations

from worldbench.datasets import (
    alpi,
    assembly101,
    dummy,
    hatrec,
    industreal,
    mimii,
    mmad,
    tep,
)

DATASET_LOADERS = {
    "tep": tep,
    "alpi": alpi,
    "mimii": mimii,
    "mmad": mmad,
    "hatrec": hatrec,
    "assembly101": assembly101,
    "industreal": industreal,
    "dummy": dummy,
}


def get_loader(dataset_id: str):
    if dataset_id not in DATASET_LOADERS:
        raise KeyError(f"No dataset loader registered for '{dataset_id}'. Known datasets: {sorted(DATASET_LOADERS)}")
    return DATASET_LOADERS[dataset_id]
