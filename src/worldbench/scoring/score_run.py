"""Shared scoring entrypoint used by everyone: predictions.jsonl -> metrics.json.

Validates that (dataset, model, task) is declared in configs/compat_matrix.yaml
before scoring anything - an undeclared combination raises CompatibilityError
instead of silently producing a number nobody agreed was meaningful.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from worldbench.tasks.registry import get_handler
from worldbench.types import Prediction, Sample

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPAT_CONFIG = _REPO_ROOT / "configs" / "compat_matrix.yaml"

# Smoke-test-only dataset (src/worldbench/datasets/dummy.py) - deliberately
# outside the real compat matrix, see its module docstring.
DUMMY_DATASET_ID = "dummy"


class CompatibilityError(Exception):
    pass


def _load_compat_matrix(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_compat(dataset_id: str, model_id: str, task_id: str, config_path: Path = _COMPAT_CONFIG) -> None:
    if dataset_id == DUMMY_DATASET_ID:
        return

    matrix = _load_compat_matrix(config_path)
    dataset_entry = matrix.get("datasets", {}).get(dataset_id)
    if dataset_entry is None:
        raise CompatibilityError(
            f"Dataset '{dataset_id}' is not declared in {config_path}. "
            "Add it under `datasets:` before running any model against it."
        )

    for entry in dataset_entry.get("entries", []):
        if entry["model"] == model_id and task_id in entry.get("tasks", []):
            return

    raise CompatibilityError(
        f"(dataset={dataset_id}, model={model_id}, task={task_id}) is not an allowed "
        f"combination per {config_path}. Add it under datasets.{dataset_id}.entries "
        "before running this - see CONTRIBUTING.md."
    )


def _read_predictions(predictions_path: Path) -> list[Prediction]:
    predictions = []
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(Prediction(**json.loads(line)))
    return predictions


def score_run(
    predictions_path: Path,
    dataset_id: str,
    model_id: str,
    task_id: str,
    samples: list[Sample],
    config_path: Path = _COMPAT_CONFIG,
) -> Path:
    check_compat(dataset_id, model_id, task_id, config_path)

    predictions = _read_predictions(predictions_path)
    handler = get_handler(task_id)
    metrics = handler.compute_metric(predictions, samples)

    metrics_path = predictions_path.with_name(predictions_path.stem + ".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "model": model_id,
                "dataset": dataset_id,
                "task": task_id,
                "n_predictions": len(predictions),
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    return metrics_path


if __name__ == "__main__":
    import argparse

    from worldbench.datasets.registry import get_loader

    parser = argparse.ArgumentParser(description="Score an existing predictions.jsonl file")
    parser.add_argument("predictions_path", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="test_locked")
    args = parser.parse_args()

    loader = get_loader(args.dataset)
    eval_samples = [s for s in loader.load(args.split) if s.task_id == args.task]
    out_path = score_run(args.predictions_path, args.dataset, args.model, args.task, eval_samples)
    print(f"Wrote {out_path}")
