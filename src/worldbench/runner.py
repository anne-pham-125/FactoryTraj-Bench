"""Shared orchestration used by both scripts/run_eval.py (CLI) and every
evaluation/examples/<model>/run.py: load samples, optionally train, predict,
write raw predictions, score them. Per-model scripts only differ in which
ModelWrapper they instantiate - this is what makes them "thin".
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from worldbench.datasets.registry import get_loader
from worldbench.models.base import ModelWrapper
from worldbench.scoring.score_run import check_compat, score_run
from worldbench.tasks.registry import get_handler
from worldbench.types import Prediction

logger = logging.getLogger("worldbench.runner")

RESULTS_DIR = Path("results")


def run_eval(model: ModelWrapper, dataset_id: str, task_id: str, split: str) -> Path:
    # Checked up front, before loading data or touching the model, so an
    # undeclared (dataset, model, task) combo never gets as far as running -
    # score_run() checks it again at the end, but that's too late to prevent
    # wasted/incorrect runs on its own.
    check_compat(dataset_id, model.model_id, task_id)

    loader = get_loader(dataset_id)
    handler = get_handler(task_id)

    samples = [s for s in loader.load(split) if s.task_id == task_id]
    if not samples:
        logger.warning(
            "No samples for dataset=%s task=%s split=%s - nothing to run.",
            dataset_id, task_id, split,
        )

    # fit() is always called, even for zero-shot models: the base
    # ModelWrapper.fit() is a no-op unless needs_training is True, but some
    # "no training" models (e.g. majority-class/persistence) still need a
    # cheap fit-like step (computing the mode) that isn't "training" in the
    # research sense tracked by needs_training.
    train_samples = [s for s in loader.load("train") if s.task_id == task_id]
    model.fit(train_samples)

    predictions: list[Prediction] = []
    for sample in samples:
        prompt = handler.build_prompt(sample)
        raw_output = model.predict(sample, prompt)
        output = handler.parse_output(raw_output)
        predictions.append(
            Prediction(
                sample_id=sample.id,
                task_id=task_id,
                dataset=dataset_id,
                model=model.model_id,
                output=output,
            )
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    predictions_path = RESULTS_DIR / f"{model.model_id}__{dataset_id}__{task_id}.jsonl"
    with open(predictions_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p.__dict__) + "\n")

    metrics_path = score_run(predictions_path, dataset_id, model.model_id, task_id, samples=samples)
    logger.info("Wrote predictions to %s and metrics to %s", predictions_path, metrics_path)
    return metrics_path
