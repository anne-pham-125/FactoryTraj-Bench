#!/usr/bin/env python
"""CLI entrypoint: run one model on one dataset/task/split and score it.

Usage:
    python scripts/run_eval.py --model majority_class --dataset dummy --task B1 --split test_locked
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldbench.models.registry import get_model  # noqa: E402
from worldbench.runner import run_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model id, see configs/models.yaml")
    parser.add_argument("--dataset", required=True, help="dataset id, see configs/compat_matrix.yaml")
    parser.add_argument("--task", required=True, help="task id (B0-B10), see configs/tasks.yaml")
    parser.add_argument("--split", default="test_locked")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG)

    model = get_model(args.model)
    metrics_path = run_eval(model, args.dataset, args.task, args.split)
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
