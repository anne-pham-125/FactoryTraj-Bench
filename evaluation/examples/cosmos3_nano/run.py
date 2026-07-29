#!/usr/bin/env python
"""Run Cosmos3-Nano Reasoner (MLLM zero-shot group) by hand:

    python evaluation/examples/cosmos3_nano/run.py --dataset hatrec --task B1

Inference itself is not wired up yet - see
src/worldbench/models/cosmos3_nano.py for the TODO.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from worldbench.models.cosmos3_nano import Cosmos3NanoModel  # noqa: E402
from worldbench.runner import run_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="test_locked")
    args = parser.parse_args()

    model = Cosmos3NanoModel()
    metrics_path = run_eval(model, args.dataset, args.task, args.split)
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
