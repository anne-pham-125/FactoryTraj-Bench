#!/usr/bin/env python
"""Run V-JEPA 2 + linear probe (embedding-based group) by hand:

    python evaluation/examples/vjepa2/run.py --dataset hatrec --task B1

Embedding extraction is not wired up yet - see src/worldbench/models/vjepa2.py
for the TODO. NOTE: prior HATREC runs with this model showed a suspicious
1.0 F1 traced to train/test leak - always check scoring/leak_check.py output
before trusting a result here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from worldbench.models.vjepa2 import VJEPA2Model  # noqa: E402
from worldbench.runner import run_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="test_locked")
    args = parser.parse_args()

    model = VJEPA2Model()
    metrics_path = run_eval(model, args.dataset, args.task, args.split)
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
