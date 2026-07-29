#!/usr/bin/env python
"""Run rule-based or majority-class/persistence baseline by hand:

    python evaluation/examples/baselines_simple/run.py --model majority_class --dataset dummy --task B1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from worldbench.models.majority_class import MajorityClassModel  # noqa: E402
from worldbench.models.rule_based import RuleBasedModel  # noqa: E402
from worldbench.runner import run_eval  # noqa: E402

MODELS = {
    "majority_class": MajorityClassModel,
    "rule_based": RuleBasedModel,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="test_locked")
    args = parser.parse_args()

    model = MODELS[args.model]()
    metrics_path = run_eval(model, args.dataset, args.task, args.split)
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
