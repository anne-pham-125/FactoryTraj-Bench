#!/usr/bin/env python
"""Run XGBoost/LightGBM or Chronos-2 (specialized baseline group) by hand:

    python evaluation/examples/xgboost_chronos/run.py --model xgboost_lightgbm --dataset tep --task B1

Feature extraction is dataset-specific and not wired up yet - see
src/worldbench/models/xgboost_lightgbm.py and chronos2.py for the TODOs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from worldbench.models.chronos2 import Chronos2Model  # noqa: E402
from worldbench.models.xgboost_lightgbm import XGBoostModel  # noqa: E402
from worldbench.runner import run_eval  # noqa: E402

MODELS = {
    "xgboost_lightgbm": XGBoostModel,
    "chronos2": Chronos2Model,
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
