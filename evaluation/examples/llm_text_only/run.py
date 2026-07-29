#!/usr/bin/env python
"""Run the text-only LLM (zero-shot, non-VLM) by hand:

    python evaluation/examples/llm_text_only/run.py --dataset alpi --task B0

Only ever receives the serialized text prompt, never raw video/image
input - this isolates language reasoning from vision. Inference itself is
not wired up yet - see src/worldbench/models/llm_text_only.py for the TODO.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from worldbench.models.llm_text_only import LLMTextOnlyModel  # noqa: E402
from worldbench.runner import run_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="test_locked")
    args = parser.parse_args()

    model = LLMTextOnlyModel()
    metrics_path = run_eval(model, args.dataset, args.task, args.split)
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
