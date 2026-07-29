"""B8: Recovery action ranking.
Input: state, candidate actions and history.
Output: ranked actions and expected recovery outcomes.
Primary metric: recall_at_k. Secondary: ndcg, regret.

No dataset currently supplies ground-truth recovery-action rankings (see
configs/compat_matrix.yaml - B8 has no dataset entries yet).
"""
from __future__ import annotations

import logging

from worldbench.tasks import metrics
from worldbench.types import Prediction, Sample

TASK_ID = "B8"
logger = logging.getLogger("worldbench.tasks.b8")


def build_prompt(sample: Sample) -> str:
    return (
        "Given the current state, candidate actions, and history, rank the "
        "actions by expected recovery outcome.\n\n"
        f"Input: {sample.raw_input}"
    )


def parse_output(raw_output: str) -> list[str]:
    return [line.strip() for line in raw_output.splitlines() if line.strip()]


def compute_metric(predictions: list[Prediction], samples: list[Sample]) -> dict:
    gt_by_id = {s.id: s.ground_truth for s in samples}
    y_true = [gt_by_id[p.sample_id] for p in predictions]
    ranked_preds = [p.output if isinstance(p.output, list) else [p.output] for p in predictions]
    logger.warning(
        "B8 regret is not computable without a simulator/environment for expected "
        "recovery outcomes - returning NaN for it; recall@k/ndcg use ranking only."
    )
    return {
        "recall_at_5": metrics.recall_at_k(y_true, ranked_preds, k=5),
        "regret": float("nan"),
    }
