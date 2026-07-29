"""V-JEPA 2 (ViT-L) - embedding-based group, trained via a linear probe on
top of frozen embeddings. needs_training = True.

TODO: wire up actual V-JEPA 2 embedding extraction - this is a structural
placeholder. Once embeddings are available, fit() trains a simple linear
probe (sklearn LogisticRegression) on top of them and predict() reuses it.
Checkpoint (probe weights) saved to checkpoints/, gitignored.

KNOWN ISSUE: prior HATREC runs with this model showed a suspicious 1.0 F1 -
traced to train/test leak. Always run scoring/leak_check.py before trusting
a result from this model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from worldbench.models.base import ModelWrapper
from worldbench.types import Sample

CHECKPOINT_DIR = Path("checkpoints")


class VJEPA2Model(ModelWrapper):
    model_id = "vjepa2"
    needs_training = True

    def __init__(self) -> None:
        self._probe = None

    def _embed(self, sample: Sample):
        raise NotImplementedError(
            f"{self.model_id}: TODO wire up V-JEPA 2 embedding extraction for {sample.dataset}."
        )

    def fit(self, train_samples: list[Sample]) -> None:
        # TODO once _embed works:
        #   from sklearn.linear_model import LogisticRegression
        #   X = [self._embed(s) for s in train_samples]
        #   y = [s.ground_truth for s in train_samples]
        #   self._probe = LogisticRegression(max_iter=1000).fit(X, y)
        #   from joblib import dump
        #   CHECKPOINT_DIR.mkdir(exist_ok=True)
        #   dump(self._probe, CHECKPOINT_DIR / f"{self.model_id}_probe.joblib")
        raise NotImplementedError(f"{self.model_id}.fit(): embedding extraction not wired up yet.")

    def predict(self, sample: Sample, prompt: str) -> Any:
        if self._probe is None:
            raise RuntimeError(f"{self.model_id}: fit() (or checkpoint load) must run before predict()")
        return self._probe.predict([self._embed(sample)])[0]
