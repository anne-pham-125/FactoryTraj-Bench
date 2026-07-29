"""XGBoost/LightGBM specialized baseline. needs_training = True.

Feature extraction from Sample.raw_input is dataset-specific (TEP sensor
window vs ALPI alarm log), so fit()/predict() raise a clear error until
that's wired up per dataset - see configs/compat_matrix.yaml for which
datasets this model is declared against (tep, alpi).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from worldbench.models.base import ModelWrapper
from worldbench.types import Sample

CHECKPOINT_DIR = Path("checkpoints")


class XGBoostModel(ModelWrapper):
    model_id = "xgboost_lightgbm"
    needs_training = True

    def __init__(self) -> None:
        self._clf = None

    def fit(self, train_samples: list[Sample]) -> None:
        # TODO: build a feature matrix from sample.raw_input (dataset-specific),
        # then e.g.:
        #   import xgboost as xgb
        #   self._clf = xgb.XGBClassifier().fit(X_train, y_train)
        #   from joblib import dump
        #   CHECKPOINT_DIR.mkdir(exist_ok=True)
        #   dump(self._clf, CHECKPOINT_DIR / f"{self.model_id}.joblib")
        raise NotImplementedError(
            f"{self.model_id}.fit(): feature extraction is dataset-specific - wire it up "
            "for the target dataset before training."
        )

    def predict(self, sample: Sample, prompt: str) -> Any:
        if self._clf is None:
            raise RuntimeError(f"{self.model_id}: fit() (or checkpoint load) must run before predict()")
        raise NotImplementedError(f"{self.model_id}.predict(): feature extraction not wired up yet.")
