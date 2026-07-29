"""model_id -> ModelWrapper class. Add a new model by adding one line here
plus a new module in this package - scripts/run_eval.py's --model flag and
evaluation/examples/*/run.py both go through this."""
from __future__ import annotations

from worldbench.models.chronos2 import Chronos2Model
from worldbench.models.cosmos3_nano import Cosmos3NanoModel
from worldbench.models.gemini import GeminiModel
from worldbench.models.gpt4o import GPT4oModel
from worldbench.models.llm_text_only import LLMTextOnlyModel
from worldbench.models.majority_class import MajorityClassModel
from worldbench.models.qwen2vl import Qwen2VLModel
from worldbench.models.rule_based import RuleBasedModel
from worldbench.models.vjepa2 import VJEPA2Model
from worldbench.models.xgboost_lightgbm import XGBoostModel

MODEL_CLASSES = {
    "cosmos3_nano": Cosmos3NanoModel,
    "qwen2vl": Qwen2VLModel,
    "vjepa2": VJEPA2Model,
    "xgboost_lightgbm": XGBoostModel,
    "chronos2": Chronos2Model,
    "rule_based": RuleBasedModel,
    "majority_class": MajorityClassModel,
    "llm_text_only": LLMTextOnlyModel,
    "gpt4o": GPT4oModel,
    "gemini": GeminiModel,
}


def get_model(model_id: str):
    if model_id not in MODEL_CLASSES:
        raise KeyError(f"No model registered for '{model_id}'. Known models: {sorted(MODEL_CLASSES)}")
    return MODEL_CLASSES[model_id]()
