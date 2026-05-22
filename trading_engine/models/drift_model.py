from __future__ import annotations

from trading_engine.features.feature_engine import FeatureSet
from trading_engine.models.base_model import BaseModel


class DriftModel(BaseModel):
    """Converts OLS price slope into P(up) via sigmoid mapping.

    drift ∈ [-1, 1] (tanh-normalised from feature engine).
    """

    def predict(self, features: FeatureSet) -> float:
        slope = features.drift
        return self._sigmoid(slope, k=3.0)

    @staticmethod
    def _sigmoid(x: float, k: float = 1.0) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-k * x))
