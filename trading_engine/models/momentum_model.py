from __future__ import annotations

from trading_engine.features.feature_engine import FeatureSet
from trading_engine.models.base_model import BaseModel


class MomentumModel(BaseModel):
    """Converts ROC momentum signal into P(up)."""

    def predict(self, features: FeatureSet) -> float:
        mom = features.momentum  # tanh-normalised ∈ [-1, 1]
        return self._sigmoid(mom, k=3.0)

    @staticmethod
    def _sigmoid(x: float, k: float = 1.0) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-k * x))
