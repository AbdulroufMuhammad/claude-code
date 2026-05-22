from __future__ import annotations

from enum import Enum

from trading_engine.features.feature_engine import FeatureSet
from trading_engine.models.base_model import BaseModel


class VolatilityRegime(str, Enum):
    CALM = "CALM"
    MODERATE = "MODERATE"
    SPIKE = "SPIKE"


class VolatilityModel(BaseModel):
    """Classifies volatility regime and returns a directional-neutral P(up).

    This model does not predict direction — it returns 0.5 in all regimes.
    Its primary value is providing the regime classification used by the
    RegimeFilter to block signals.
    """

    SPIKE_THRESHOLD = 2.5
    MODERATE_THRESHOLD = 1.0

    def predict(self, features: FeatureSet) -> float:
        return 0.5  # regime-neutral

    def classify(self, features: FeatureSet) -> VolatilityRegime:
        vz = features.vol_zscore
        if vz > self.SPIKE_THRESHOLD:
            return VolatilityRegime.SPIKE
        if vz > self.MODERATE_THRESHOLD:
            return VolatilityRegime.MODERATE
        return VolatilityRegime.CALM
