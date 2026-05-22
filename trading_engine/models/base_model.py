from __future__ import annotations

from abc import ABC, abstractmethod

from trading_engine.features.feature_engine import FeatureSet


class BaseModel(ABC):
    """All models must implement this interface."""

    @abstractmethod
    def predict(self, features: FeatureSet) -> float:
        """Return P(next candle closes UP) in [0.0, 1.0]."""
        ...
