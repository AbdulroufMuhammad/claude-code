from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from trading_engine.config import UserConfig
from trading_engine.features.feature_engine import FeatureSet


class Direction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


@dataclass(frozen=True)
class EnsembleResult:
    final_score: float
    p_up: float
    p_down: float
    confidence: float
    drift_signal: float
    mc_bias: float
    ml_prob: float
    momentum_signal: float
    direction: Direction
    timestamp: datetime


class EnsembleEngine:
    def __init__(self, user_config: UserConfig) -> None:
        self._cfg = user_config

    def score(
        self,
        drift_signal: float,
        mc_bias: float,
        ml_prob: float,
        momentum_signal: float,
        features: FeatureSet,
    ) -> EnsembleResult:
        w = self._cfg.model_weights
        final_score = (
            w["mc"] * mc_bias
            + w["ml"] * ml_prob
            + w["drift"] * drift_signal
            + w["momentum"] * momentum_signal
        )
        final_score = max(0.0, min(1.0, final_score))
        confidence = abs(final_score - 0.5) * 2.0

        if final_score > 0.75:
            direction = Direction.BUY
        elif final_score < 0.25:
            direction = Direction.SELL
        else:
            direction = Direction.NO_TRADE

        return EnsembleResult(
            final_score=final_score,
            p_up=final_score,
            p_down=1.0 - final_score,
            confidence=confidence,
            drift_signal=drift_signal,
            mc_bias=mc_bias,
            ml_prob=ml_prob,
            momentum_signal=momentum_signal,
            direction=direction,
            timestamp=datetime.now(tz=timezone.utc),
        )
