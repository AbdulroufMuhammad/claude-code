from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from trading_engine.config import UserConfig
from trading_engine.ensemble.ensemble_engine import Direction, EnsembleResult
from trading_engine.features.feature_engine import FeatureSet


class BlockReason(str, Enum):
    NONE = "NONE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    SIDEWAYS = "SIDEWAYS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    FREQUENCY_LIMIT = "FREQUENCY_LIMIT"


class Regime(str, Enum):
    TRENDING = "TRENDING"
    SIDEWAYS = "SIDEWAYS"
    SPIKE = "SPIKE"


@dataclass(frozen=True)
class SignalDecision:
    direction: Direction
    block_reason: BlockReason
    regime: Regime
    ensemble: EnsembleResult
    features: FeatureSet
    is_actionable: bool
    timestamp: datetime


class RegimeFilter:
    def __init__(self, user_config: UserConfig) -> None:
        self._cfg = user_config
        self._signal_times: deque[datetime] = deque()

    def evaluate(self, ensemble: EnsembleResult, features: FeatureSet) -> SignalDecision:
        regime = self._classify_regime(features)
        block = self._check_blocks(ensemble, features, regime)
        is_actionable = block == BlockReason.NONE and ensemble.direction != Direction.NO_TRADE

        if is_actionable and ensemble.direction != Direction.NO_TRADE:
            self._signal_times.append(datetime.now(tz=timezone.utc))

        direction = ensemble.direction
        if block != BlockReason.NONE:
            direction = Direction.NO_TRADE

        return SignalDecision(
            direction=direction,
            block_reason=block,
            regime=regime,
            ensemble=ensemble,
            features=features,
            is_actionable=is_actionable,
            timestamp=datetime.now(tz=timezone.utc),
        )

    # ------------------------------------------------------------------

    def _classify_regime(self, features: FeatureSet) -> Regime:
        if features.vol_zscore > self._cfg.volatility_threshold:
            return Regime.SPIKE
        ma_diff = abs(features.ma20 / features.ma50 - 1.0) if features.ma50 != 0 else 0.0
        if (
            ma_diff < 0.0005
            and abs(features.momentum) < 0.02
        ):
            return Regime.SIDEWAYS
        return Regime.TRENDING

    def _check_blocks(
        self,
        ensemble: EnsembleResult,
        features: FeatureSet,
        regime: Regime,
    ) -> BlockReason:
        if regime == Regime.SPIKE:
            return BlockReason.HIGH_VOLATILITY
        if regime == Regime.SIDEWAYS:
            return BlockReason.SIDEWAYS
        if ensemble.confidence < self._cfg.confidence_threshold:
            return BlockReason.LOW_CONFIDENCE
        if self._over_frequency_limit():
            return BlockReason.FREQUENCY_LIMIT
        return BlockReason.NONE

    def _over_frequency_limit(self) -> bool:
        now = datetime.now(tz=timezone.utc)
        # Purge entries older than 1 hour
        while self._signal_times and (now - self._signal_times[0]).total_seconds() > 3600:
            self._signal_times.popleft()
        return len(self._signal_times) >= self._cfg.max_trades_per_hour
