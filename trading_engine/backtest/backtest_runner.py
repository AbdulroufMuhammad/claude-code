from __future__ import annotations

import asyncio
import logging
from typing import List

from trading_engine.candle_engine.candle_store import CandleStore
from trading_engine.candle_engine.models import Candle
from trading_engine.config import Config, UserConfig
from trading_engine.ensemble.ensemble_engine import EnsembleEngine
from trading_engine.execution.signal_handler import SignalStore
from trading_engine.features.feature_engine import FeatureEngine
from trading_engine.models.drift_model import DriftModel
from trading_engine.models.ml_model import MLModel
from trading_engine.models.momentum_model import MomentumModel
from trading_engine.models.monte_carlo import MonteCarloModel
from trading_engine.monitoring.outcome_tracker import OutcomeTracker
from trading_engine.monitoring.prediction_logger import PredictionLogger
from trading_engine.risk.regime_filter import RegimeFilter
from trading_engine.storage.db import Database

logger = logging.getLogger(__name__)


class BacktestRunner:
    def __init__(
        self,
        candles: List[Candle],
        config: Config,
        user_config: UserConfig,
        db: Database,
    ) -> None:
        self._candles = candles
        self._cfg = config
        self._user_cfg = user_config
        self._db = db

    async def run(self) -> None:
        candle_store = CandleStore(maxlen=self._cfg.CANDLE_STORE_MAXLEN)
        feature_engine = FeatureEngine(candle_store, self._cfg)
        drift_model = DriftModel()
        momentum_model = MomentumModel()
        mc_model = MonteCarloModel(self._cfg)
        ml_model = MLModel(self._cfg)
        ensemble_engine = EnsembleEngine(self._user_cfg)
        regime_filter = RegimeFilter(self._user_cfg)
        signal_store = SignalStore()
        logger_svc = PredictionLogger(self._db)
        outcome_tracker = OutcomeTracker(self._db)

        signals = []

        for idx, candle in enumerate(self._candles):
            await candle_store.append(candle)
            features = await feature_engine.compute()

            if features is None:
                remaining = self._cfg.MIN_CANDLES - await candle_store.count()
                if idx % 10 == 0:
                    logger.info("Warming up — %d candles until ready", max(0, remaining))
                continue

            drift_sig = drift_model.predict(features)
            mc_bias = mc_model.predict(features, seed=idx)
            ml_prob = ml_model.predict(features)
            momentum_sig = momentum_model.predict(features)

            ensemble = ensemble_engine.score(drift_sig, mc_bias, ml_prob, momentum_sig, features)
            decision = regime_filter.evaluate(ensemble, features)
            await signal_store.set(decision)

            pred_id = await logger_svc.log(decision, candle)
            outcome_tracker.register(pred_id, candle.close_time, decision.direction.value)

            # Resolve outcomes from previous candles
            await outcome_tracker.on_candle_close(candle)

            # Accumulate ML training sample from this candle (label known at next candle)
            if idx > 0:
                prev_candle = self._candles[idx - 1]
                label = 1 if candle.close >= candle.open else 0
                ml_model.add_sample(features, label)
                if len(ml_model._samples) >= self._cfg.ML_TRAIN_MIN_SAMPLES:
                    ml_model.maybe_fit()

            signals.append({
                "candle": idx,
                "direction": decision.direction.value,
                "score": round(ensemble.final_score, 4),
                "regime": decision.regime.value,
                "is_actionable": decision.is_actionable,
            })

            logger.info(
                "Candle %d | %-8s | score=%.3f | regime=%-9s | block=%s",
                idx,
                decision.direction.value,
                ensemble.final_score,
                decision.regime.value,
                decision.block_reason.value,
            )

        logger.info("Backtest complete. %d candles processed, %d signals emitted.", len(self._candles), len(signals))
        actionable = [s for s in signals if s["is_actionable"]]
        logger.info("Actionable signals: %d", len(actionable))
