from __future__ import annotations

import uuid
from datetime import datetime, timezone

from trading_engine.candle_engine.models import Candle
from trading_engine.risk.regime_filter import SignalDecision
from trading_engine.storage.db import Database


class PredictionLogger:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def log(self, decision: SignalDecision, candle: Candle) -> str:
        pred_id = str(uuid.uuid4())
        e = decision.ensemble
        f = decision.features
        await self._db.execute_async(
            """
            INSERT OR IGNORE INTO predictions (
                prediction_id, asset, timeframe,
                candle_open_time, candle_close_time,
                direction, final_score, confidence,
                regime, block_reason, is_actionable,
                mc_bias, ml_prob, drift_signal, momentum_signal,
                ma20, ma50, volatility, vol_zscore,
                created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                pred_id, candle.asset, candle.timeframe,
                candle.open_time.isoformat(), candle.close_time.isoformat(),
                decision.direction.value, e.final_score, e.confidence,
                decision.regime.value, decision.block_reason.value,
                int(decision.is_actionable),
                e.mc_bias, e.ml_prob, e.drift_signal, e.momentum_signal,
                f.ma20, f.ma50, f.volatility, f.vol_zscore,
                datetime.now(tz=timezone.utc).isoformat(),
            ),
        )
        return pred_id
