from __future__ import annotations

import time
from typing import Any, Dict

from trading_engine.storage.db import Database

_CACHE_TTL = 30.0  # seconds


class MetricsService:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._cache: Dict[str, Any] = {}
        self._cache_ts: float = 0.0

    async def get_metrics(self) -> Dict[str, Any]:
        if time.time() - self._cache_ts < _CACHE_TTL and self._cache:
            return self._cache

        total_row = await self._db.fetchall_async(
            "SELECT COUNT(*) FROM predictions"
        )
        total = total_row[0][0] if total_row else 0

        actionable_row = await self._db.fetchall_async(
            "SELECT COUNT(*) FROM predictions WHERE is_actionable = 1"
        )
        actionable = actionable_row[0][0] if actionable_row else 0

        accuracy_row = await self._db.fetchall_async(
            """
            SELECT COUNT(*), SUM(o.was_correct)
            FROM outcomes o
            JOIN predictions p ON p.prediction_id = o.prediction_id
            WHERE p.is_actionable = 1
            """
        )
        resolved, correct = (accuracy_row[0][0] or 0, accuracy_row[0][1] or 0) if accuracy_row else (0, 0)
        accuracy = round(correct / resolved * 100, 2) if resolved > 0 else None

        regime_rows = await self._db.fetchall_async(
            "SELECT regime, COUNT(*) FROM predictions GROUP BY regime"
        )
        regime_breakdown = {r[0]: r[1] for r in regime_rows}

        block_rows = await self._db.fetchall_async(
            "SELECT block_reason, COUNT(*) FROM predictions GROUP BY block_reason"
        )
        block_breakdown = {r[0]: r[1] for r in block_rows}

        self._cache = {
            "total_predictions": total,
            "actionable_signals": actionable,
            "resolved_outcomes": resolved,
            "correct_predictions": correct,
            "accuracy_pct": accuracy,
            "regime_breakdown": regime_breakdown,
            "block_reason_breakdown": block_breakdown,
        }
        self._cache_ts = time.time()
        return self._cache
