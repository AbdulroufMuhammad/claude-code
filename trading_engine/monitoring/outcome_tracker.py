from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Tuple

from trading_engine.candle_engine.models import Candle
from trading_engine.storage.db import Database

logger = logging.getLogger(__name__)

# (prediction_id, candle_close_time, direction_predicted)
_PendingEntry = Tuple[str, datetime, str]


class OutcomeTracker:
    """Resolves prediction outcomes when the next candle closes."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._pending: Deque[_PendingEntry] = deque()

    def register(self, prediction_id: str, candle_close_time: datetime, direction: str) -> None:
        """Called right after a prediction is logged."""
        self._pending.append((prediction_id, candle_close_time, direction))

    async def on_candle_close(self, candle: Candle) -> None:
        """Called on each new candle close. Resolves pending predictions."""
        now = candle.close_time
        resolved: list[_PendingEntry] = []
        remaining: Deque[_PendingEntry] = deque()

        for entry in self._pending:
            pred_id, close_time, predicted_dir = entry
            if now >= close_time:
                resolved.append(entry)
            else:
                remaining.append(entry)

        self._pending = remaining

        for pred_id, _, predicted_dir in resolved:
            actual = "UP" if candle.close >= candle.open else "DOWN"
            was_correct = int(
                (predicted_dir == "BUY" and actual == "UP")
                or (predicted_dir == "SELL" and actual == "DOWN")
            )
            try:
                await self._db.execute_async(
                    """
                    INSERT INTO outcomes (prediction_id, actual_direction, was_correct, resolved_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (pred_id, actual, was_correct, datetime.now(tz=timezone.utc).isoformat()),
                )
            except Exception as exc:
                logger.warning("OutcomeTracker write failed: %s", exc)
