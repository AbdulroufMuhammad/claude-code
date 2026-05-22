from __future__ import annotations

import asyncio
from collections import deque
from typing import Optional

import pandas as pd

from trading_engine.candle_engine.models import Candle


class CandleStore:
    """Thread-safe (asyncio-safe) in-memory store for completed candles."""

    def __init__(self, maxlen: int = 200) -> None:
        self._deque: deque[Candle] = deque(maxlen=maxlen)
        self._lock = asyncio.Lock()
        self._candle_count = 0

    async def append(self, candle: Candle) -> None:
        async with self._lock:
            self._deque.append(candle)
            self._candle_count += 1

    async def get_dataframe(self, n: Optional[int] = None) -> pd.DataFrame:
        async with self._lock:
            candles = list(self._deque) if n is None else list(self._deque)[-n:]
        if not candles:
            return pd.DataFrame(
                columns=["open_time", "open", "high", "low", "close", "volume"]
            )
        return pd.DataFrame(
            {
                "open_time": [c.open_time for c in candles],
                "open": [c.open for c in candles],
                "high": [c.high for c in candles],
                "low": [c.low for c in candles],
                "close": [c.close for c in candles],
                "volume": [c.volume for c in candles],
            }
        )

    async def latest(self) -> Optional[Candle]:
        async with self._lock:
            return self._deque[-1] if self._deque else None

    async def count(self) -> int:
        async with self._lock:
            return len(self._deque)

    @property
    def total_processed(self) -> int:
        return self._candle_count
