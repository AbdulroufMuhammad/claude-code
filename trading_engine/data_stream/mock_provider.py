from __future__ import annotations

import asyncio
import math
import random
import uuid
from datetime import datetime, timezone

from trading_engine.candle_engine.models import Tick
from trading_engine.config import Config
from trading_engine.data_stream.base_provider import MarketDataProvider


class MockProvider(MarketDataProvider):
    """Generates synthetic GBM tick stream — no network required.

    Used as the default provider for development and testing.
    """

    def __init__(self, asset: str, config: Config, seed: int | None = None) -> None:
        self._asset = asset
        self._cfg = config
        self._price = config.MOCK_INITIAL_PRICE
        self._rng = random.Random(seed)

    async def connect(self) -> None:
        pass  # nothing to connect to

    async def stream(self, queue: asyncio.Queue) -> None:
        while True:
            tick = self._next_tick()
            await queue.put(tick)
            await asyncio.sleep(self._cfg.MOCK_TICK_INTERVAL)

    def normalize(self, raw: dict) -> Tick:
        return Tick(
            asset=raw["asset"],
            price=raw["price"],
            volume=raw["volume"],
            timestamp=raw["timestamp"],
            tick_id=raw["tick_id"],
        )

    # ------------------------------------------------------------------

    def _next_tick(self) -> Tick:
        dt = self._cfg.MOCK_TICK_INTERVAL
        drift = self._cfg.MOCK_DRIFT
        sigma = self._cfg.MOCK_VOLATILITY
        shock = self._rng.gauss(0, 1)
        self._price *= math.exp((drift - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * shock)
        return Tick(
            asset=self._asset,
            price=round(self._price, 5),
            volume=round(self._rng.uniform(0.1, 10.0), 4),
            timestamp=datetime.now(tz=timezone.utc),
            tick_id=str(uuid.uuid4()),
        )
