from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from trading_engine.candle_engine.models import Tick


class MarketDataProvider(ABC):
    """Abstract interface for all market data sources."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def stream(self, queue: asyncio.Queue) -> None:
        """Push normalized Tick objects into queue until cancelled."""
        ...

    @abstractmethod
    def normalize(self, raw: dict) -> Tick: ...

    async def disconnect(self) -> None:
        pass
