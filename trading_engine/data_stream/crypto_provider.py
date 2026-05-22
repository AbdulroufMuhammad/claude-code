from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from trading_engine.candle_engine.models import Tick
from trading_engine.config import Config
from trading_engine.data_stream.base_provider import MarketDataProvider

logger = logging.getLogger(__name__)

try:
    import websockets
    _HAS_WEBSOCKETS = True
except ImportError:
    _HAS_WEBSOCKETS = False


class CryptoProvider(MarketDataProvider):
    """Live WebSocket provider for Binance trade stream.

    Connects to wss://stream.binance.com/ws/<symbol>@trade.
    No API key required for public market data.

    asset should be lowercase symbol, e.g. "btcusdt".
    """

    def __init__(self, asset: str, config: Config) -> None:
        self._asset = asset.lower()
        self._cfg = config
        self._ws: Any = None

    @property
    def _stream_url(self) -> str:
        return f"{self._cfg.WS_URL_CRYPTO}/{self._asset}@trade"

    async def connect(self) -> None:
        if not _HAS_WEBSOCKETS:
            raise RuntimeError("websockets package not installed")
        logger.info("CryptoProvider connecting to %s", self._stream_url)

    async def stream(self, queue: asyncio.Queue) -> None:
        if not _HAS_WEBSOCKETS:
            raise RuntimeError("websockets package not installed")
        import websockets  # type: ignore

        delay = self._cfg.WS_RECONNECT_BASE_DELAY
        attempts = 0
        while attempts < self._cfg.WS_RECONNECT_MAX_RETRIES:
            try:
                async with websockets.connect(self._stream_url) as ws:
                    self._ws = ws
                    delay = self._cfg.WS_RECONNECT_BASE_DELAY
                    attempts = 0
                    async for message in ws:
                        raw = json.loads(message)
                        if raw.get("e") != "trade":
                            continue
                        tick = self.normalize(raw)
                        await queue.put(tick)
            except Exception as exc:
                logger.warning("CryptoProvider disconnected: %s — retrying in %.1fs", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._cfg.WS_RECONNECT_MAX_DELAY)
                attempts += 1
        logger.error("CryptoProvider exhausted retries — giving up")

    def normalize(self, raw: dict) -> Tick:
        ts = datetime.fromtimestamp(raw["T"] / 1000.0, tz=timezone.utc)
        return Tick(
            asset=self._asset.upper(),
            price=float(raw["p"]),
            volume=float(raw["q"]),
            timestamp=ts,
            tick_id=str(raw.get("t", uuid.uuid4())),
        )
