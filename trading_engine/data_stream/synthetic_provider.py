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


class SyntheticProvider(MarketDataProvider):
    """Live WebSocket provider for Deriv/Binary.com synthetic indices.

    Connects to wss://ws.derivws.com/websockets/v3 and subscribes to
    tick stream for the given symbol (e.g. "R_10", "R_25", "R_75").

    Requires a Deriv API token set as TRADING_DERIV_TOKEN env var.
    """

    def __init__(self, asset: str, config: Config, api_token: str = "") -> None:
        self._asset = asset
        self._cfg = config
        self._api_token = api_token
        self._ws: Any = None

    async def connect(self) -> None:
        if not _HAS_WEBSOCKETS:
            raise RuntimeError("websockets package not installed")
        logger.info("SyntheticProvider connecting to %s", self._cfg.WS_URL_SYNTHETIC)

    async def stream(self, queue: asyncio.Queue) -> None:
        if not _HAS_WEBSOCKETS:
            raise RuntimeError("websockets package not installed")
        import websockets  # type: ignore

        delay = self._cfg.WS_RECONNECT_BASE_DELAY
        attempts = 0
        while attempts < self._cfg.WS_RECONNECT_MAX_RETRIES:
            try:
                async with websockets.connect(self._cfg.WS_URL_SYNTHETIC) as ws:
                    self._ws = ws
                    delay = self._cfg.WS_RECONNECT_BASE_DELAY
                    attempts = 0
                    if self._api_token:
                        await ws.send(json.dumps({"authorize": self._api_token}))
                        await ws.recv()
                    await ws.send(json.dumps({"ticks": self._asset, "subscribe": 1}))
                    async for message in ws:
                        raw = json.loads(message)
                        if "tick" not in raw:
                            continue
                        tick = self.normalize(raw["tick"])
                        await queue.put(tick)
            except Exception as exc:
                logger.warning("SyntheticProvider disconnected: %s — retrying in %.1fs", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._cfg.WS_RECONNECT_MAX_DELAY)
                attempts += 1
        logger.error("SyntheticProvider exhausted retries — giving up")

    def normalize(self, raw: dict) -> Tick:
        ts = datetime.fromtimestamp(float(raw.get("epoch", 0)), tz=timezone.utc)
        return Tick(
            asset=self._asset,
            price=float(raw["quote"]),
            volume=1.0,
            timestamp=ts,
            tick_id=str(raw.get("id", uuid.uuid4())),
        )
