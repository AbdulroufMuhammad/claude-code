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


class ForexProvider(MarketDataProvider):
    """WebSocket provider stub for forex pairs.

    Connects to OANDA streaming API.
    Requires OANDA_ACCOUNT_ID and OANDA_API_TOKEN environment variables.

    Swap the URL / auth logic to integrate another forex broker.
    """

    def __init__(self, asset: str, config: Config, account_id: str = "", api_token: str = "") -> None:
        self._asset = asset
        self._cfg = config
        self._account_id = account_id
        self._api_token = api_token
        self._ws: Any = None

    async def connect(self) -> None:
        logger.info("ForexProvider configured for %s", self._asset)

    async def stream(self, queue: asyncio.Queue) -> None:
        """Stream forex ticks via OANDA REST streaming endpoint.

        Falls back to mock data if no credentials are configured.
        """
        if not self._api_token or not self._account_id:
            logger.warning(
                "ForexProvider: no OANDA credentials — falling back to mock ticks for %s",
                self._asset,
            )
            from trading_engine.data_stream.mock_provider import MockProvider
            mock = MockProvider(self._asset, self._cfg)
            await mock.stream(queue)
            return

        try:
            import httpx  # type: ignore
        except ImportError:
            raise RuntimeError("httpx package required for ForexProvider")

        url = f"https://stream-fxtrade.oanda.com/v3/accounts/{self._account_id}/pricing/stream"
        headers = {"Authorization": f"Bearer {self._api_token}"}
        params = {"instruments": self._asset}

        delay = self._cfg.WS_RECONNECT_BASE_DELAY
        attempts = 0
        while attempts < self._cfg.WS_RECONNECT_MAX_RETRIES:
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("GET", url, headers=headers, params=params) as resp:
                        resp.raise_for_status()
                        delay = self._cfg.WS_RECONNECT_BASE_DELAY
                        attempts = 0
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            raw = json.loads(line)
                            if raw.get("type") != "PRICE":
                                continue
                            tick = self.normalize(raw)
                            await queue.put(tick)
            except Exception as exc:
                logger.warning("ForexProvider error: %s — retrying in %.1fs", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._cfg.WS_RECONNECT_MAX_DELAY)
                attempts += 1

    def normalize(self, raw: dict) -> Tick:
        bids = raw.get("bids", [{}])
        asks = raw.get("asks", [{}])
        bid = float(bids[0].get("price", 0)) if bids else 0.0
        ask = float(asks[0].get("price", 0)) if asks else 0.0
        mid = (bid + ask) / 2.0 if bid and ask else bid or ask
        ts_str = raw.get("time", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            ts = datetime.now(tz=timezone.utc)
        return Tick(
            asset=self._asset,
            price=mid,
            volume=1.0,
            timestamp=ts,
            tick_id=str(uuid.uuid4()),
        )
