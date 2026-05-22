from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from trading_engine.risk.regime_filter import SignalDecision

logger = logging.getLogger(__name__)


class SignalStore:
    """Thread-safe (asyncio-safe) container for the latest signal."""

    def __init__(self) -> None:
        self._latest: Optional[Dict[str, Any]] = None
        self._lock = asyncio.Lock()

    async def set(self, decision: SignalDecision) -> None:
        payload = _to_dict(decision)
        async with self._lock:
            self._latest = payload

    async def get(self) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._latest


def _to_dict(decision: SignalDecision) -> Dict[str, Any]:
    e = decision.ensemble
    f = decision.features
    return {
        "timestamp": decision.timestamp.isoformat(),
        "direction": decision.direction.value,
        "p_up": round(e.p_up, 4),
        "p_down": round(e.p_down, 4),
        "final_score": round(e.final_score, 4),
        "confidence": round(e.confidence, 4),
        "regime": decision.regime.value,
        "block_reason": decision.block_reason.value,
        "is_actionable": decision.is_actionable,
        "model_details": {
            "drift": round(e.drift_signal, 4),
            "mc": round(e.mc_bias, 4),
            "ml": round(e.ml_prob, 4),
            "momentum": round(e.momentum_signal, 4),
        },
        "features": {
            "ma20": round(f.ma20, 6),
            "ma50": round(f.ma50, 6),
            "volatility": round(f.volatility, 6),
            "momentum": round(f.momentum, 4),
            "drift": round(f.drift, 4),
            "vol_zscore": round(f.vol_zscore, 4),
            "candle_count": f.candle_count,
        },
    }
