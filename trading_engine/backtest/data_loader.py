from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator, List

import pandas as pd

from trading_engine.candle_engine.models import Candle


def load_candles_from_csv(path: str, asset: str = "BACKTEST", timeframe: str = "15m") -> List[Candle]:
    """Load candles from a CSV file.

    Expected columns: open_time, open, high, low, close, volume
    open_time can be ISO8601 string or Unix timestamp (seconds).
    """
    df = pd.read_csv(path)
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if "open_time" not in df.columns:
        df["open_time"] = pd.RangeIndex(len(df))

    if "volume" not in df.columns:
        df["volume"] = 1.0

    candles: List[Candle] = []
    for _, row in df.iterrows():
        try:
            ot = _parse_timestamp(row["open_time"])
        except Exception:
            ot = datetime.now(tz=timezone.utc)

        candles.append(Candle(
            asset=asset,
            timeframe=timeframe,
            open_time=ot,
            close_time=ot,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            tick_count=1,
        ))
    return candles


def _parse_timestamp(value) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    s = str(value).strip()
    if s.isdigit():
        return datetime.fromtimestamp(int(s), tz=timezone.utc)
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
