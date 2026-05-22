from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Tick:
    asset: str
    price: float
    volume: float
    timestamp: datetime
    tick_id: str


@dataclass
class Candle:
    asset: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int
    is_closed: bool = True


@dataclass
class CandleState:
    asset: str
    timeframe: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int = 0
