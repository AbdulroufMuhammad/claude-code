from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from trading_engine.candle_engine.models import Candle, CandleState, Tick
from trading_engine.config import TIMEFRAME_SECONDS, Config


class DynamicCandleEngine:
    """Aggregates ticks into OHLC candles for a user-defined timeframe.

    Stateless between candles except for the in-progress CandleState.
    Safe to call from a single asyncio task — no locking needed.
    """

    def __init__(self, timeframe: str, config: Config) -> None:
        self._config = config
        self._timeframe = timeframe
        self._period_seconds = TIMEFRAME_SECONDS[timeframe]
        self._state: Optional[CandleState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def timeframe(self) -> str:
        return self._timeframe

    def set_timeframe(self, timeframe: str) -> None:
        """Hot-switch timeframe — flushes any in-progress candle."""
        self._timeframe = timeframe
        self._period_seconds = TIMEFRAME_SECONDS[timeframe]
        self._state = None

    def on_tick(self, tick: Tick) -> Optional[Candle]:
        """Process a tick. Returns a closed Candle when the window closes."""
        window_open = self._floor_to_window(tick.timestamp)

        if self._state is None:
            self._state = self._new_state(tick, window_open)
            return None

        if window_open > self._state.open_time:
            closed = self._close_candle(tick.timestamp)
            self._state = self._new_state(tick, window_open)
            return closed

        self._update_state(tick)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _floor_to_window(self, ts: datetime) -> datetime:
        epoch = ts.timestamp()
        floored = (epoch // self._period_seconds) * self._period_seconds
        return datetime.fromtimestamp(floored, tz=timezone.utc)

    def _new_state(self, tick: Tick, window_open: datetime) -> CandleState:
        return CandleState(
            asset=tick.asset,
            timeframe=self._timeframe,
            open_time=window_open,
            open=tick.price,
            high=tick.price,
            low=tick.price,
            close=tick.price,
            volume=tick.volume,
            tick_count=1,
        )

    def _update_state(self, tick: Tick) -> None:
        s = self._state
        s.high = max(s.high, tick.price)
        s.low = min(s.low, tick.price)
        s.close = tick.price
        s.volume += tick.volume
        s.tick_count += 1

    def _close_candle(self, close_ts: datetime) -> Candle:
        s = self._state
        # close_time is the tick that triggered the new window — use window_open - 1s
        close_time = datetime.fromtimestamp(
            s.open_time.timestamp() + self._period_seconds, tz=timezone.utc
        )
        return Candle(
            asset=s.asset,
            timeframe=s.timeframe,
            open_time=s.open_time,
            close_time=close_time,
            open=s.open,
            high=s.high,
            low=s.low,
            close=s.close,
            volume=s.volume,
            tick_count=s.tick_count,
        )
