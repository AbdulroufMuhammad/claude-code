from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from trading_engine.candle_engine.models import Tick

_MAX_CACHE = 10_000


class TickDeduplicator:
    """Drops ticks whose tick_id has already been seen."""

    def __init__(self, maxsize: int = _MAX_CACHE) -> None:
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._maxsize = maxsize

    def is_duplicate(self, tick: Tick) -> bool:
        if tick.tick_id in self._seen:
            return True
        self._seen[tick.tick_id] = None
        if len(self._seen) > self._maxsize:
            self._seen.popitem(last=False)
        return False

    def check(self, tick: Tick) -> Optional[Tick]:
        """Return tick if unique, None if duplicate."""
        return None if self.is_duplicate(tick) else tick
