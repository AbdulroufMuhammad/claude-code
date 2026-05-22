from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from trading_engine.candle_engine.candle_store import CandleStore
from trading_engine.config import Config
from trading_engine.features import indicators


@dataclass(frozen=True)
class FeatureSet:
    ma20: float
    ma50: float
    volatility: float
    momentum: float       # ROC normalised to [-1, 1] via tanh
    drift: float          # OLS slope normalised to [-1, 1]
    vol_zscore: float
    candle_count: int


class FeatureEngine:
    def __init__(self, candle_store: CandleStore, config: Config) -> None:
        self._store = candle_store
        self._cfg = config

    async def compute(self) -> Optional[FeatureSet]:
        """Returns None during warm-up (fewer than MIN_CANDLES candles)."""
        count = await self._store.count()
        if count < self._cfg.MIN_CANDLES:
            return None

        df = await self._store.get_dataframe()
        close = df["close"]

        ma20 = indicators.ma(close, self._cfg.MA_SHORT)
        ma50 = indicators.ma(close, self._cfg.MA_LONG)
        vol = indicators.volatility(close, self._cfg.VOL_WINDOW)
        mom = indicators.roc(close, self._cfg.MOMENTUM_WINDOW)
        drift = indicators.linear_slope(close, self._cfg.DRIFT_WINDOW)
        vz = indicators.vol_zscore(close, self._cfg.VOL_WINDOW, self._cfg.VOL_ZSCORE_WINDOW)

        import math
        mom_norm = float(math.tanh(mom * 100))

        return FeatureSet(
            ma20=ma20,
            ma50=ma50,
            volatility=vol,
            momentum=mom_norm,
            drift=drift,
            vol_zscore=vz,
            candle_count=count,
        )
