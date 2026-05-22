from __future__ import annotations

import numpy as np
import pandas as pd


def ma(series: pd.Series, n: int) -> float:
    """Simple moving average of last n values."""
    if len(series) < n:
        return float(series.mean())
    return float(series.iloc[-n:].mean())


def volatility(close: pd.Series, n: int = 20) -> float:
    """Annualized-equivalent std of log-returns over last n closes."""
    if len(close) < 2:
        return 0.0
    prices = close.iloc[-n:] if len(close) >= n else close
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) == 0:
        return 0.0
    return float(log_returns.std())


def roc(close: pd.Series, n: int = 10) -> float:
    """Rate of change: (close[-1] - close[-n]) / close[-n]."""
    if len(close) < n + 1:
        return 0.0
    return float((close.iloc[-1] - close.iloc[-n]) / close.iloc[-n])


def vol_zscore(close: pd.Series, vol_window: int = 20, zscore_window: int = 50) -> float:
    """Current volatility z-score vs rolling mean/std over zscore_window."""
    if len(close) < vol_window + 2:
        return 0.0
    log_returns = np.log(close / close.shift(1)).dropna()
    if len(log_returns) < zscore_window:
        return 0.0
    rolling_vol = log_returns.rolling(vol_window).std().dropna()
    if len(rolling_vol) < 2:
        return 0.0
    mean = float(rolling_vol.iloc[-zscore_window:].mean())
    std = float(rolling_vol.iloc[-zscore_window:].std())
    if std == 0:
        return 0.0
    current_vol = float(rolling_vol.iloc[-1])
    return (current_vol - mean) / std


def linear_slope(close: pd.Series, n: int) -> float:
    """OLS slope of last n closes, normalised to [-1, 1] via tanh."""
    if len(close) < n:
        n = len(close)
    if n < 2:
        return 0.0
    y = close.iloc[-n:].values.astype(float)
    x = np.arange(n, dtype=float)
    x -= x.mean()
    slope = float(np.dot(x, y) / np.dot(x, x))
    price_scale = float(np.abs(y).mean()) or 1.0
    normalised = slope / price_scale * n
    return float(np.tanh(normalised))
