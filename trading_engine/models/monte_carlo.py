from __future__ import annotations

from typing import Optional

import numpy as np

from trading_engine.config import Config
from trading_engine.features.feature_engine import FeatureSet
from trading_engine.models.base_model import BaseModel


class MonteCarloModel(BaseModel):
    """Geometric Brownian Motion Monte Carlo simulator.

    Simulates MC_SIMULATIONS paths of MC_STEPS candles using the
    estimated drift and volatility from the feature set.
    Returns the fraction of paths where the final price exceeds the initial.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config

    def predict(self, features: FeatureSet, seed: Optional[int] = None) -> float:
        drift = features.drift * 0.01   # scale from tanh units to per-candle drift
        sigma = max(features.volatility, 1e-6)
        ratios = self._simulate(drift, sigma, self._cfg.MC_SIMULATIONS, self._cfg.MC_STEPS, seed)
        return float(np.mean(ratios > 1.0))

    def _simulate(
        self,
        drift: float,
        sigma: float,
        n_sims: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        shocks = rng.standard_normal((n_sims, n_steps))
        # GBM: S_t = S_0 * exp(sum((drift - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z))
        log_returns = (drift - 0.5 * sigma**2) + sigma * shocks
        cumulative = np.cumsum(log_returns, axis=1)
        final_ratio = np.exp(cumulative[:, -1])
        return final_ratio
