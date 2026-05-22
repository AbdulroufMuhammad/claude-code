from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

TIMEFRAME_SECONDS: Dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
}

VALID_TIMEFRAMES = list(TIMEFRAME_SECONDS.keys())


@dataclass
class UserConfig:
    asset: str = "MOCK"
    timeframe: str = "15m"
    model_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "mc": 0.30,
            "ml": 0.30,
            "drift": 0.20,
            "momentum": 0.20,
        }
    )
    volatility_threshold: float = 2.5
    confidence_threshold: float = 0.60
    max_trades_per_hour: int = 4

    def validate(self) -> None:
        if self.timeframe not in VALID_TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {VALID_TIMEFRAMES}")
        w = self.model_weights
        required = {"mc", "ml", "drift", "momentum"}
        if set(w.keys()) != required:
            raise ValueError(f"model_weights must have keys {required}")
        total = sum(w.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"model_weights must sum to 1.0, got {total:.3f}")


@dataclass(frozen=True)
class Config:
    # --- Data stream ---
    WS_URL_SYNTHETIC: str = "wss://ws.derivws.com/websockets/v3"
    WS_URL_CRYPTO: str = "wss://stream.binance.com/ws"
    WS_URL_FOREX: str = "wss://api-fxtrade.oanda.com/v3/accounts/{}/pricing/stream"
    WS_RECONNECT_MAX_RETRIES: int = 20
    WS_RECONNECT_BASE_DELAY: float = 1.0
    WS_RECONNECT_MAX_DELAY: float = 60.0

    # --- Candle engine ---
    CANDLE_STORE_MAXLEN: int = 200

    # --- Feature engine ---
    MIN_CANDLES: int = 60
    MA_SHORT: int = 20
    MA_LONG: int = 50
    VOL_WINDOW: int = 20
    MOMENTUM_WINDOW: int = 10
    DRIFT_WINDOW: int = 20
    VOL_ZSCORE_WINDOW: int = 50

    # --- Monte Carlo ---
    MC_SIMULATIONS: int = 1_000
    MC_STEPS: int = 8

    # --- Ensemble thresholds ---
    BUY_THRESHOLD: float = 0.75
    SELL_THRESHOLD: float = 0.25

    # --- Regime filter ---
    SIDEWAYS_MA_DIFF: float = 0.0005
    MOMENTUM_ZERO_BAND: float = 0.02

    # --- ML model ---
    ML_MODEL_PATH: str = "trading_engine/data/ml_model.pkl"
    ML_TRAIN_MIN_SAMPLES: int = 100
    ML_USE_XGBOOST: bool = False
    ML_REFIT_INTERVAL: int = 500

    # --- Storage ---
    DB_PATH: str = "trading_engine/data/trading.db"

    # --- API ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # --- Backtest ---
    BACKTEST_DATA_PATH: str = ""

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- Mock provider ---
    MOCK_TICK_INTERVAL: float = 0.1
    MOCK_INITIAL_PRICE: float = 1000.0
    MOCK_DRIFT: float = 0.0001
    MOCK_VOLATILITY: float = 0.005


def load_config() -> Config:
    overrides: dict = {}
    prefix = "TRADING_"
    fields = Config.__dataclass_fields__
    for field_name, field_obj in fields.items():
        env_key = f"{prefix}{field_name}"
        if env_key in os.environ:
            raw = os.environ[env_key]
            ft = field_obj.type
            if ft in ("int", "int "):
                overrides[field_name] = int(raw)
            elif ft in ("float", "float "):
                overrides[field_name] = float(raw)
            elif ft in ("bool", "bool "):
                overrides[field_name] = raw.lower() in ("1", "true", "yes")
            else:
                overrides[field_name] = raw
    return Config(**overrides)
