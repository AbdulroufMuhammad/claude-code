from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, field_validator


class ModelWeights(BaseModel):
    mc: float = 0.30
    ml: float = 0.30
    drift: float = 0.20
    momentum: float = 0.20

    @field_validator("mc", "ml", "drift", "momentum")
    @classmethod
    def between_zero_one(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("weight must be between 0 and 1")
        return v


class ConfigUpdateRequest(BaseModel):
    asset: Optional[str] = None
    timeframe: Optional[str] = None
    model_weights: Optional[ModelWeights] = None
    volatility_threshold: Optional[float] = None
    confidence_threshold: Optional[float] = None
    max_trades_per_hour: Optional[int] = None


class SignalResponse(BaseModel):
    signal: Optional[Dict[str, Any]] = None
    status: str = "ok"
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    mode: str
    candles_processed: int
    candle_count: int
    is_warmed_up: bool
    asset: str
    timeframe: str


class MetricsResponse(BaseModel):
    metrics: Dict[str, Any]
    status: str = "ok"
