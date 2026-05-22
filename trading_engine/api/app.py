from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trading_engine.candle_engine.candle_builder import DynamicCandleEngine
from trading_engine.candle_engine.candle_store import CandleStore
from trading_engine.config import Config, UserConfig, VALID_TIMEFRAMES
from trading_engine.execution.signal_handler import SignalStore
from trading_engine.monitoring.metrics import MetricsService
from trading_engine.api.schemas import (
    ConfigUpdateRequest,
    HealthResponse,
    MetricsResponse,
    SignalResponse,
)


@dataclass
class AppState:
    config: Config
    user_config: UserConfig
    candle_store: CandleStore
    candle_engine: DynamicCandleEngine
    signal_store: SignalStore
    metrics_service: MetricsService
    mode: str = "live"
    provider_reset_requested: bool = False


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="Probabilistic Market Intelligence Engine", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/signal", response_model=SignalResponse)
    async def get_signal():
        signal = await state.signal_store.get()
        if signal is None:
            return SignalResponse(status="warming_up", message="Engine is still warming up")
        return SignalResponse(signal=signal)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        count = await state.candle_store.count()
        return HealthResponse(
            status="ok",
            mode=state.mode,
            candles_processed=state.candle_store.total_processed,
            candle_count=count,
            is_warmed_up=count >= state.config.MIN_CANDLES,
            asset=state.user_config.asset,
            timeframe=state.user_config.timeframe,
        )

    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics():
        data = await state.metrics_service.get_metrics()
        return MetricsResponse(metrics=data)

    @app.post("/config/update")
    async def update_config(body: ConfigUpdateRequest):
        from fastapi import HTTPException
        uc = state.user_config

        if body.asset is not None:
            uc.asset = body.asset.upper()
            state.provider_reset_requested = True

        if body.timeframe is not None:
            if body.timeframe not in VALID_TIMEFRAMES:
                raise HTTPException(400, detail=f"timeframe must be one of {VALID_TIMEFRAMES}")
            uc.timeframe = body.timeframe
            state.candle_engine.set_timeframe(body.timeframe)

        if body.model_weights is not None:
            uc.model_weights = body.model_weights.model_dump()

        if body.volatility_threshold is not None:
            uc.volatility_threshold = body.volatility_threshold

        if body.confidence_threshold is not None:
            uc.confidence_threshold = body.confidence_threshold

        if body.max_trades_per_hour is not None:
            uc.max_trades_per_hour = body.max_trades_per_hour

        return {"status": "ok", "config": {
            "asset": uc.asset,
            "timeframe": uc.timeframe,
            "model_weights": uc.model_weights,
            "volatility_threshold": uc.volatility_threshold,
            "confidence_threshold": uc.confidence_threshold,
            "max_trades_per_hour": uc.max_trades_per_hour,
        }}

    return app
