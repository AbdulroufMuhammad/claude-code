from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException

from trading_engine.api.schemas import (
    ConfigUpdateRequest,
    HealthResponse,
    MetricsResponse,
    SignalResponse,
)
from trading_engine.config import VALID_TIMEFRAMES, UserConfig

if TYPE_CHECKING:
    from trading_engine.api.app import AppState

router = APIRouter()


def get_state(request) -> "AppState":
    return request.app.state.engine_state


@router.get("/signal", response_model=SignalResponse)
async def get_signal(request=None):
    state = get_state(request)
    signal = await state.signal_store.get()
    if signal is None:
        return SignalResponse(status="warming_up", message="Engine is still warming up")
    return SignalResponse(signal=signal)


@router.get("/health", response_model=HealthResponse)
async def health(request=None):
    state = get_state(request)
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


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(request=None):
    state = get_state(request)
    data = await state.metrics_service.get_metrics()
    return MetricsResponse(metrics=data)


@router.post("/config/update")
async def update_config(body: ConfigUpdateRequest, request=None):
    state = get_state(request)
    uc: UserConfig = state.user_config

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
