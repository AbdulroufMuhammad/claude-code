from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from trading_engine.api.app import AppState, create_app
from trading_engine.candle_engine.candle_builder import DynamicCandleEngine
from trading_engine.candle_engine.candle_store import CandleStore
from trading_engine.config import Config, UserConfig, load_config
from trading_engine.data_stream.tick_deduplicator import TickDeduplicator
from trading_engine.data_stream.tick_normalizer import get_provider
from trading_engine.ensemble.ensemble_engine import EnsembleEngine
from trading_engine.execution.signal_handler import SignalStore
from trading_engine.features.feature_engine import FeatureEngine
from trading_engine.models.drift_model import DriftModel
from trading_engine.models.ml_model import MLModel
from trading_engine.models.momentum_model import MomentumModel
from trading_engine.models.monte_carlo import MonteCarloModel
from trading_engine.monitoring.metrics import MetricsService
from trading_engine.monitoring.outcome_tracker import OutcomeTracker
from trading_engine.monitoring.prediction_logger import PredictionLogger
from trading_engine.risk.regime_filter import RegimeFilter
from trading_engine.storage.db import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("trading_engine")


async def main_live(config: Config, user_config: UserConfig) -> None:
    db = Database(config.DB_PATH)
    db.connect()

    candle_store = CandleStore(maxlen=config.CANDLE_STORE_MAXLEN)
    candle_engine = DynamicCandleEngine(user_config.timeframe, config)
    deduplicator = TickDeduplicator()

    feature_engine = FeatureEngine(candle_store, config)
    drift_model = DriftModel()
    momentum_model = MomentumModel()
    mc_model = MonteCarloModel(config)
    ml_model = MLModel(config)
    signal_store = SignalStore()
    pred_logger = PredictionLogger(db)
    outcome_tracker = OutcomeTracker(db)
    metrics_service = MetricsService(db)

    state = AppState(
        config=config,
        user_config=user_config,
        candle_store=candle_store,
        candle_engine=candle_engine,
        signal_store=signal_store,
        metrics_service=metrics_service,
        mode="live",
    )

    tick_queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)
    candle_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    provider = get_provider(user_config, config)
    await provider.connect()

    async def feed_task():
        await provider.stream(tick_queue)

    async def tick_processor():
        while True:
            tick = await tick_queue.get()
            unique = deduplicator.check(tick)
            if unique is None:
                continue
            candle = candle_engine.on_tick(unique)
            if candle is not None:
                await candle_store.append(candle)
                await candle_queue.put(candle)

    async def decision_loop():
        ensemble_engine = EnsembleEngine(user_config)
        regime_filter = RegimeFilter(user_config)
        candle_count = 0

        while True:
            candle = await candle_queue.get()
            candle_count += 1

            features = await feature_engine.compute()
            if features is None:
                logger.info("Warming up (%d/%d candles)...", candle_count, config.MIN_CANDLES)
                continue

            # Rebuild ensemble/regime with current user config (may have been updated)
            ensemble_engine = EnsembleEngine(user_config)
            regime_filter = RegimeFilter(user_config)

            drift_sig = drift_model.predict(features)
            mc_bias = mc_model.predict(features)
            ml_prob = ml_model.predict(features)
            mom_sig = momentum_model.predict(features)

            ensemble = ensemble_engine.score(drift_sig, mc_bias, ml_prob, mom_sig, features)
            decision = regime_filter.evaluate(ensemble, features)

            await signal_store.set(decision)
            pred_id = await pred_logger.log(decision, candle)
            outcome_tracker.register(pred_id, candle.close_time, decision.direction.value)
            await outcome_tracker.on_candle_close(candle)

            # ML training sample
            ml_model.add_sample(features, 1 if candle.close >= candle.open else 0)
            if len(ml_model._samples) % config.ML_REFIT_INTERVAL == 0 and len(ml_model._samples) > 0:
                asyncio.create_task(_refit_ml(ml_model))

            logger.info(
                "SIGNAL | %-8s | score=%.3f | conf=%.3f | regime=%-9s | %s",
                decision.direction.value,
                ensemble.final_score,
                ensemble.confidence,
                decision.regime.value,
                decision.block_reason.value,
            )

    async def _refit_ml(model: MLModel) -> None:
        loop = asyncio.get_event_loop()
        fitted = await loop.run_in_executor(None, model.maybe_fit)
        if fitted:
            logger.info("MLModel refitted successfully")

    app = create_app(state)

    import uvicorn
    server_config = uvicorn.Config(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="warning",
        loop="none",
    )
    server = uvicorn.Server(server_config)

    await asyncio.gather(
        feed_task(),
        tick_processor(),
        decision_loop(),
        server.serve(),
        return_exceptions=True,
    )


async def main_backtest(config: Config, user_config: UserConfig, data_path: str) -> None:
    from trading_engine.backtest.backtest_runner import BacktestRunner
    from trading_engine.backtest.data_loader import load_candles_from_csv

    db = Database(config.DB_PATH)
    db.connect()

    candles = load_candles_from_csv(data_path, asset=user_config.asset, timeframe=user_config.timeframe)
    logger.info("Loaded %d candles from %s", len(candles), data_path)

    runner = BacktestRunner(candles, config, user_config, db)
    await runner.run()
    db.close()


def _parse_args():
    parser = argparse.ArgumentParser(description="Probabilistic Market Intelligence Engine")
    parser.add_argument("--mode", choices=["live", "backtest"], default="live")
    parser.add_argument("--data-path", default="", help="CSV path for backtest mode")
    parser.add_argument("--asset", default="MOCK", help="Asset symbol (MOCK, BTCUSDT, EURUSD, R_75, ...)")
    parser.add_argument("--timeframe", default="15m", help="Candle timeframe (1m, 5m, 15m, 1h, 4h)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config()

    user_cfg = UserConfig(
        asset=args.asset or os.environ.get("TRADING_ASSET", "MOCK"),
        timeframe=args.timeframe or os.environ.get("TRADING_TIMEFRAME", "15m"),
    )

    if args.mode == "live":
        asyncio.run(main_live(cfg, user_cfg))
    else:
        if not args.data_path:
            print("ERROR: --data-path required for backtest mode", file=sys.stderr)
            sys.exit(1)
        asyncio.run(main_backtest(cfg, user_cfg, args.data_path))
