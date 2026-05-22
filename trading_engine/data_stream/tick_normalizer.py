from __future__ import annotations

import os

from trading_engine.config import Config, UserConfig
from trading_engine.data_stream.base_provider import MarketDataProvider


def get_provider(user_config: UserConfig, config: Config) -> MarketDataProvider:
    """Select and instantiate the correct MarketDataProvider for the given asset."""
    asset = user_config.asset.upper()

    if asset == "MOCK" or asset.startswith("MOCK_"):
        from trading_engine.data_stream.mock_provider import MockProvider
        return MockProvider(asset, config)

    if asset in ("R_10", "R_25", "R_50", "R_75", "R_100", "RDBULL", "RDBEAR",
                 "SYNTH_IDX", "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V"):
        from trading_engine.data_stream.synthetic_provider import SyntheticProvider
        token = os.environ.get("TRADING_DERIV_TOKEN", "")
        return SyntheticProvider(asset, config, api_token=token)

    forex_pairs = {
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
        "NZDUSD", "USDCAD", "EURGBP", "EURJPY", "GBPJPY",
    }
    if asset in forex_pairs:
        from trading_engine.data_stream.forex_provider import ForexProvider
        account_id = os.environ.get("TRADING_OANDA_ACCOUNT", "")
        token = os.environ.get("TRADING_OANDA_TOKEN", "")
        return ForexProvider(asset, config, account_id=account_id, api_token=token)

    # Default: treat as a crypto pair (Binance)
    from trading_engine.data_stream.crypto_provider import CryptoProvider
    return CryptoProvider(asset, config)
