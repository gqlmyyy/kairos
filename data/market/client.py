"""Compatibility shim. Market data now comes from MT5, not QuantDinger.

This module used to be the QuantDinger REST client. QuantDinger has been
removed: it was a pass-through that fetched raw OHLC candles and returned no
indicators of its own (every RSI/ATR/MACD figure was computed locally in this
project anyway), while adding a second MT5 session that contended with the
bot's own over the same IPC channel.

The names below are re-exported from ``mt5_client`` so the modules that import
this one keep working unchanged:

    data/market/atr.py
    analysis/entry_v2/candle_loader.py
    analysis/features/historical_dataset_builder.py
    execution/risk_management/market_regime_detector.py
    scripts/backtest_exit_dataset_builder.py
    scripts/build_real_exit_dataset.py

New code should import from ``data.market.mt5_client`` directly.
"""

from __future__ import annotations

from .mt5_client import (  # noqa: F401
    CACHE_TTL,
    FALLBACK_ATR,
    FALLBACK_INDICATORS,
    clear_cache,
    get_account_summary as get_account_info,
    get_atr,
    get_candles,
    get_equity,
    get_indicators,
    get_macd,
    get_price,
    get_rsi,
    set_token,
)

__all__ = [
    "get_candles",
    "get_indicators",
    "get_atr",
    "get_rsi",
    "get_macd",
    "get_price",
    "get_equity",
    "get_account_info",
    "set_token",
    "clear_cache",
    "FALLBACK_ATR",
    "FALLBACK_INDICATORS",
    "CACHE_TTL",
]
