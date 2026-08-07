"""Compatibility shim. There is no longer anything hybrid about this.

This module used to try QuantDinger first and fall back to MT5. With QuantDinger
removed there is a single source, so the ``*_hybrid`` names simply forward to
``mt5_client``.

Retained because these names are imported by:

    main.py                                  (get_atr)
    data/market/market_snapshot_builder.py   (get_indicators_hybrid)
    analysis/multi_timeframe/analyzer.py     (get_candles, get_indicators_hybrid)
    execution/post_entry/post_entry_manager.py (get_atr_hybrid, get_indicators_hybrid)

One behavioural note: the old ``get_indicators_hybrid`` raised RuntimeError when
both sources failed. That contract is preserved — callers such as the post-entry
manager rely on the exception rather than a silently degraded reading.

New code should import from ``data.market.mt5_client`` directly.
"""

from __future__ import annotations

from utils.logger import get_logger

from .mt5_client import (  # noqa: F401
    FALLBACK_INDICATORS,
    get_candles,
    get_indicators,
    get_tick_price,
)

logger = get_logger("hybrid_market_client")


def get_indicators_hybrid(symbol: str, timeframe: str = "4H") -> dict:
    """Indicators from MT5.

    Raises:
        RuntimeError: when no usable data is available, matching the previous
            behaviour that callers depend on.
    """
    indicators = get_indicators(symbol, timeframe)
    if not indicators:
        raise RuntimeError(f"No market data available for {symbol} {timeframe}")
    return indicators


def get_atr_hybrid(symbol: str, timeframe: str = "4H") -> float:
    return float(get_indicators_hybrid(symbol, timeframe).get("atr", 0.001) or 0.001)


def get_rsi_hybrid(symbol: str, timeframe: str = "4H") -> float:
    return float(get_indicators_hybrid(symbol, timeframe).get("rsi", 50.0) or 50.0)


def get_macd_hybrid(symbol: str, timeframe: str = "4H") -> float:
    return float(get_indicators_hybrid(symbol, timeframe).get("macd", 0.0) or 0.0)


def get_price_hybrid(symbol: str, timeframe: str = "4H") -> float:
    return float(get_indicators_hybrid(symbol, timeframe).get("close", 0.0) or 0.0)


# Aliases kept from the previous module.
get_atr = get_atr_hybrid
get_rsi = get_rsi_hybrid
get_macd = get_macd_hybrid
get_price = get_price_hybrid

__all__ = [
    "get_indicators_hybrid",
    "get_atr_hybrid",
    "get_rsi_hybrid",
    "get_macd_hybrid",
    "get_price_hybrid",
    "get_atr",
    "get_rsi",
    "get_macd",
    "get_price",
    "get_candles",
    "get_tick_price",
    "FALLBACK_INDICATORS",
]
