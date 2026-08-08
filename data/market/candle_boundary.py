"""Candle Boundary Utility - Get actual last completed candle time from broker.

Replaces wall-clock based candle detection (`int(time.time() / 3600) * 3600`)
with the actual candle open time from MT5's copy_rates_from_pos.

This correctly handles:
  - Broker server time vs local time offsets
  - Weekend gaps and market closures
  - Any timeframe (reads from config)
"""

from __future__ import annotations

from typing import Optional

from utils.logger import get_logger

logger = get_logger("candle_boundary")

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    mt5 = None

# MT5 timeframe constants
_MT5_TF_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "H6": "TIMEFRAME_H6",
    "H12": "TIMEFRAME_H12",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
    "MN1": "TIMEFRAME_MN1",
    # Aliases
    "1H": "TIMEFRAME_H1",
    "4H": "TIMEFRAME_H4",
    "15M": "TIMEFRAME_M15",
    "1D": "TIMEFRAME_D1",
    "1W": "TIMEFRAME_W1",
}


def get_last_completed_candle_time(symbol: str, timeframe: str = "H1") -> Optional[int]:
    """Return the open time (epoch seconds) of the last COMPLETED candle.

    Uses MT5 copy_rates_from_pos to fetch the most recent candle.
    The last candle in the returned array is the currently-forming candle,
    so we return the second-to-last candle's open time (the last completed one).

    Returns None if MT5 is unavailable or data cannot be fetched.
    """
    if mt5 is None:
        logger.warning("[CANDLE_BOUNDARY] MT5 library not available")
        return None

    tf_str = str(timeframe or "").strip().upper()
    tf_attr = _MT5_TF_MAP.get(tf_str)
    if tf_attr is None:
        logger.warning(f"[CANDLE_BOUNDARY] Unknown timeframe: {tf_str}")
        return None

    tf_const = getattr(mt5, tf_attr, None)
    if tf_const is None:
        logger.warning(f"[CANDLE_BOUNDARY] MT5 timeframe constant not found: {tf_attr}")
        return None

    try:
        # Session ownership belongs to mt5_session — do not initialize here.
        from data.market.mt5_session import ensure_symbol, mt5_call

        if not ensure_symbol(symbol):
            logger.warning("[CANDLE_BOUNDARY] symbol unavailable: %s", symbol)
            return None

        # Fetch 2 candles: the last completed + the currently forming
        with mt5_call():
            bars = mt5.copy_rates_from_pos(symbol, tf_const, 0, 2)
        if bars is None or len(bars) < 2:
            logger.warning(f"[CANDLE_BOUNDARY] copy_rates_from_pos returned insufficient data for {symbol} {tf_str}")
            return None

        # bars[0] is the last completed candle (bars[1] is the forming one)
        last_completed_ts = int(bars[0]["time"])
        return last_completed_ts
    except Exception as e:
        logger.warning(f"[CANDLE_BOUNDARY] Error fetching candle time for {symbol} {tf_str}: {e}")
        return None


def has_new_candle_closed(symbol: str, timeframe: str, last_known_ts: Optional[int]) -> bool:
    """Check if a new candle has closed since the last known candle time.

    Args:
        symbol: trading symbol
        timeframe: timeframe string (e.g. "H1", "H4")
        last_known_ts: the last candle open time we've seen (epoch seconds)

    Returns:
        True if a new candle has closed (i.e. the last completed candle time
        is different from last_known_ts).
    """
    current_ts = get_last_completed_candle_time(symbol, timeframe)
    if current_ts is None:
        # Fallback: if we can't get broker time, use wall-clock as a safe default
        # (this is a degradation, not the primary path)
        return True

    if last_known_ts is None:
        # First time - treat as new candle
        return True

    return current_ts != last_known_ts