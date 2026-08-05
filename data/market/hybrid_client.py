#!/usr/bin/env python3
"""
Hybrid Market Data Client
- QuantDinger as PRIMARY source
- MT5 as FALLBACK source (when QuantDinger fails or returns empty data)
- If both fail, the system fails
"""

from utils.logger import get_logger
from data.market.client import (
    get_candles as qd_get_candles,
    get_indicators as qd_get_indicators,
)
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

logger = get_logger("hybrid_market_client")

# ==============================
# MT5 Fallback Support
# ==============================
try:
    import MetaTrader5 as mt5  # type: ignore
    MT5_AVAILABLE = True
except Exception:
    mt5 = None
    MT5_AVAILABLE = False

# MT5 timeframe mapping (matches config.py TF_TREND/TF_DECISION/TF_TIMING)
_MT5_TF_MAP = {
    "M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30",
    "H1": "H1", "H4": "H4", "H6": "H6", "H12": "H12",
    "D1": "D1", "W1": "W1", "MN1": "MN1",
    # Aliases used by the bot
    "1H": "H1", "4H": "H4", "15M": "M15", "1D": "D1", "1W": "W1",
}

# Number of candles to fetch from MT5 (need at least 50 for MA50)
_MT5_CANDLE_COUNT = 100


def _mt5_timeframe(timeframe: str) -> str:
    """Convert bot timeframe string to MT5 timeframe string."""
    tf = str(timeframe or "").strip().upper()
    return _MT5_TF_MAP.get(tf, "H4")


def _ensure_mt5_initialized() -> bool:
    """Initialize MT5 connection (best-effort, non-fatal)."""
    if not MT5_AVAILABLE or mt5 is None:
        logger.warning("[MT5_FALLBACK] MetaTrader5 library not available")
        return False

    try:
        if not mt5.terminal_info():
            if not mt5.initialize():
                logger.warning(f"[MT5_FALLBACK] mt5.initialize() failed: {mt5.last_error()}")
                return False
    except Exception as e:
        logger.warning(f"[MT5_FALLBACK] mt5 initialize check failed: {e}")
        return False

    # Login (best-effort; some environments already logged in)
    try:
        if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            logger.warning(f"[MT5_FALLBACK] mt5.login() failed: {mt5.last_error()}")
            return False
    except Exception as e:
        logger.warning(f"[MT5_FALLBACK] mt5.login() exception: {e}")
        return False

    return True


def _ensure_symbol_selected(symbol: str) -> bool:
    """Select symbol in MT5 (best-effort)."""
    if not MT5_AVAILABLE or mt5 is None:
        return False
    try:
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"[MT5_FALLBACK] mt5.symbol_select({symbol}) failed")
            return False
    except Exception as e:
        logger.warning(f"[MT5_FALLBACK] mt5.symbol_select exception for {symbol}: {e}")
        return False
    return True


def _get_mt5_candles(symbol: str, timeframe: str = "H4", count: int = _MT5_CANDLE_COUNT) -> list:
    """Fetch candles from MT5 using copy_rates_from_pos.

    Returns a list of dicts with keys: time, open, high, low, close, tick_volume.
    Returns empty list on failure.
    """
    if not MT5_AVAILABLE or mt5 is None:
        return []

    if not _ensure_mt5_initialized():
        return []

    if not _ensure_symbol_selected(symbol):
        return []

    tf_str = _mt5_timeframe(timeframe)
    tf_attr = f"TIMEFRAME_{tf_str}"
    tf_const = getattr(mt5, tf_attr, None)
    if tf_const is None:
        logger.warning(f"[MT5_FALLBACK] Unknown MT5 timeframe: {tf_str}")
        return []

    try:
        bars = mt5.copy_rates_from_pos(symbol, tf_const, 0, count)
        if bars is None or len(bars) == 0:
            logger.warning(f"[MT5_FALLBACK] copy_rates_from_pos returned empty for {symbol} {tf_str}")
            return []

        # Convert numpy structured array to list of dicts
        candles = []
        for bar in bars:
            candles.append({
                "time": int(bar["time"]),
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "tick_volume": int(bar["tick_volume"]),
            })
        logger.debug(f"[MT5_FALLBACK] Got {len(candles)} candles for {symbol} {tf_str}")
        return candles
    except Exception as e:
        logger.warning(f"[MT5_FALLBACK] copy_rates_from_pos exception for {symbol} {tf_str}: {e}")
        return []


def _compute_indicators_from_candles(symbol: str, candles: list) -> dict:
    """Compute RSI, ATR, MACD, MA trend from candle dicts (same logic as client.py)."""
    if not candles or len(candles) < 20:
        return {}

    try:
        closes = [float(c.get("close", 0)) for c in candles]
        highs = [float(c.get("high", 0)) for c in candles]
        lows = [float(c.get("low", 0)) for c in candles]

        # RSI(14) - Wilder smoothing
        gains, losses = [], []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            if diff > 0:
                gains.append(diff)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(diff))

        if len(gains) < 14:
            return {}

        avg_gain = sum(gains[:14]) / 14
        avg_loss = sum(losses[:14]) / 14
        for i in range(14, len(gains)):
            avg_gain = (avg_gain * 13 + gains[i]) / 14
            avg_loss = (avg_loss * 13 + losses[i]) / 14

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # ATR(14) - Wilder smoothing
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)

        if len(trs) < 14:
            return {}

        atr = sum(trs[:14]) / 14
        for i in range(14, len(trs)):
            atr = (atr * 13 + trs[i]) / 14

        # MACD(12,26) - simple EMA approximations via averages (same as client.py)
        ema12 = sum(closes[-12:]) / 12
        ema26 = sum(closes[-26:]) / 26
        macd = ema12 - ema26

        # MA Trend
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else closes[-1]
        price = closes[-1]

        if price > ma20 > ma50:
            ma_trend = "strong uptrend"
        elif price > ma20:
            ma_trend = "uptrend"
        elif price < ma20 < ma50:
            ma_trend = "strong downtrend"
        elif price < ma20:
            ma_trend = "downtrend"
        else:
            ma_trend = "sideways"

        return {
            "rsi": round(rsi, 2),
            "atr": round(atr, 6),
            "macd": round(macd, 6),
            "ma_trend": ma_trend,
            "close": price,
        }
    except Exception as e:
        logger.warning(f"[MT5_FALLBACK] indicator computation failed for {symbol}: {e}")
        return {}


def _get_indicators_from_mt5(symbol: str, timeframe: str = "H4") -> dict:
    """Fetch indicators from MT5 as fallback source."""
    candles = _get_mt5_candles(symbol, timeframe)
    if not candles:
        logger.warning(f"[MT5_FALLBACK] No candles from MT5 for {symbol} {timeframe}")
        return {}

    indicators = _compute_indicators_from_candles(symbol, candles)
    if not indicators:
        logger.warning(f"[MT5_FALLBACK] Could not compute indicators from MT5 candles for {symbol} {timeframe}")
        return {}

    logger.info(f"[MT5_FALLBACK] ✓ Got live data from MT5 for {symbol} {timeframe}")
    return indicators


def get_indicators_hybrid(symbol: str, timeframe: str = "4H") -> dict:
    """
    Get indicators from QuantDinger first, then fall back to MT5.

    Strategy:
    1. Try QuantDinger first.
    2. If QuantDinger fails or returns empty/placeholder data, try MT5.
    3. If both fail, raise exception.
    """

    # 1) Try QuantDinger first
    try:
        logger.debug(f"Fetching indicators for {symbol} (QuantDinger first)")
        indicators = qd_get_indicators(symbol, timeframe)

        if indicators and indicators.get("rsi") != 50.0:
            logger.debug(f"✓ Got live data from QuantDinger for {symbol}")
            return indicators

        if indicators and indicators.get("rsi") == 50.0:
            logger.warning(f"QuantDinger returned placeholder data (rsi=50.0) for {symbol} - trying MT5 fallback")
        else:
            logger.warning(f"QuantDinger returned empty data for {symbol} - trying MT5 fallback")
    except Exception as e:
        logger.warning(f"QuantDinger failed for {symbol}: {e} - trying MT5 fallback")

    # 2) Try MT5 fallback
    mt5_indicators = _get_indicators_from_mt5(symbol, timeframe)
    if mt5_indicators:
        return mt5_indicators

    # 3) Both failed
    raise RuntimeError(
        f"Both QuantDinger and MT5 failed for {symbol} {timeframe} - no data available"
    )


def get_atr_hybrid(symbol: str, timeframe: str = "4H") -> float:
    """Get ATR from QuantDinger with MT5 fallback"""
    indicators = get_indicators_hybrid(symbol, timeframe)
    return float(indicators.get("atr", 0.001) or 0.001)


def get_rsi_hybrid(symbol: str, timeframe: str = "4H") -> float:
    """Get RSI from QuantDinger with MT5 fallback"""
    indicators = get_indicators_hybrid(symbol, timeframe)
    return float(indicators.get("rsi", 50.0) or 50.0)


def get_macd_hybrid(symbol: str, timeframe: str = "4H") -> float:
    """Get MACD from QuantDinger with MT5 fallback"""
    indicators = get_indicators_hybrid(symbol, timeframe)
    return float(indicators.get("macd", 0.0) or 0.0)


def get_price_hybrid(symbol: str, timeframe: str = "4H") -> float:
    """Get price from QuantDinger with MT5 fallback"""
    indicators = get_indicators_hybrid(symbol, timeframe)
    return float(indicators.get("close", 0.0) or 0.0)


def get_candles(symbol: str, timeframe: str = "4H", count: int = 100) -> list:
    """Get candles from QuantDinger with MT5 fallback"""
    # Try QuantDinger first
    try:
        candles = qd_get_candles(symbol, timeframe, count)
        if candles:
            return candles
        logger.warning(f"QuantDinger returned empty candles for {symbol} - trying MT5 fallback")
    except Exception as e:
        logger.warning(f"QuantDinger candles failed for {symbol}: {e} - trying MT5 fallback")

    # MT5 fallback
    mt5_candles = _get_mt5_candles(symbol, timeframe, count)
    if mt5_candles:
        return mt5_candles

    return []


# Alias for compatibility
get_atr = get_atr_hybrid
get_rsi = get_rsi_hybrid
get_macd = get_macd_hybrid
get_price = get_price_hybrid


if __name__ == "__main__":
    # Test the hybrid client with MT5 fallback
    from config import SYMBOLS

    print("\n" + "=" * 60)
    print("Hybrid Market Data Client Test (QuantDinger + MT5 Fallback)")
    print("=" * 60 + "\n")

    for symbol in SYMBOLS:
        try:
            print(f"{symbol}:")
            indicators = get_indicators_hybrid(symbol)
            print(f"  RSI:  {indicators.get('rsi')}")
            print(f"  ATR:  {indicators.get('atr')}")
            print(f"  MACD: {indicators.get('macd')}")
            print(f"  Price: {indicators.get('close')}")
        except RuntimeError as e:
            print(f"  ERROR: {e}")
        print()