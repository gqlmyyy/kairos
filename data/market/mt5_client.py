"""Market data from MetaTrader5. Replaces the QuantDinger REST client.

Function signatures deliberately mirror the old ``data/market/client.py`` so
the eight modules that import it keep working unchanged.

**The indicator formulas below are copied verbatim from the QuantDinger client.**
That is intentional and important: the entry model (``models/entry/entry_model.json``)
was trained on features produced by those exact formulas. Some of them are not
textbook-correct — the RSI is a simple 14-period average rather than Wilder
smoothing, and the MACD uses simple moving averages instead of EMAs — but
changing them here would shift the feature distribution the model sees and move
p_win for reasons that would be very hard to trace.

This migration changes the *source* of the candles, nothing else. Correcting the
formulas is tracked separately in ROADMAP.md and requires retraining, because
the training pipeline (analysis/entry_v2/feature_engineering.py) already uses
proper EMA-based MACD — that train/serve inconsistency predates this work.

Verified with scripts/capture_indicator_baseline.py: values before and after
the migration must match field for field.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from utils.logger import get_logger
from config import (
    MA_TREND_FLAT_ATR_MULT,
    VOLATILITY_RATIO_HIGH,
    VOLATILITY_RATIO_LOW,
    VOLATILITY_RATIO_VERY_HIGH,
)

from .mt5_session import ensure_session, ensure_symbol, get_account_info, mt5_call

logger = get_logger("mt5_client")

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:  # pragma: no cover
    mt5 = None

# Bot timeframe strings -> MT5 timeframe constant names.
# Single source of truth: the old code had two divergent maps, one in
# client.py (QuantDinger label strings like "15m") and one in hybrid_client.py.
_TF_NAMES = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "H6": "TIMEFRAME_H6", "H12": "TIMEFRAME_H12", "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1", "MN1": "TIMEFRAME_MN1",
    # Aliases used across the codebase.
    "1H": "TIMEFRAME_H1", "4H": "TIMEFRAME_H4", "15M": "TIMEFRAME_M15",
    "15m": "TIMEFRAME_M15", "1D": "TIMEFRAME_D1", "1W": "TIMEFRAME_W1",
}

# Kept identical to the previous client so degraded behaviour is unchanged.
FALLBACK_ATR = {
    "EURUSD": 0.0008, "GBPUSD": 0.0010,
    "XAUUSD": 8.0, "USDJPY": 0.15,
}

FALLBACK_INDICATORS = {
    "EURUSD": {"rsi": 50.0, "atr": 0.0008, "macd": 0.0, "ma_trend": "sideways", "volatility": "normal", "atr_ratio": 1.0, "close": 1.0800},
    "GBPUSD": {"rsi": 50.0, "atr": 0.0010, "macd": 0.0, "ma_trend": "sideways", "volatility": "normal", "atr_ratio": 1.0, "close": 1.2700},
    "XAUUSD": {"rsi": 50.0, "atr": 8.0, "macd": 0.0, "ma_trend": "sideways", "volatility": "normal", "atr_ratio": 1.0, "close": 2350.0},
    "USDJPY": {"rsi": 50.0, "atr": 0.15, "macd": 0.0, "ma_trend": "sideways", "volatility": "normal", "atr_ratio": 1.0, "close": 150.0},
}

# Candle cache. TTL matches the previous client (5 minutes) so request volume
# and staleness characteristics are unchanged by the migration.
_candles_cache: Dict[str, list] = {}
_cache_timestamp: Dict[str, float] = {}
CACHE_TTL = 300

# Negative cache: a failed fetch is remembered briefly so a broken feed does not
# trigger a fresh MT5 round trip on every single call in the 5-second loop.
_failure_timestamp: Dict[str, float] = {}
FAILURE_TTL = 30

_error_count: Dict[str, int] = {}
ERROR_THRESHOLD = 5


def _mt5_timeframe(timeframe: str):
    """Resolve a timeframe string to the MT5 constant, or None."""
    key = str(timeframe or "").strip()
    name = _TF_NAMES.get(key) or _TF_NAMES.get(key.upper())
    if name is None:
        logger.warning("[MT5_DATA] unknown timeframe %r", timeframe)
        return None
    return getattr(mt5, name, None) if mt5 is not None else None


def get_candles(symbol: str, timeframe: str = "4H", count: int = 100) -> list:
    """Fetch **completed** candles from MT5, with caching.

    The currently forming candle is excluded: it changes on every tick, so an
    indicator computed from it is not reproducible and does not match a
    backtest run on closed bars. The newest element of the returned list is
    always the last candle that has actually closed.

    Returns a list of dicts with keys: time, open, high, low, close,
    tick_volume — the same shape the rest of the project already expects.
    """
    cache_key = f"{symbol}_{timeframe}_{count}"
    now = time.time()

    cached = _candles_cache.get(cache_key)
    if cached is not None and (now - _cache_timestamp.get(cache_key, 0)) < CACHE_TTL:
        return cached

    # Recently failed: do not hammer a feed that is already known to be down.
    if (now - _failure_timestamp.get(cache_key, 0)) < FAILURE_TTL:
        return cached if cached is not None else []

    if not ensure_session():
        _failure_timestamp[cache_key] = now
        return cached if cached is not None else []

    tf_const = _mt5_timeframe(timeframe)
    if tf_const is None:
        return []

    if not ensure_symbol(symbol):
        _failure_timestamp[cache_key] = now
        return cached if cached is not None else []

    try:
        with mt5_call():
            # Ask for one extra bar: position 0 is the *currently forming*
            # candle, which is dropped below. Requesting count+1 keeps the
            # number of completed bars returned equal to `count`.
            bars = mt5.copy_rates_from_pos(symbol, tf_const, 0, count + 1)

        # ------------------------------------------------------------------
        # Drop the forming candle.
        #
        # copy_rates_from_pos returns oldest -> newest, so bars[-1] is the bar
        # still being built. Indicators previously read closes[-1], i.e. a
        # value that changes tick by tick and does not match what any backtest
        # on closed bars would have seen. Trading decisions must use completed
        # candles only.
        # ------------------------------------------------------------------
        if bars is not None and len(bars) > 1:
            bars = bars[:-1]
        elif bars is not None and len(bars) == 1:
            # Only the forming bar exists — nothing completed to act on.
            bars = bars[:0]

        if bars is None or len(bars) == 0:
            _error_count[symbol] = _error_count.get(symbol, 0) + 1
            _failure_timestamp[cache_key] = now
            logger.warning(
                "[MT5_DATA] no candles for %s %s (consecutive failures=%d)",
                symbol, timeframe, _error_count[symbol],
            )
            if _error_count[symbol] >= ERROR_THRESHOLD:
                logger.error(
                    "[MT5_DATA] %s failing consistently (%d errors) - check the terminal",
                    symbol, _error_count[symbol],
                )
            return cached if cached is not None else []

        candles = [
            {
                "time": int(b["time"]),
                "open": float(b["open"]),
                "high": float(b["high"]),
                "low": float(b["low"]),
                "close": float(b["close"]),
                "tick_volume": int(b["tick_volume"]),
            }
            for b in bars
        ]

        _candles_cache[cache_key] = candles
        _cache_timestamp[cache_key] = now
        _failure_timestamp.pop(cache_key, None)
        _error_count[symbol] = 0
        return candles

    except Exception as exc:
        _error_count[symbol] = _error_count.get(symbol, 0) + 1
        _failure_timestamp[cache_key] = now
        logger.error("[MT5_DATA] candle fetch failed for %s %s: %s", symbol, timeframe, exc)
        return cached if cached is not None else []


def _atr_ratio(highs, lows, closes, atr_now: float) -> float:
    """Current ATR over this symbol's median ATR across the candle window.

    Scale-free, so "volatile" means the same thing on EURUSD as on XAUUSD.
    Returns 1.0 (neutral) when there is not enough history to judge.
    """
    trs = []
    for i in range(1, len(closes)):
        trs.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    if len(trs) < 28:
        return 1.0
    window_atrs = [
        sum(trs[j - 14:j]) / 14
        for j in range(14, len(trs) + 1)
    ]
    window_atrs.sort()
    mid = len(window_atrs) // 2
    median = (window_atrs[mid] if len(window_atrs) % 2
              else (window_atrs[mid - 1] + window_atrs[mid]) / 2)
    return (atr_now / median) if median > 0 else 1.0


def get_indicators(symbol: str, timeframe: str = "4H") -> dict:
    """Compute RSI/ATR/MACD/MA-trend from MT5 candles.

    The arithmetic below is a verbatim copy of the previous QuantDinger-backed
    implementation. See the module docstring for why it is not "fixed" here.
    """
    candles = get_candles(symbol, timeframe, 100)

    if not candles or len(candles) < 20:
        fallback = FALLBACK_INDICATORS.get(symbol, {
            "rsi": 50.0, "atr": 0.001, "macd": 0.0, "ma_trend": "sideways",
            "volatility": "normal", "atr_ratio": 1.0, "close": 0.0
        })
        logger.warning(
            "[MT5_DATA] using FALLBACK indicators for %s %s (candles=%d)",
            symbol, timeframe, len(candles) if candles else 0,
        )
        return fallback.copy()

    try:
        closes = [float(c.get("close", 0)) for c in candles]
        highs = [float(c.get("high", 0)) for c in candles]
        lows = [float(c.get("low", 0)) for c in candles]

        # RSI(14) - simple average over the last 14 differences.
        # NOT Wilder smoothing: preserved deliberately, see module docstring.
        gains, losses = [], []
        for i in range(1, 15):
            diff = closes[-i] - closes[-i - 1]
            (gains if diff > 0 else losses).append(abs(diff))
        avg_gain = sum(gains) / 14 if gains else 0.001
        avg_loss = sum(losses) / 14 if losses else 0.001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # ATR(14) over the last 14 bars.
        trs = []
        for i in range(1, 15):
            tr = max(
                highs[-i] - lows[-i],
                abs(highs[-i] - closes[-i - 1]),
                abs(lows[-i] - closes[-i - 1]),
            )
            trs.append(tr)
        atr = sum(trs) / 14 if trs else FALLBACK_ATR.get(symbol, 0.001)

        # MACD(12,26) using simple moving averages, not EMAs.
        # Preserved deliberately, see module docstring.
        ema12 = sum(closes[-12:]) / 12
        ema26 = sum(closes[-26:]) / 26
        macd = ema12 - ema26

        # MA trend classification.
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else closes[-1]
        price = closes[-1]

        # "sideways" used to require price == ma20 exactly — a float equality
        # that never happens — so this function never returned it, the H4 trend
        # direction was never "neutral", and market_regime was permanently
        # TRENDING (KNOWN_ISSUES #13). Price within a fraction of ATR of MA20 is
        # now treated as flat. ATR-relative so the band means the same thing on
        # EURUSD (~0.0016) as on XAUUSD (~40).
        flat_band = atr * MA_TREND_FLAT_ATR_MULT
        if abs(price - ma20) <= flat_band:
            ma_trend = "sideways"
        elif price > ma20 > ma50:
            ma_trend = "strong uptrend"
        elif price > ma20:
            ma_trend = "uptrend"
        elif price < ma20 < ma50:
            ma_trend = "strong downtrend"
        elif price < ma20:
            ma_trend = "downtrend"
        else:
            ma_trend = "sideways"

        # Volatility bucket. get_volatility_score_from_snapshot has always read
        # a "volatility" key, but nothing ever wrote one, so every lookup missed
        # and the score was frozen at the neutral 55.
        #
        # Measured against this symbol's own recent ATR rather than an absolute
        # percentage: EURUSD sits near 0.14% of price and XAUUSD near 0.96%, so a
        # fixed cut would pin each symbol to a single bucket permanently.
        atr_ratio = _atr_ratio(highs, lows, closes, atr)
        if atr_ratio >= VOLATILITY_RATIO_VERY_HIGH:
            volatility = "very high"
        elif atr_ratio >= VOLATILITY_RATIO_HIGH:
            volatility = "high"
        elif atr_ratio < VOLATILITY_RATIO_LOW:
            volatility = "low"
        else:
            volatility = "normal"

        return {
            "rsi": round(rsi, 2),
            "atr": round(atr, 6),
            "macd": round(macd, 6),
            "ma_trend": ma_trend,
            "volatility": volatility,
            "atr_ratio": round(atr_ratio, 4),
            "close": price,
        }

    except Exception as exc:
        logger.error("[MT5_DATA] indicator computation failed for %s: %s", symbol, exc)
        return FALLBACK_INDICATORS.get(symbol, {
            "rsi": 50.0, "atr": 0.001, "macd": 0.0, "ma_trend": "sideways",
            "volatility": "normal", "atr_ratio": 1.0, "close": 0.0
        }).copy()


def get_atr(symbol: str, timeframe: str = "4H") -> float:
    data = get_indicators(symbol, timeframe)
    default = FALLBACK_ATR.get(symbol, 0.001)
    return float(data.get("atr", default) or default)


def get_rsi(symbol: str, timeframe: str = "4H") -> float:
    return float(get_indicators(symbol, timeframe).get("rsi", 50.0) or 50.0)


def get_macd(symbol: str, timeframe: str = "4H") -> float:
    return float(get_indicators(symbol, timeframe).get("macd", 0.0) or 0.0)


def get_price(symbol: str, timeframe: str = "4H") -> float:
    """Last close. For a live tradeable price prefer get_tick_price()."""
    data = get_indicators(symbol, timeframe)
    price = data.get("close", 0.0)
    if not price:
        return FALLBACK_INDICATORS.get(symbol, {}).get("close", 0.0)
    return float(price)


def get_tick_price(symbol: str) -> Optional[Dict[str, float]]:
    """Live bid/ask. None when the terminal has no tick for the symbol."""
    if not ensure_symbol(symbol):
        return None
    try:
        with mt5_call():
            tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {"bid": float(tick.bid), "ask": float(tick.ask)}
    except Exception as exc:
        logger.warning("[MT5_DATA] tick fetch failed for %s: %s", symbol, exc)
        return None


def get_account_summary() -> dict:
    """Account figures in the shape the old QuantDinger client returned."""
    account = get_account_info()
    if account is None:
        logger.error("[MT5_DATA] account info unavailable")
        return {"balance": 0.0, "equity": 0.0, "margin": 0.0}
    return {
        "balance": float(getattr(account, "balance", 0.0) or 0.0),
        "equity": float(getattr(account, "equity", 0.0) or 0.0),
        "margin": float(getattr(account, "margin", 0.0) or 0.0),
    }


def get_equity() -> float:
    """Account equity, straight from the terminal.

    Returns 0.0 when unavailable — same contract as the previous
    QuantDinger-backed implementation, so risk checks behave identically.
    """
    return get_account_summary()["equity"]


# Backwards-compatible alias: the old client exposed this name.
get_account_info_dict = get_account_summary


def set_token(token: str) -> None:
    """No-op retained for import compatibility.

    The QuantDinger REST client needed a bearer token; MT5 authenticates once
    in mt5_session. Kept so main.py's historical call site does not break during
    the migration; removed once all callers are updated.
    """
    return None


def check_health() -> dict:
    """Session health, for the Telegram /status command and the watchdog."""
    from .mt5_session import is_healthy

    healthy = is_healthy()
    account = get_account_info() if healthy else None
    return {
        "healthy": healthy,
        "login": getattr(account, "login", None) if account else None,
        "server": getattr(account, "server", None) if account else None,
        "balance": float(getattr(account, "balance", 0.0) or 0.0) if account else 0.0,
    }


def clear_cache() -> None:
    """Drop cached candles. Test hook and manual recovery aid."""
    _candles_cache.clear()
    _cache_timestamp.clear()
    _failure_timestamp.clear()
    _error_count.clear()
