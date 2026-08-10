"""Recompute the live indicator values from raw candles, bit-for-bit.

Every function here is a deliberate transcription of the arithmetic in
``data/market/mt5_client.get_indicators`` and the score bucketing in
``analysis/technical/indicators`` / ``analysis/technical/regime``. Where that
code is unusual — a simple-average RSI instead of Wilder, an SMA-based MACD
instead of EMA — this module reproduces the unusual version *on purpose*.

The goal is not correct technical analysis. The goal is that a training row and
a live inference call, given the same candles, produce the same ten numbers. A
"better" formula here would silently reintroduce the train/serve skew that made
the previous model unusable.

`tests/test_entry_feature_parity.py` asserts these against the real live
functions rather than trusting this docstring.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from config import (
    MA_TREND_FLAT_ATR_MULT,
    VOLATILITY_RATIO_HIGH,
    VOLATILITY_RATIO_LOW,
    VOLATILITY_RATIO_VERY_HIGH,
)

# Live uses a 100-candle window for indicators (get_candles(symbol, tf, 100)).
LIVE_INDICATOR_WINDOW = 100

# Minimum bars before the live formulas produce a meaningful value: MACD needs
# 26, ma_trend needs 50.
MIN_BARS = 50


def live_rsi(closes: Sequence[float]) -> float:
    """RSI as mt5_client computes it: simple average over the last 14 diffs.

    Note the quirk this reproduces: gains and losses are accumulated into
    separate lists but *both* are divided by 14, not by their own lengths. A
    period with 3 up-moves and 11 down-moves divides the 3 gains by 14. This is
    not Wilder smoothing and not a standard RSI; it is what production computes.
    """
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, 15):
        diff = closes[-i] - closes[-i - 1]
        (gains if diff > 0 else losses).append(abs(diff))

    # The epsilon guard must trigger on a zero *average*, not an empty list.
    #
    # `diff > 0` sends everything else — including an exact 0.0 — to `losses`,
    # so a flat stretch of 14 bars fills `losses` with fourteen zeros. The list
    # is non-empty, the `if losses` guard passes, and avg_loss is 0.0: a
    # ZeroDivisionError. Flat H1 stretches are ordinary in real data (weekends,
    # holidays, thin sessions), and the original guard could never catch them.
    #
    # In live this raised inside get_indicators' try/except and silently
    # returned FALLBACK_INDICATORS — which is where the rsi=50.0 / atr=0.001
    # constants polluting the recorded dataset came from (KNOWN_ISSUES #3).
    #
    # Applying the same 0.001 epsilon the original author intended, to both
    # averages, makes a flat window score RSI 50 (rs = 1), an all-up window ~100
    # and an all-down window ~0. Inputs that did not previously raise are
    # unaffected, so the feature distribution does not shift.
    avg_gain = (sum(gains) / 14) or 0.001
    avg_loss = (sum(losses) / 14) or 0.001
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def live_atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> float:
    """ATR as mt5_client computes it: plain mean of the last 14 true ranges."""
    trs: List[float] = []
    for i in range(1, 15):
        tr = max(
            highs[-i] - lows[-i],
            abs(highs[-i] - closes[-i - 1]),
            abs(lows[-i] - closes[-i - 1]),
        )
        trs.append(tr)
    return sum(trs) / 14 if trs else 0.001


def live_macd(closes: Sequence[float]) -> float:
    """MACD as mt5_client computes it: SMA12 - SMA26 (the variables are named
    ema12/ema26 in the original, but the arithmetic is a simple mean)."""
    return sum(closes[-12:]) / 12 - sum(closes[-26:]) / 26


def live_ma_trend(closes: Sequence[float], atr: float) -> str:
    """The ma_trend string, classified exactly as mt5_client does.

    ``atr`` is required for the flat band: price within
    MA_TREND_FLAT_ATR_MULT ATRs of MA20 counts as sideways. Before that band
    existed, "sideways" needed price == ma20 exactly and was therefore
    unreachable, which froze market_regime at TRENDING.
    """
    ma20 = sum(closes[-20:]) / 20
    ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else closes[-1]
    price = closes[-1]

    if abs(price - ma20) <= atr * MA_TREND_FLAT_ATR_MULT:
        return "sideways"
    if price > ma20 > ma50:
        return "strong uptrend"
    if price > ma20:
        return "uptrend"
    if price < ma20 < ma50:
        return "strong downtrend"
    if price < ma20:
        return "downtrend"
    return "sideways"


def live_atr_ratio(highs, lows, closes, atr_now: float) -> float:
    """Current ATR over this symbol's median ATR across the window.

    Transcribed from mt5_client._atr_ratio. Scale-free, so "volatile" means the
    same thing on EURUSD as on XAUUSD; an absolute ATR% cut would pin each
    symbol to one bucket permanently.
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
    window_atrs = [sum(trs[j - 14:j]) / 14 for j in range(14, len(trs) + 1)]
    window_atrs.sort()
    mid = len(window_atrs) // 2
    median = (window_atrs[mid] if len(window_atrs) % 2
              else (window_atrs[mid - 1] + window_atrs[mid]) / 2)
    return (atr_now / median) if median > 0 else 1.0


def live_volatility_bucket(atr_ratio: float) -> str:
    """The volatility string, bucketed exactly as mt5_client does."""
    if atr_ratio >= VOLATILITY_RATIO_VERY_HIGH:
        return "very high"
    if atr_ratio >= VOLATILITY_RATIO_HIGH:
        return "high"
    if atr_ratio < VOLATILITY_RATIO_LOW:
        return "low"
    return "normal"


def live_indicators(candles: Sequence[Dict[str, float]]) -> Optional[Dict[str, object]]:
    """The dict `get_indicators` returns, from a window of raw candles.

    Returns None when there are too few candles — live falls back to a static
    table in that case, which must never become a training row.
    """
    window = list(candles)[-LIVE_INDICATOR_WINDOW:]
    if len(window) < MIN_BARS:
        return None

    closes = [float(c["close"]) for c in window]
    highs = [float(c["high"]) for c in window]
    lows = [float(c["low"]) for c in window]

    atr = live_atr(highs, lows, closes)
    price = closes[-1]
    atr_ratio = live_atr_ratio(highs, lows, closes, atr)
    return {
        "rsi": round(live_rsi(closes), 2),
        "atr": round(atr, 6),
        "macd": round(live_macd(closes), 6),
        "ma_trend": live_ma_trend(closes, atr),
        "volatility": live_volatility_bucket(atr_ratio),
        "atr_ratio": round(atr_ratio, 4),
        "close": price,
    }


# ---------------------------------------------------------------------------
# Score bucketing — transcribed from analysis/technical/indicators.py
# ---------------------------------------------------------------------------

def trend_score_from_indicators(h4: Dict[str, object]) -> tuple:
    """(score, direction) from H4, mirroring get_trend_score_from_snapshot."""
    if not h4 or (not h4.get("ma_trend") and not h4.get("rsi")):
        return 40, "neutral"

    ma_trend = str(h4.get("ma_trend", "")).lower()
    rsi = float(h4.get("rsi", 50))

    if "strong uptrend" in ma_trend:
        return 85, "bullish"
    if "uptrend" in ma_trend:
        return 70, "bullish"
    if "strong downtrend" in ma_trend:
        return 85, "bearish"
    if "downtrend" in ma_trend:
        return 70, "bearish"

    if rsi > 65:
        return 75, "bullish"
    if rsi > 55:
        return 65, "bullish"
    if rsi < 35:
        return 75, "bearish"
    if rsi < 45:
        return 65, "bearish"
    return 40, "neutral"


def momentum_score_from_indicators(h1: Dict[str, object]) -> tuple:
    """(score, direction) from H1 RSI, mirroring get_momentum_score_from_snapshot."""
    if not h1:
        return 40, "neutral"

    rsi = float(h1.get("rsi", 50))
    if rsi < 30:
        return 85, "bullish"
    if rsi > 70:
        return 85, "bearish"
    if rsi < 45:
        return 65, "bearish"
    if rsi > 55:
        return 65, "bullish"
    return 40, "neutral"


def volatility_score_from_indicators(h1: Dict[str, object]) -> float:
    """Mirrors get_volatility_score_from_snapshot.

    Reads the `volatility` bucket that get_indicators now emits. Before that
    key existed every lookup missed and this was frozen at the neutral 55.
    """
    if not h1:
        return 50.0
    vol = str(h1.get("volatility", "")).lower()
    if "very high" in vol:
        return 20.0
    if "high" in vol:
        return 35.0
    if "low" in vol:
        return 80.0
    return 55.0


def regime_from_scores(trend_direction: str, volatility_score: float) -> str:
    """Mirrors get_market_regime_from_snapshot.

    All four outcomes are now reachable: volatility_score is a real reading, and
    the sideways band makes trend_direction "neutral" possible.
    """
    if volatility_score < 30:
        return "HIGH_VOLATILITY"
    if trend_direction != "neutral":
        return "TRENDING"
    if volatility_score > 70:
        return "LOW_VOLATILITY"
    return "RANGING"


def mtf_strength_from_directions(h4_dir: str, h1_dir: str, m15_dir: str) -> str:
    """Mirrors analyzer_snapshot's alignment logic, returning its string.

    Encoding to a number is the feature spec's job
    (`entry_feature_spec.encode_trend_strength`), so live and training share one
    mapping rather than each inventing its own.
    """
    directions = [h4_dir, h1_dir, m15_dir]
    non_neutral = [d for d in directions if d != "neutral"]

    if len(non_neutral) == 3 and all(d == non_neutral[0] for d in non_neutral):
        return "strong"
    if len(non_neutral) >= 2 and all(d == non_neutral[0] for d in non_neutral):
        return "moderate"
    if h4_dir != "neutral" and h1_dir == h4_dir:
        return "moderate"
    if h1_dir != "neutral" and m15_dir == h1_dir:
        return "weak"
    return "weak"
