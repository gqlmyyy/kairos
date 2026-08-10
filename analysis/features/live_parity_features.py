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
    avg_gain = sum(gains) / 14 if gains else 0.001
    avg_loss = sum(losses) / 14 if losses else 0.001
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


def live_ma_trend(closes: Sequence[float]) -> str:
    """The ma_trend string, classified exactly as mt5_client does."""
    ma20 = sum(closes[-20:]) / 20
    ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else closes[-1]
    price = closes[-1]

    if price > ma20 > ma50:
        return "strong uptrend"
    if price > ma20:
        return "uptrend"
    if price < ma20 < ma50:
        return "strong downtrend"
    if price < ma20:
        return "downtrend"
    return "sideways"


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

    return {
        "rsi": round(live_rsi(closes), 2),
        "atr": round(live_atr(highs, lows, closes), 6),
        "macd": round(live_macd(closes), 6),
        "ma_trend": live_ma_trend(closes),
        "close": closes[-1],
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


def volatility_score_live() -> float:
    """Always 55.0.

    `get_volatility_score_from_snapshot` reads a `volatility` key that
    `get_indicators` never produces, so every live lookup misses and lands on
    the final `else`. Reproduced as a constant rather than "fixed", so training
    matches serving. See entry_feature_spec's module docstring.
    """
    return 55.0


def regime_from_scores(trend_direction: str, volatility_score: float) -> str:
    """Mirrors get_market_regime_from_snapshot.

    With volatility_score frozen at 55, the HIGH_VOLATILITY (<30) and
    LOW_VOLATILITY (>70) branches are unreachable in production; this function
    still implements them so it stays a faithful mirror if that changes.
    """
    if volatility_score < 30:
        return "HIGH_VOLATILITY"
    if trend_direction != "neutral":
        return "TRENDING"
    if volatility_score > 70:
        return "LOW_VOLATILITY"
    return "RANGING"


def trend_strength_live() -> float:
    """Always 0.0 — `mtf.strength` is a string, so main.py's isinstance guard
    always falls through to 0.0. See entry_feature_spec's module docstring."""
    return 0.0
