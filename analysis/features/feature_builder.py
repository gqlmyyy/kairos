# Trading Bot V3 - Feature Snapshot Builder (ML-ready)
#
# Source of truth for all features fed into execution_dataset expected_* columns.
# No other module should compute features directly.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


# -----------------------------
# Normalization helpers
# -----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_0_1(x: float, lo: float, hi: float) -> float | None:
    """
    Normalize x from [lo, hi] to [0,1]. Returns None if x is None/NaN.
    """
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if xf != xf:  # NaN
        return None
    if hi == lo:
        return None
    return _clamp((xf - lo) / (hi - lo), 0.0, 1.0)


def _norm_pm1(x: float, lo: float, hi: float) -> float | None:
    """
    Normalize x from [lo, hi] to [-1, 1] assuming lo<hi.
    """
    v01 = _norm_0_1(x, lo, hi)
    if v01 is None:
        return None
    return _clamp(v01 * 2.0 - 1.0, -1.0, 1.0)


def _normalize_direction_to_signed(direction: str) -> float | None:
    """
    Direction -> signed [-1,1] (used only internally)
    """
    if direction is None:
        return None
    d = str(direction).strip().lower()
    if d in ("buy", "long", "1", "bullish", "bull"):
        return 1.0
    if d in ("sell", "short", "0", "bearish", "bear"):
        return -1.0
    return None


def _session_from_utc(dt: datetime) -> str:
    """
    London/NY/Asia based on UTC hour.
    Approx mapping:
      - Asia: 00-07 UTC
      - London: 07-13 UTC
      - NY: 13-20 UTC
      - Derived/Other: 20-24 UTC (treated as Asia-like)
    """
    h = dt.hour
    if 0 <= h < 7:
        return "Asia"
    if 7 <= h < 13:
        return "London"
    if 13 <= h < 20:
        return "NY"
    return "Asia"


# -----------------------------
# Main API
# -----------------------------

def build_trade_features(
    symbol: str,
    market_data: dict | None,
    indicators: dict | None,
    ai_analysis: dict | None,
    sentiment: dict | None,
    regime: dict | None,
    mtf_data: dict | None,
) -> dict:
    """
    Build unified ML-ready feature vector for a trade snapshot.

    Returns dict with STRUCTURED OUTPUT (expected_* keys).
    Any feature not available is kept as None (not excluded).

    Normalization:
      - RSI -> [0,1] from [0,100]
      - ATR, spread -> [0,1] using simple clamped min/max guards if present
      - MACD -> [-1,1] if available (expects MACD line or normalized macd)
      - trend_strength -> [0,1] (expects 0-100 or qualitative mapped)
      - momentum_score, volatility_score -> [0,1] from [0,100]
      - market_regime categorical -> [-1,1] signed based on label
      - sentiment_score/news_impact_score/ai_score -> [0,1] from [0,100] (or [0,1] if already)
      - session categorical -> [0,1] via deterministic mapping
      - direction -> signed [-1,1] for model training
    """

    # ---- Extract and normalize technical ----
    indicators = indicators or {}
    market_data = market_data or {}
    ai_analysis = ai_analysis or {}
    sentiment = sentiment or {}
    regime = regime or {}
    mtf_data = mtf_data or {}

    # RSI
    rsi_raw = indicators.get("rsi", None)
    rsi = _norm_0_1(rsi_raw, 0.0, 100.0)
    if rsi is None:
        # Fallback: use trend_strength as a proxy to avoid None in expected_*.
        trend_strength_fallback = indicators.get("trend_strength", None) or mtf_data.get("strength", None)
        if isinstance(trend_strength_fallback, str):
            ts = trend_strength_fallback.strip().lower()
            if "strong" in ts:
                rsi = 1.0
            elif "moderate" in ts:
                rsi = 0.66
            elif "weak" in ts:
                rsi = 0.33
            else:
                rsi = 0.0
        else:
            # If still unknown, clamp a minimal safe value.
            rsi = 0.0


    # ATR
    atr_raw = indicators.get("atr", None) or market_data.get("atr", None)
    # ATR is instrument dependent; we only provide a stable normalization guard.
    atr = _norm_0_1(atr_raw, 0.0, 50.0)
    if atr is None:
        # Hard fallback to 0.0 to ensure expected_atr never becomes None.
        atr = 0.0


    # MACD
    macd_raw = indicators.get("macd", None)
    # QuantDinger provides macd as numeric. Force to numeric and avoid None.
    macd = _norm_pm1(macd_raw, -1.0, 1.0) if macd_raw is not None else 0.0
    if macd is None:
        macd = 0.0



    # trend_strength (expects 0-100 or qualitative)
    trend_strength_raw = indicators.get("trend_strength", None) or regime.get("trend_strength", None)
    if isinstance(trend_strength_raw, str):
        ts = trend_strength_raw.strip().lower()
        # Map qualitative -> 0..1
        if "strong" in ts:
            trend_strength = 1.0
        elif "moderate" in ts:
            trend_strength = 0.66
        elif "weak" in ts:
            trend_strength = 0.33
        elif "neutral" in ts:
            trend_strength = 0.0
        else:
            trend_strength = None
    else:
        trend_strength = _norm_0_1(trend_strength_raw, 0.0, 100.0)

    # momentum_score, volatility_score (expected 0-100)
    momentum_score_raw = indicators.get("momentum_score", None) or indicators.get("momentum", None) or mtf_data.get("momentum_score", None)
    momentum_score = _norm_0_1(momentum_score_raw, 0.0, 100.0)

    volatility_score_raw = indicators.get("volatility_score", None) or indicators.get("volatility", None) or mtf_data.get("volatility_score", None)
    volatility_score = _norm_0_1(volatility_score_raw, 0.0, 100.0)

    # ---- Market context ----
    # market_regime -> Trending/Ranging/High Volatility mapped to [-1,1] signed
    # market_regime
    market_regime_raw = regime.get("market_regime", None) or regime.get("regime", None) or regime.get("name", None)
    if market_regime_raw is None:
        market_regime = 0.0
    elif isinstance(market_regime_raw, str):
        mr = market_regime_raw.strip().lower()
        if "trending" in mr:
            market_regime = 1.0
        elif "ranging" in mr:
            market_regime = 0.0
        elif "high volatility" in mr or "high_volatility" in mr:
            market_regime = -1.0
        else:
            market_regime = 0.0
    else:
        market_regime = 0.0


    # session derived from UTC time (no external call required)
    now_utc = datetime.now(timezone.utc)
    session = _session_from_utc(now_utc)
    # Map session categorical -> [0,1]
    session_map = {"Asia": 0.25, "London": 0.55, "NY": 0.85}
    session_norm = session_map.get(session, 0.25)


    # spread
    spread_raw = market_data.get("spread", None) or indicators.get("spread", None)
    spread = _norm_0_1(spread_raw, 0.0, 0.05) if spread_raw is not None else 0.0
    if spread is None:
        spread = 0.0


    # ---- AI features ----
    ai_score_raw = ai_analysis.get("impact_score", None) or ai_analysis.get("ai_score", None) or ai_analysis.get("score", None)
    # normalize: if already 0-1 treat as such
    if ai_score_raw is not None:
        try:
            ai_sf = float(ai_score_raw)
            ai_score = ai_sf if 0.0 <= ai_sf <= 1.0 else _norm_0_1(ai_sf, 0.0, 100.0)
        except (TypeError, ValueError):
            ai_score = None
    else:
        ai_score = None
    if ai_score is None:
        ai_score = 0.0


    # sentiment_score (0-100)
    sentiment_score_raw = sentiment.get("sentiment_score", None) or sentiment.get("score", None)
    if sentiment_score_raw is not None:
        sentiment_score = _norm_0_1(sentiment_score_raw, 0.0, 100.0)
    else:
        sentiment_score = None
    if sentiment_score is None:
        sentiment_score = 0.0


    # news_impact_score (0-100)
    news_impact_score_raw = ai_analysis.get("news_impact_score", None) or ai_analysis.get("news_impact", None)
    if news_impact_score_raw is not None:
        news_impact_score = _norm_0_1(news_impact_score_raw, 0.0, 100.0)
    else:
        news_impact_score = None
    if news_impact_score is None:
        news_impact_score = 0.0


    # ---- Structured output ----
    # expected_entry from market_data if available
    expected_entry_raw = market_data.get("expected_entry", None) or market_data.get("entry_price", None) or market_data.get("entry", None)
    # We normalize expected_entry based on price range guard; if unavailable, None.
    # If QuantDinger provides raw expected_entry, we store it as-is via normalization guard with fallback.
    # NOTE: execution_dataset expected_entry is numeric in DB; for training prefer normalized.
    expected_entry = _norm_0_1(expected_entry_raw, 0.0, 2.0) if expected_entry_raw is not None else None

    expected_confidence_raw = ai_analysis.get("confidence", None) or ai_analysis.get("ai_confidence", None) or ai_analysis.get("expected_confidence", None)
    expected_confidence = _norm_0_1(expected_confidence_raw, 0.0, 1.0) if expected_confidence_raw is not None else None

    expected_final_score_raw = ai_analysis.get("final_score", None) or ai_analysis.get("expected_final_score", None)
    if expected_final_score_raw is not None:
        expected_final_score = _norm_0_1(expected_final_score_raw, 0.0, 100.0)
    else:
        expected_final_score = None

    direction_raw = ai_analysis.get("direction", None) or mtf_data.get("direction", None) or regime.get("direction", None)
    # Map BUY/SELL to [-1,1]
    direction_signed = _normalize_direction_to_signed(direction_raw)

    return {
        # TECHNICAL FEATURES
        "rsi": rsi,
        "atr": atr,
        "macd": macd,
        "trend_strength": trend_strength,
        "momentum_score": momentum_score,
        "volatility_score": volatility_score,

        # MARKET CONTEXT
        "market_regime": market_regime,
        "session": session_norm,
        "spread": spread,

        # AI FEATURES
        "ai_score": ai_score,
        "sentiment_score": sentiment_score,
        "news_impact_score": news_impact_score,

        # STRUCTURED OUTPUT
        "expected_entry": expected_entry,
        "expected_confidence": expected_confidence,
        "expected_final_score": expected_final_score,
        "direction": direction_signed,
    }
