"""Entry features, in groups, computed once for both training and live.

Why this module exists
----------------------
The sweep in Phase 2/3 showed the labelling is not the bottleneck: across nine
(SL, TP, horizon) configurations on three years of real candles, walk-forward
AUC never left 0.505-0.523, and in every case the fold-to-fold spread was
2-12x the distance from 0.5. Tripling the horizon moved AUC by less than 0.02.
The features cannot separate the label, so the features are what must change.

Three defects in the original ten are fixed here rather than carried forward:

1. **`market_regime` collapsed four regimes into two values.** RANGING,
   HIGH_VOLATILITY and LOW_VOLATILITY all encoded to 0.0, so the model could
   not distinguish them — and LOW_VOLATILITY was the only slice in the last
   run that scored above chance (AUC 0.609). Now one distinct value each.

2. **`atr` and `macd` were raw and pooled across symbols.** EURUSD ATR is
   ~0.0017 and XAUUSD ATR is ~41.7 — a 25,000x gap. A tree splitting on raw
   ATR is learning *which instrument this is*, not whether volatility is high.
   Everything here is scale-free: a ratio, a percentage of price, or a
   distance measured in ATRs.

3. **Several features were re-encodings of each other.** `momentum_score` was
   a three-bucket discretisation of `rsi`, which was already feature #1;
   `market_regime` and `trend_strength` were deterministic functions of the
   others. Ten slots carried about six independent dimensions.

Parity by construction (Phase 8)
--------------------------------
Live and training call :func:`build_entry_features` with raw candle lists. There
is no second implementation to drift — the failure mode that produced the
65-vs-10 mismatch. Every value is derived from candles at or before the entry
bar; nothing reads forward.

Groups exist so they can be added one at a time and each one's out-of-sample
contribution measured (Phase 7). Adding thirty features at once and reporting
the total is how overfitting hides.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from analysis.features import live_parity_features as lpf

# Candles needed before any feature is trustworthy. ma50 needs 50, the ATR
# window needs 28, the 20-bar structure window needs 20; 100 matches what live
# fetches and leaves headroom for the lag features.
MIN_BARS = lpf.LIVE_INDICATOR_WINDOW

# Feature groups, in the order Phase 7 adds them. `baseline` is the corrected
# version of the original ten; everything after is a candidate that must earn
# its place on out-of-sample lift.
FEATURE_GROUPS: Dict[str, tuple] = {
    "baseline": (
        "rsi_h1",
        "atr_pct",            # was raw `atr` — now scale-free
        "macd_atr",           # was raw `macd` — now in ATR units
        "trend_strength",
        "trend_score",
        "volatility_score",
        "market_regime",      # now 4 distinct values, not 2
        "session",
        "direction",
    ),
    "trend": (
        "adx",
        "di_plus",
        "di_minus",
        "di_spread",
        "ma20_slope_atr",
        "price_vs_ma20_atr",
        "price_vs_ma50_atr",
        "ma20_vs_ma50_atr",
        "trend_persistence",
    ),
    "volatility": (
        "atr_ratio",
        "atr_expansion",
        "bb_width_pct",
        "vol_percentile",
    ),
    "momentum": (
        "rsi_h4",
        "rsi_delta_3",
        "roc_5_atr",
        "roc_10_atr",
        "macd_slope_atr",
    ),
    "structure": (
        "dist_high20_atr",
        "dist_low20_atr",
        "range_position_20",
        "breakout_state",
        "pullback_depth",
    ),
    "mtf": (
        "h1_h4_trend_agree",
        "rsi_spread_h1_h4",
        "atr_ratio_h1_h4",
        "h1_slope_atr",
    ),
}

# Values substituted when a feature genuinely cannot be computed. Every one is
# neutral for its scale, and identical in both paths — a NaN reaching XGBoost
# would be treated as `missing` and follow a default branch, which is exactly
# the silent behaviour the feature contract exists to prevent.
NEUTRAL: Dict[str, float] = {
    "adx": 20.0, "di_plus": 20.0, "di_minus": 20.0, "di_spread": 0.0,
    "atr_ratio": 1.0, "atr_expansion": 1.0, "vol_percentile": 0.5,
    "range_position_20": 0.5, "trend_persistence": 0.5,
}


def group_names(groups: Optional[Sequence[str]] = None) -> List[str]:
    """Feature names for the selected groups, in a stable order."""
    selected = list(groups) if groups is not None else list(FEATURE_GROUPS)
    names: List[str] = []
    for g in selected:
        if g not in FEATURE_GROUPS:
            raise ValueError(f"unknown feature group {g!r}; "
                             f"known: {sorted(FEATURE_GROUPS)}")
        names.extend(FEATURE_GROUPS[g])
    return names


# ---------------------------------------------------------------------------
# Primitives — all read backwards from the last bar only
# ---------------------------------------------------------------------------

def _sma(values: Sequence[float], period: int) -> float:
    return sum(values[-period:]) / period


def _wilder_dmi(highs, lows, closes, period: int = 14):
    """ADX / DI+ / DI- with Wilder smoothing.

    Standard formulation, unlike the project's RSI and MACD. That is safe here
    because ADX is *new* — there is no deployed model whose input distribution
    could shift, and no existing live formula to stay bug-compatible with.
    """
    n = len(closes)
    if n < period * 2 + 1:
        return None

    plus_dm, minus_dm, trs = [], [], []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        trs.append(max(highs[i] - lows[i],
                       abs(highs[i] - closes[i - 1]),
                       abs(lows[i] - closes[i - 1])))

    def smooth(seq):
        acc = sum(seq[:period])
        out = [acc]
        for v in seq[period:]:
            acc = acc - acc / period + v
            out.append(acc)
        return out

    s_tr, s_plus, s_minus = smooth(trs), smooth(plus_dm), smooth(minus_dm)
    dis_plus, dis_minus, dxs = [], [], []
    for tr, p, m in zip(s_tr, s_plus, s_minus):
        if tr <= 0:
            continue
        dp, dm = 100.0 * p / tr, 100.0 * m / tr
        dis_plus.append(dp)
        dis_minus.append(dm)
        denom = dp + dm
        dxs.append(100.0 * abs(dp - dm) / denom if denom > 0 else 0.0)

    if not dxs:
        return None
    adx = sum(dxs[-period:]) / min(len(dxs), period)
    return adx, dis_plus[-1], dis_minus[-1]


def _bollinger_width_pct(closes: Sequence[float], period: int = 20, k: float = 2.0) -> float:
    window = closes[-period:]
    mid = sum(window) / period
    var = sum((c - mid) ** 2 for c in window) / period
    sd = math.sqrt(var)
    return (2 * k * sd / mid) if mid else 0.0


def _percentile_rank(values: Sequence[float], target: float) -> float:
    if not values:
        return NEUTRAL["vol_percentile"]
    below = sum(1 for v in values if v < target)
    return below / len(values)


def _atr_series(highs, lows, closes, period: int = 14) -> List[float]:
    trs = []
    for i in range(1, len(closes)):
        trs.append(max(highs[i] - lows[i],
                       abs(highs[i] - closes[i - 1]),
                       abs(lows[i] - closes[i - 1])))
    if len(trs) < period:
        return []
    return [sum(trs[j - period:j]) / period for j in range(period, len(trs) + 1)]


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------

def build_entry_features(
    h4: Sequence[Dict[str, float]],
    h1: Sequence[Dict[str, float]],
    *,
    direction: str,
    timestamp: float,
    groups: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, float]]:
    """Named features from raw candles. ``None`` when there is too little history.

    `h4` and `h1` must end at or before the entry bar — the caller is
    responsible for slicing, and nothing here looks past the last element.
    Returning None rather than a partial vector is deliberate: live substitutes
    fallback constants when history is short, and such a row must never become
    a training example.
    """
    from analysis.models import entry_feature_spec as spec

    if len(h4) < MIN_BARS or len(h1) < MIN_BARS:
        return None

    selected = list(groups) if groups is not None else list(FEATURE_GROUPS)
    out: Dict[str, float] = {}

    h4_ind = lpf.live_indicators(h4)
    h1_ind = lpf.live_indicators(h1)
    if h4_ind is None or h1_ind is None:
        return None

    h4_close = [float(c["close"]) for c in h4]
    h4_high = [float(c["high"]) for c in h4]
    h4_low = [float(c["low"]) for c in h4]
    h1_close = [float(c["close"]) for c in h1]

    price = h4_close[-1]
    atr = float(h4_ind["atr"])
    # Guard once: a flat window gives ATR 0, and every ATR-normalised feature
    # below would divide by it.
    atr_safe = atr if atr > 0 else max(price * 1e-6, 1e-9)

    trend_score, trend_dir = lpf.trend_score_from_indicators(h4_ind)
    _, mom_dir = lpf.momentum_score_from_indicators(h1_ind)
    vol_score = lpf.volatility_score_from_indicators(h1_ind)
    regime = lpf.regime_from_scores(trend_dir, vol_score)

    # --- baseline (corrected) ------------------------------------------------
    if "baseline" in selected:
        out["rsi_h1"] = float(h1_ind["rsi"])
        out["atr_pct"] = atr / price if price else 0.0
        out["macd_atr"] = float(h1_ind["macd"]) / atr_safe
        out["trend_strength"] = spec.encode_trend_strength(
            lpf.mtf_strength_from_directions(trend_dir, mom_dir, mom_dir))
        out["trend_score"] = float(trend_score)
        out["volatility_score"] = float(vol_score)
        out["market_regime"] = encode_regime_4(regime)
        out["session"] = spec.encode_session(spec.session_from_timestamp(timestamp))
        out["direction"] = spec.encode_direction(direction)

    # --- trend ---------------------------------------------------------------
    if "trend" in selected:
        dmi = _wilder_dmi(h4_high, h4_low, h4_close)
        if dmi is None:
            out["adx"], out["di_plus"], out["di_minus"] = (
                NEUTRAL["adx"], NEUTRAL["di_plus"], NEUTRAL["di_minus"])
        else:
            out["adx"], out["di_plus"], out["di_minus"] = dmi
        out["di_spread"] = out["di_plus"] - out["di_minus"]

        ma20 = _sma(h4_close, 20)
        ma50 = _sma(h4_close, 50)
        ma20_prev = _sma(h4_close[:-5], 20)
        out["ma20_slope_atr"] = (ma20 - ma20_prev) / atr_safe
        out["price_vs_ma20_atr"] = (price - ma20) / atr_safe
        out["price_vs_ma50_atr"] = (price - ma50) / atr_safe
        out["ma20_vs_ma50_atr"] = (ma20 - ma50) / atr_safe
        # Fraction of the last 20 bars closing on the same side of MA20 as now.
        side = 1.0 if price > ma20 else -1.0
        agree = sum(1 for j in range(-20, 0)
                    if (h4_close[j] - _sma(h4_close[:len(h4_close) + j + 1], 20)) * side > 0)
        out["trend_persistence"] = agree / 20.0

    # --- volatility ----------------------------------------------------------
    if "volatility" in selected:
        out["atr_ratio"] = float(h4_ind["atr_ratio"])
        atrs = _atr_series(h4_high, h4_low, h4_close)
        if len(atrs) >= 10:
            out["atr_expansion"] = atrs[-1] / atrs[-6] if atrs[-6] > 0 else NEUTRAL["atr_expansion"]
            out["vol_percentile"] = _percentile_rank(atrs, atrs[-1])
        else:
            out["atr_expansion"] = NEUTRAL["atr_expansion"]
            out["vol_percentile"] = NEUTRAL["vol_percentile"]
        out["bb_width_pct"] = _bollinger_width_pct(h4_close)

    # --- momentum ------------------------------------------------------------
    if "momentum" in selected:
        out["rsi_h4"] = float(h4_ind["rsi"])
        out["rsi_delta_3"] = float(h1_ind["rsi"]) - lpf.live_rsi(h1_close[:-3])
        out["roc_5_atr"] = (price - h4_close[-6]) / atr_safe
        out["roc_10_atr"] = (price - h4_close[-11]) / atr_safe
        out["macd_slope_atr"] = (
            float(h1_ind["macd"]) - lpf.live_macd(h1_close[:-3])) / atr_safe

    # --- structure -----------------------------------------------------------
    if "structure" in selected:
        hi20 = max(h4_high[-20:])
        lo20 = min(h4_low[-20:])
        rng = hi20 - lo20
        out["dist_high20_atr"] = (hi20 - price) / atr_safe
        out["dist_low20_atr"] = (price - lo20) / atr_safe
        out["range_position_20"] = ((price - lo20) / rng) if rng > 0 else NEUTRAL["range_position_20"]
        # +1 closing above the prior 20-bar high, -1 below the prior low, else 0.
        prior_hi = max(h4_high[-21:-1])
        prior_lo = min(h4_low[-21:-1])
        out["breakout_state"] = 1.0 if price > prior_hi else (-1.0 if price < prior_lo else 0.0)
        # How far price has retraced from the recent extreme, in ATRs, signed by
        # which extreme it is retracing from.
        out["pullback_depth"] = ((hi20 - price) / atr_safe if price > (lo20 + rng / 2)
                                 else -(price - lo20) / atr_safe)

    # --- multi-timeframe -----------------------------------------------------
    if "mtf" in selected:
        h1_trend_score, h1_trend_dir = lpf.trend_score_from_indicators(h1_ind)
        out["h1_h4_trend_agree"] = (
            1.0 if (h1_trend_dir == trend_dir and trend_dir != "neutral")
            else (-1.0 if (h1_trend_dir != trend_dir
                           and "neutral" not in (h1_trend_dir, trend_dir)) else 0.0))
        out["rsi_spread_h1_h4"] = float(h1_ind["rsi"]) - float(h4_ind["rsi"])
        h1_atr = float(h1_ind["atr"])
        out["atr_ratio_h1_h4"] = h1_atr / atr_safe
        h1_ma20 = _sma(h1_close, 20)
        h1_ma20_prev = _sma(h1_close[:-5], 20)
        out["h1_slope_atr"] = (h1_ma20 - h1_ma20_prev) / max(h1_atr, 1e-9)

    return out


def encode_regime_4(regime: Any) -> float:
    """One distinct value per regime.

    The original map had only RANGING=0 and TRENDING=1, so HIGH_VOLATILITY and
    LOW_VOLATILITY both fell to the 0 default and were indistinguishable from
    RANGING. That mattered: LOW_VOLATILITY was the only per-regime slice in the
    last training run to score above chance.
    """
    return {
        "RANGING": 0.0,
        "TRENDING": 1.0,
        "HIGH_VOLATILITY": 2.0,
        "LOW_VOLATILITY": 3.0,
        "ranging": 0.0,
        "trending": 1.0,
    }.get(regime, 0.0)


def build_vector(
    h4, h1, *, direction: str, timestamp: float,
    groups: Optional[Sequence[str]] = None,
) -> Optional[List[float]]:
    """The named features flattened in group order — what XGBoost consumes."""
    named = build_entry_features(h4, h1, direction=direction,
                                 timestamp=timestamp, groups=groups)
    if named is None:
        return None
    return [float(named[n]) for n in group_names(groups)]
