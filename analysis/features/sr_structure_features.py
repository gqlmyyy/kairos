"""Support/resistance structure features, adapted from SR_Mapping_NN.

Reference: https://github.com/Mrizalfahlepi/SR_Mapping_NN (commit c0115aa).
That project's 26-feature set is the source of the *ideas* here — fractal
support/resistance, ATR-normalised distances, candle anatomy, fractal age.
None of its code, model, or fitted constants are used, for reasons recorded
in SR_MAPPING_INTEGRATION_REPORT.md.

The one defect that had to be fixed before any of it could be used
--------------------------------------------------------------------
Its Williams Fractal marks bar `i` as a fractal by comparing it against bars
`i+1` and `i+2`:

    if all(highs[i] > highs[i-j] and highs[i] > highs[i+j] for j in 1..2)

so `resistance`/`support` at bar i — and every feature derived from them —
depend on price two bars into the future. Demonstrated on that project's own
data: mutating only bars after index 3000 changed `fractal_down[3000]` from
NaN to 3115.20. Seven of its 26 features inherit the leak: dist_res_norm,
dist_sup_norm, sr_position, near_support, near_resistance,
bars_since_frac_up, bars_since_frac_down.

The fix is not to drop the idea but to respect when it is knowable. A fractal
centred on bar i is *confirmed* at bar i+CONFIRM_BARS, so at decision time t
the newest usable fractal is the one centred CONFIRM_BARS bars ago. Every
level here is therefore taken from the confirmed set only, which is also what
a live chart would show: a trader cannot see a fractal that has not printed.

Everything is computed from a `closed_slice` result, so the Phase 3 alignment
guarantees carry over unchanged. Research-only — this module is not part of
the production ten-feature contract.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence

MIN_BARS = 100
FRACTAL_LOOKBACK = 2
# A fractal centred on bar i needs FRACTAL_LOOKBACK bars after it to be
# confirmed, so it only becomes knowable FRACTAL_LOOKBACK bars later.
CONFIRM_BARS = FRACTAL_LOOKBACK

# Fraction of ATR within which price counts as "at" a level.
NEAR_TOLERANCE_ATR = 0.20

FEATURE_NAMES = (
    "dist_res_atr",
    "dist_sup_atr",
    "sr_position",
    "near_resistance",
    "near_support",
    "bars_since_frac_up",
    "bars_since_frac_down",
    "body_size_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "is_bullish",
    "ret_3_atr",
    "ret_5_atr",
    "ret_10_atr",
    "ret_20_atr",
    "atr_percentile",
    "atr_change_pct",
    "rsi_sma_10",
    "rsi_slope_3",
)

NEUTRAL: Dict[str, float] = {
    "dist_res_atr": 0.0, "dist_sup_atr": 0.0, "sr_position": 0.5,
    "near_resistance": 0.0, "near_support": 0.0,
    "bars_since_frac_up": 0.0, "bars_since_frac_down": 0.0,
    "body_size_atr": 0.0, "upper_wick_atr": 0.0, "lower_wick_atr": 0.0,
    "is_bullish": 0.0,
    "ret_3_atr": 0.0, "ret_5_atr": 0.0, "ret_10_atr": 0.0, "ret_20_atr": 0.0,
    "atr_percentile": 0.5, "atr_change_pct": 0.0,
    "rsi_sma_10": 50.0, "rsi_slope_3": 0.0,
}


def confirmed_fractals(
    highs: Sequence[float],
    lows: Sequence[float],
    lookback: int = FRACTAL_LOOKBACK,
) -> tuple:
    """Fractals that have actually printed, indexed by when they are knowable.

    Returns (up, down) arrays the same length as the input, where a value at
    index k is the fractal price CONFIRMED at bar k — that is, a fractal
    centred on bar k-lookback. Reading index k therefore never consumes a bar
    later than k, which is the property the reference implementation lacked.
    """
    n = len(highs)
    up: List[Optional[float]] = [None] * n
    down: List[Optional[float]] = [None] * n

    for centre in range(lookback, n - lookback):
        confirmed_at = centre + lookback
        if confirmed_at >= n:
            break
        if all(highs[centre] > highs[centre - j] and highs[centre] > highs[centre + j]
               for j in range(1, lookback + 1)):
            up[confirmed_at] = float(highs[centre])
        if all(lows[centre] < lows[centre - j] and lows[centre] < lows[centre + j]
               for j in range(1, lookback + 1)):
            down[confirmed_at] = float(lows[centre])
    return up, down


def _carry_forward(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    last: Optional[float] = None
    for value in values:
        if value is not None:
            last = value
        out.append(last)
    return out


def _bars_since(values: Sequence[Optional[float]]) -> List[float]:
    out: List[float] = []
    counter = 0.0
    for value in values:
        if value is not None:
            counter = 0.0
        else:
            counter += 1.0
        out.append(counter)
    return out


def _rsi_series(closes: Sequence[float], period: int = 14) -> List[float]:
    """Wilder RSI, matching the reference project's definition.

    Deliberately NOT the live simple-average RSI in live_parity_features: this
    module's `rsi_sma_10` and `rsi_slope_3` are new research features whose
    definition is fixed here and used identically in training and serving. The
    production `rsi` feature keeps its own live-parity definition; the two do
    not have to agree because they are different features with different names.
    """
    if len(closes) < period + 1:
        return [50.0] * len(closes)

    out = [50.0] * len(closes)
    gains = losses = 0.0
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain, avg_loss = gains / period, losses / period

    for i in range(period, len(closes)):
        if i > period:
            change = closes[i] - closes[i - 1]
            avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
            avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
        if avg_loss <= 0:
            out[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def _atr_series(candles: Sequence[Dict[str, Any]], period: int = 14) -> List[float]:
    if len(candles) < period + 1:
        return [0.0] * len(candles)
    out = [0.0] * len(candles)
    trs: List[float] = []
    for i, candle in enumerate(candles):
        high, low = float(candle["high"]), float(candle["low"])
        if i == 0:
            trs.append(high - low)
        else:
            prev_close = float(candles[i - 1]["close"])
            trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        if i >= period:
            out[i] = sum(trs[i - period + 1:i + 1]) / period
    return out


def build_sr_features(
    h4: Sequence[Dict[str, Any]],
    *,
    timestamp: float,
    atr: float,
) -> Optional[Dict[str, float]]:
    """S/R structure features knowable at `timestamp`, or None.

    `h4` must already be a `closed_slice` result. `atr` is the live-parity ATR
    the rest of the pipeline uses, passed in so distances are normalised by
    the same volatility measure the SL/TP barriers use.
    """
    if len(h4) < MIN_BARS or atr <= 0:
        return None

    highs = [float(c["high"]) for c in h4]
    lows = [float(c["low"]) for c in h4]
    closes = [float(c["close"]) for c in h4]
    opens = [float(c["open"]) for c in h4]

    up, down = confirmed_fractals(highs, lows)
    resistance = _carry_forward(up)[-1]
    support = _carry_forward(down)[-1]
    since_up = _bars_since(up)[-1]
    since_down = _bars_since(down)[-1]

    close = closes[-1]
    out: Dict[str, float] = dict(NEUTRAL)

    if resistance is not None:
        distance = resistance - close
        out["dist_res_atr"] = distance / atr
        out["near_resistance"] = 1.0 if abs(distance) <= atr * NEAR_TOLERANCE_ATR else 0.0
    if support is not None:
        distance = close - support
        out["dist_sup_atr"] = distance / atr
        out["near_support"] = 1.0 if abs(distance) <= atr * NEAR_TOLERANCE_ATR else 0.0
    if resistance is not None and support is not None and resistance > support:
        out["sr_position"] = (close - support) / (resistance - support)

    out["bars_since_frac_up"] = float(since_up)
    out["bars_since_frac_down"] = float(since_down)

    # Candle anatomy, ATR-normalised so it is comparable across instruments.
    open_ = opens[-1]
    high, low = highs[-1], lows[-1]
    out["body_size_atr"] = abs(close - open_) / atr
    out["upper_wick_atr"] = (high - max(close, open_)) / atr
    out["lower_wick_atr"] = (min(close, open_) - low) / atr
    out["is_bullish"] = 1.0 if close > open_ else 0.0

    # Returns in ATR units rather than percent: the reference used percent,
    # which is scale-dependent across instruments in exactly the way the
    # earlier KAIROS investigation found makes a tree split on instrument
    # identity instead of on market state.
    for period in (3, 5, 10, 20):
        if len(closes) > period:
            out[f"ret_{period}_atr"] = (close - closes[-1 - period]) / atr

    atrs = _atr_series(h4)
    window = [a for a in atrs[-168:] if a > 0]
    if window:
        below = sum(1 for a in window if a < atr)
        out["atr_percentile"] = below / len(window)
    if len(atrs) > 5 and atrs[-6] > 0:
        out["atr_change_pct"] = (atr - atrs[-6]) / atrs[-6] * 100.0

    rsis = _rsi_series(closes)
    if len(rsis) >= 10:
        out["rsi_sma_10"] = statistics.mean(rsis[-10:])
    if len(rsis) > 3:
        out["rsi_slope_3"] = rsis[-1] - rsis[-4]

    return {name: float(out[name]) for name in FEATURE_NAMES}


def build_vector(h4: Sequence[Dict[str, Any]], *, timestamp: float, atr: float
                 ) -> Optional[List[float]]:
    named = build_sr_features(h4, timestamp=timestamp, atr=atr)
    return None if named is None else [named[n] for n in FEATURE_NAMES]
