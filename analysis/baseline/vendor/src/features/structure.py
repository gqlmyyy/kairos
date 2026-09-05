"""Market-structure features (fractals, swing points, support/resistance).

These require future bars to CONFIRM a pivot at bar i (a fractal high at i
needs `window` bars after i with lower highs). That confirmation information
does not exist until bar i + window. Section 12/15 of the spec forbids using
an unconfirmed pivot before its confirmation instant, so every signal here is
shifted forward by (window + confirmation_lag) rows before being exposed as a
feature — the value seen at row j only ever reflects pivots that were fully
confirmed at or before row j.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def support_resistance(high: pd.Series, low: pd.Series, lookback: int) -> pd.DataFrame:
    """Causal rolling support/resistance: simply the trailing high/low, which
    requires no future confirmation (unlike fractals)."""
    resistance = high.rolling(window=lookback, min_periods=lookback).max()
    support = low.rolling(window=lookback, min_periods=lookback).min()
    return pd.DataFrame({"resistance_level": resistance, "support_level": support})


def fractal_signals(high: pd.Series, low: pd.Series, window: int, confirmation_lag: int) -> pd.DataFrame:
    """Bill Williams-style fractal: a fractal high/low at bar i requires
    `window` bars on each side confirming it's a local extreme. The raw
    detection at index i therefore needs data through i+window, so the
    exposed feature is shifted forward by (window + confirmation_lag) so it
    is never available before it could actually be known.
    """
    n = len(high)
    is_fractal_high = np.zeros(n, dtype=float)
    is_fractal_low = np.zeros(n, dtype=float)
    h = high.to_numpy()
    l = low.to_numpy()
    for i in range(window, n - window):
        left_h, right_h = h[i - window:i], h[i + 1:i + window + 1]
        if h[i] > left_h.max() and h[i] > right_h.max():
            is_fractal_high[i] = 1.0
        left_l, right_l = l[i - window:i], l[i + 1:i + window + 1]
        if l[i] < left_l.min() and l[i] < right_l.min():
            is_fractal_low[i] = 1.0

    shift = window + confirmation_lag
    fractal_high = pd.Series(is_fractal_high, index=high.index).shift(shift).fillna(0.0)
    fractal_low = pd.Series(is_fractal_low, index=low.index).shift(shift).fillna(0.0)
    return pd.DataFrame({"fractal_high": fractal_high, "fractal_low": fractal_low})


def swing_points(high: pd.Series, low: pd.Series, lookback: int) -> pd.DataFrame:
    """Simplified swing high/low using the same window+lag-confirmed fractal
    logic, then forward-filled so the most recent confirmed swing level is
    always available (never an in-progress, unconfirmed one)."""
    window = max(2, lookback // 4)
    fr = fractal_signals(high, low, window=window, confirmation_lag=window)
    swing_high_level = high.where(fr["fractal_high"] == 1.0).ffill()
    swing_low_level = low.where(fr["fractal_low"] == 1.0).ffill()
    return pd.DataFrame({"swing_high_level": swing_high_level, "swing_low_level": swing_low_level})
