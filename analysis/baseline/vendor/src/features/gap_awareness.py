"""Gap-aware feature validity (Phase 3, Section 6).

Feature Engineering NEVER repairs gaps -- no forward fill, no interpolation,
no synthetic candles, no timestamp edits (that remains true everywhere in
src/features). What this module adds is the PREDEFINED POLICY for what a
feature's status is when its lookback window crosses a gap or sits inside
the warmup region:

    FEATURE_VALID          (0)  window fully backed by contiguous bars
    INSUFFICIENT_HISTORY   (1)  fewer than minimum_history bars exist yet
    GAP_CROSSED            (2)  window spans a break in the bar sequence

The policy is FLAG-ONLY: values are still computed from the real bars that
exist (an indicator across a weekend closure is computed from the actual
adjacent candles, exactly like any charting package). Nothing is dropped,
filled or fabricated here; the flags travel with the dataset so a later
phase can make an informed dataset-level decision with full information.

Everything is vectorized: one diff + rolling-max per window size, never an
iterrows loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_VALID = 0
INSUFFICIENT_HISTORY = 1
GAP_CROSSED = 2

STATUS_NAMES = {
    FEATURE_VALID: "feature_valid",
    INSUFFICIENT_HISTORY: "insufficient_history",
    GAP_CROSSED: "gap_crossed",
}


def _break_indicator(timestamps: pd.Series, timeframe_minutes: int) -> np.ndarray:
    """b[i] == 1 iff the bar at i does not follow i-1 by exactly tf minutes.

    Row 0 has no predecessor and is NOT itself a break (its status is decided
    purely by insufficient_history)."""
    ts = pd.to_datetime(timestamps)
    if len(ts) == 0:
        return np.zeros(0, dtype=np.int8)
    expected = ts.shift(1) + pd.Timedelta(minutes=timeframe_minutes)
    # `np.array(..., copy=True)`: pandas 3 hands back a READ-ONLY view from
    # .to_numpy() under copy-on-write, and this function writes to index 0.
    brk = np.array((ts != expected).to_numpy(), copy=True)
    brk[0] = False
    return brk.astype(np.int8)


def feature_validity(
    timestamps: pd.Series,
    timeframe_minutes: int,
    windows: dict[str, int],
) -> pd.DataFrame:
    """Per-feature validity codes for every row.

    `windows` maps feature name -> effective minimum_history (bars including
    the current one). A feature at row i reads rows [i-w+1..i]. The break
    between rows j-1 and j lies INSIDE that read-set iff
    j-1 >= i-w+1 and j <= i, i.e. iff i in [j .. j+w-2]. Equivalently:
    any break in the trailing (w-1)-long window ending at i. A single-bar
    window (w == 1) therefore can never cross a gap. Insufficient history
    (fewer than w rows exist yet) takes precedence over gap-crossing."""
    n = len(timestamps)
    breaks = _break_indicator(timestamps, timeframe_minutes)
    out = {}
    for name, w in windows.items():
        w = max(int(w), 1)
        codes = np.full(n, FEATURE_VALID, dtype=np.int8)
        if n:
            span = max(w - 1, 1)
            rolled = pd.Series(breaks).rolling(window=span, min_periods=1).max().to_numpy()
            if w > 1:
                codes[rolled > 0] = GAP_CROSSED
            warmup = min(w - 1, n)
            codes[:warmup] = INSUFFICIENT_HISTORY
        out[name] = codes
    return pd.DataFrame(out, index=pd.RangeIndex(n))


def aggregate_validity(validity: pd.DataFrame) -> pd.Series:
    """Worst-case status across all features per row:
    GAP_CROSSED > INSUFFICIENT_HISTORY > FEATURE_VALID."""
    if validity.shape[1] == 0:
        return pd.Series(np.zeros(len(validity), dtype=np.int8), index=validity.index)
    arr = validity.to_numpy(dtype=np.int8)
    worst = np.maximum(arr.max(axis=1), 0)
    # max() alone would let GAP_CROSSED(2) beat INSUFFICIENT_HISTORY(1), which
    # is already the intended priority; keep the explicit ordering documented.
    return pd.Series(worst.astype(np.int8), index=validity.index)


def validity_summary(codes: pd.Series) -> dict[str, int]:
    """Counts per human-readable status, for reports/dataset metadata."""
    counts = codes.value_counts()
    summary = {STATUS_NAMES[c]: int(counts.get(c, 0)) for c in STATUS_NAMES}
    total = int(len(codes))
    summary["total_rows"] = total
    return summary
