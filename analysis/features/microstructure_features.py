"""Spread- and volume-derived features. Research-only — not the production ten.

These are properties of the closed bar itself: MT5's rate struct returns
`spread` and `real_volume` alongside OHLC, in the same message, at the same
instant. Capturing them adds no look-ahead risk beyond what `closed_slice`
already guarantees for OHLC. What they DO require is graceful absence
handling — candle files fetched before `spread` was added to
`fetch_training_candles.py` do not have the key, and this module must say so
rather than treat a missing field as zero.

Nothing here touches `analysis.models.entry_feature_spec`. The production
contract stays exactly ten features; these are additional columns for the
information-discovery research dataset only, joined on but never mixed into
the deployed feature vector.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence

from analysis.features import timeframe_alignment as ta

MIN_BARS = 100

# Names in a fixed order, so a caller can build a stable column vector.
FEATURE_NAMES = (
    "spread_atr",
    "spread_percentile",
    "spread_zscore",
    "real_volume_zscore",
    "real_volume_percentile",
    "volume_zscore",
)


class AvailabilityError(ValueError):
    """A required field is not present in this candle series at all."""


def field_availability(candles: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """AVAILABLE / NOT AVAILABLE / INSUFFICIENT HISTORY per field.

    Checked on the raw series, not after truncation to a decision point, so a
    caller can decide once whether this instrument's export even carries these
    columns before trying to build features from them per row.
    """
    report: Dict[str, str] = {}
    for field in ("spread", "real_volume", "volume"):
        if not candles or field not in candles[0]:
            report[field] = "NOT AVAILABLE"
            continue
        if len(candles) < MIN_BARS:
            report[field] = "INSUFFICIENT HISTORY"
            continue
        values = [float(c.get(field, 0.0)) for c in candles]
        if all(v == values[0] for v in values):
            # A field present but constant (real_volume is frequently all
            # zero on FX/CFD brokers) carries no information regardless of
            # how it is later encoded.
            report[field] = "AVAILABLE (constant — no information)"
        else:
            report[field] = "AVAILABLE"
    return report


def _percentile_rank(window: List[float], value: float) -> float:
    if not window:
        return 50.0
    below = sum(1 for w in window if w < value)
    return 100.0 * below / len(window)


def _zscore(window: List[float], value: float) -> float:
    if len(window) < 2:
        return 0.0
    mean = statistics.mean(window)
    stdev = statistics.pstdev(window)
    return (value - mean) / stdev if stdev > 0 else 0.0


def build_microstructure_features(
    h4: Sequence[Dict[str, Any]],
    *,
    timestamp: float,
    atr: float,
) -> Optional[Dict[str, float]]:
    """Spread/volume features knowable at `timestamp`, or None.

    `h4` must already be a `closed_slice` result — this function does not
    re-check timeframe alignment, only that enough history exists and that the
    required fields are actually present (not defaulted to zero, which would
    misrepresent an unavailable field as a measured one).

    Returns None when there is insufficient history OR when none of the
    source fields are available at all, matching the None-on-insufficient-
    history contract `entry_features.build_entry_features` already uses.
    """
    if len(h4) < MIN_BARS:
        return None

    availability = field_availability(h4)
    if availability["spread"].startswith("NOT AVAILABLE"):
        return None

    window = h4[-MIN_BARS:]
    current = window[-1]

    spreads = [float(c["spread"]) for c in window]
    spread_now = spreads[-1]
    history = spreads[:-1]

    out: Dict[str, float] = {
        "spread_atr": spread_now / atr if atr > 0 else 0.0,
        "spread_percentile": _percentile_rank(history, spread_now),
        "spread_zscore": _zscore(history, spread_now),
    }

    for field, prefix in (("real_volume", "real_volume"), ("volume", "volume")):
        if availability.get(field, "NOT AVAILABLE").startswith("AVAILABLE") \
                and "constant" not in availability[field]:
            series = [float(c.get(field, 0.0)) for c in window]
            val = series[-1]
            hist = series[:-1]
            out[f"{prefix}_zscore"] = _zscore(hist, val)
            if prefix == "real_volume":
                out["real_volume_percentile"] = _percentile_rank(hist, val)
        else:
            # Explicit neutral marker, not a fabricated reading. Kept in the
            # vector so the column exists across every row (a hard requirement
            # for a feature matrix), but every row for this field carries the
            # same neutral value when the source itself is degenerate or
            # absent — which the information audit's variance/MI checks will
            # then correctly report as zero information, not silently omit.
            if prefix == "real_volume":
                out["real_volume_percentile"] = 50.0
            out[f"{prefix}_zscore"] = 0.0

    return {name: out[name] for name in FEATURE_NAMES}


def build_vector(h4: Sequence[Dict[str, Any]], *, timestamp: float, atr: float
                 ) -> Optional[List[float]]:
    named = build_microstructure_features(h4, timestamp=timestamp, atr=atr)
    return None if named is None else [named[name] for name in FEATURE_NAMES]
