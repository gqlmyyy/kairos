"""When is a candle knowable, and which candles may a decision at time t see?

The entry_v2 dataset got this wrong in the most expensive way available: it
attached "the latest H4 candle at or before t", where candle stamps are OPEN
times. The candle whose open is at or before t is the one *still forming*, and
its close, high and low are up to three hours of future price. Measured on the
shipped dataset, h4_close equalled the H1 close three hours later 96.6% of the
time.

Two definitions fix it, and everything else here follows from them.

**A candle stamped T on a timeframe of duration D is knowable only from T+D.**
Its open is known at T, but its close, high, low and therefore every indicator
derived from it are not settled until it closes. Nothing may read a candle
before `close_time(T) = T + D`.

**The decision timestamp is a close time, not an open time.** The bot wakes
when a bar closes, computes indicators from bars that have closed, and acts.
So a decision "on bar i" happens at `close_time(bar i)`, and the information
set is every candle, on every timeframe, that has closed by then.

This module is deliberately timeframe-agnostic. The audit found the bug in H4,
but nothing about it was specific to H4 — the same forward-fill would have
leaked a day of future price from D1, and more from W1. Any timeframe the
system adds gets the same treatment by construction.
"""

from __future__ import annotations

import bisect
from typing import Any, Dict, List, Optional, Sequence

# Seconds per bar. Names match config.TF_TREND / TF_DECISION / TF_TIMING and
# the MT5 client's timeframe strings.
TIMEFRAME_SECONDS: Dict[str, int] = {
    "M1": 60,
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
    "D1": 24 * 60 * 60,
    "W1": 7 * 24 * 60 * 60,
}


class AlignmentError(ValueError):
    """Raised when a series cannot be aligned safely."""


def duration(timeframe: str) -> int:
    try:
        return TIMEFRAME_SECONDS[timeframe.strip().upper()]
    except KeyError:
        raise AlignmentError(
            f"unknown timeframe {timeframe!r}; known: {sorted(TIMEFRAME_SECONDS)}"
        ) from None


def close_time(open_time: float, timeframe: str) -> float:
    """The moment a candle stamped `open_time` becomes knowable."""
    return float(open_time) + duration(timeframe)


def decision_time(candles: Sequence[Dict[str, Any]], index: int, timeframe: str) -> float:
    """When a decision taken "on bar `index`" actually happens.

    This is the bar's close, which is the first moment its indicators exist.
    Treating the bar's *open* as the decision time is the error that let three
    hours of future price into the entry_v2 features.
    """
    return close_time(float(candles[index]["t"]), timeframe)


def last_closed_index(
    candles: Sequence[Dict[str, Any]],
    timeframe: str,
    at: float,
) -> Optional[int]:
    """Index of the newest candle that has fully closed by `at`.

    Returns None when no candle has closed yet. Assumes `candles` is sorted by
    `t` ascending, which `validate_series` checks.
    """
    if not candles:
        return None
    span = duration(timeframe)
    # A candle at index i is usable when t[i] + span <= at, i.e. t[i] <= at - span.
    # bisect_right over the open times finds the first candle past that bound.
    #
    # The `key=` form matters: materialising [float(c["t"]) for c in candles]
    # here made the lookup O(n), and this is called once per decision bar, so
    # building a 20,000-bar M15 dataset spent most of its time rebuilding the
    # same timestamp list 20,000 times. With key= it is a genuine O(log n)
    # search over the candles themselves.
    cutoff = float(at) - span
    position = bisect.bisect_right(candles, cutoff, key=lambda c: float(c["t"]))
    return position - 1 if position > 0 else None


def closed_slice(
    candles: Sequence[Dict[str, Any]],
    timeframe: str,
    at: float,
) -> List[Dict[str, Any]]:
    """Every candle knowable at `at`, oldest first.

    This is the only sanctioned way to hand history to an indicator. Passing a
    raw slice risks including the forming bar, which is exactly the defect this
    module exists to prevent.
    """
    index = last_closed_index(candles, timeframe, at)
    return [] if index is None else list(candles[: index + 1])


def closed_window(
    candles: Sequence[Dict[str, Any]],
    timeframe: str,
    at: float,
    size: int,
) -> List[Dict[str, Any]]:
    """The last `size` candles knowable at `at`, oldest first.

    Identical guarantee to `closed_slice` — nothing that has not closed by
    `at` can appear — but bounded in length. `closed_slice` copies the whole
    prefix, so calling it once per decision bar is quadratic in the length of
    the series: on 20,000 M15 bars that is ~200M element copies before a
    single feature is computed. Every consumer here reads a bounded lookback
    (indicators 100 bars, the ATR percentile 168), so handing them the whole
    history was only ever waste.

    Pass a `size` comfortably larger than the deepest lookback any consumer
    needs; too small silently shortens an indicator's window rather than
    raising, which is why callers state it explicitly.
    """
    index = last_closed_index(candles, timeframe, at)
    if index is None:
        return []
    start = max(0, index + 1 - size)
    return list(candles[start:index + 1])


def next_executable_index(
    candles: Sequence[Dict[str, Any]],
    timeframe: str,
    at: float,
) -> Optional[int]:
    """The first bar that opens at or after `at` — where an order can fill.

    A decision made at a bar's close cannot be executed at that close: the
    price is already history. The earliest tradeable moment is the open of the
    next bar, so that bar's open is the entry price, and the walk-forward for
    TP/SL starts there.
    """
    if not candles:
        return None
    position = bisect.bisect_left(candles, float(at), key=lambda c: float(c["t"]))
    return position if position < len(candles) else None


def validate_series(
    candles: Sequence[Dict[str, Any]],
    timeframe: str,
    *,
    max_gap_bars: int = 0,
) -> Dict[str, Any]:
    """Structural checks on one (symbol, timeframe) series.

    Reports rather than raises, so a caller can decide what is fatal. Gaps are
    counted but not condemned: FX closes at weekends and holidays, so a series
    with no gaps at all would be the suspicious one.
    """
    issues: Dict[str, Any] = {
        "count": len(candles),
        "unsorted": 0,
        "duplicate_timestamps": 0,
        "misaligned_to_grid": 0,
        "bad_ohlc": 0,
        "non_finite": 0,
        "gaps": 0,
        "largest_gap_bars": 0,
    }
    if not candles:
        return issues

    span = duration(timeframe)
    previous: Optional[float] = None
    seen = set()

    for candle in candles:
        t = float(candle["t"])
        if previous is not None:
            if t < previous:
                issues["unsorted"] += 1
            else:
                step = round((t - previous) / span)
                if step > 1:
                    issues["gaps"] += 1
                    issues["largest_gap_bars"] = max(issues["largest_gap_bars"], step - 1)
        if t in seen:
            issues["duplicate_timestamps"] += 1
        seen.add(t)

        # Candle opens should land on a multiple of the timeframe. A series
        # that does not is usually two timeframes concatenated, or a broker
        # offset that will silently shift every alignment.
        if t % span != 0:
            issues["misaligned_to_grid"] += 1

        try:
            o, h, l, c = (float(candle["open"]), float(candle["high"]),
                          float(candle["low"]), float(candle["close"]))
        except (KeyError, TypeError, ValueError):
            issues["bad_ohlc"] += 1
            previous = t
            continue

        if not all(v == v and abs(v) != float("inf") for v in (o, h, l, c)):
            issues["non_finite"] += 1
        elif h < max(o, c) or l > min(o, c) or h < l:
            issues["bad_ohlc"] += 1

        previous = t

    if max_gap_bars and issues["largest_gap_bars"] > max_gap_bars:
        issues["gap_exceeds_limit"] = True
    return issues


def assert_no_lookahead(
    candles: Sequence[Dict[str, Any]],
    timeframe: str,
    at: float,
) -> None:
    """Hard check that a slice contains nothing unknowable at `at`.

    Cheap enough to run inside a dataset build, and the one assertion that
    would have caught the entry_v2 defect on the first row.
    """
    span = duration(timeframe)
    for candle in candles:
        settled = float(candle["t"]) + span
        if settled > at:
            raise AlignmentError(
                f"{timeframe} candle opening at {candle['t']} closes at {settled}, "
                f"after the decision time {at} — it is not knowable yet"
            )
