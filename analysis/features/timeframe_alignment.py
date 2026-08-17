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
from collections import Counter
from datetime import datetime, timedelta, timezone
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


def diagnose_grid(candles: Sequence[Dict[str, Any]], timeframe: str) -> Dict[str, Any]:
    """Is this series on the UTC grid, on a shifted grid, or on no grid at all?

    A constant shift is a broker serving server time. The alignment
    arithmetic in this module survives it — every function above only ever
    compares timestamps to each other — but anything that converts a
    timestamp to a UTC hour (a session label, a calendar-day gap check) would
    be off by the offset. That is worth knowing precisely, which is why this
    reports the offset rather than only flagging violations.

    Moved here from scripts/validate_real_dataset.py so `classify_gaps` below
    can share it: a gap classifier that assumes a fixed weekend-close hour
    would silently mis-classify every broker whose clock is not on UTC.
    """
    span = duration(timeframe)
    offsets = Counter(int(float(c["t"]) % span) for c in candles)
    modal, modal_count = offsets.most_common(1)[0]
    return {
        "distinct_offsets": len(offsets),
        "modal_offset_seconds": modal,
        "modal_offset_hours": round(modal / 3600.0, 2),
        "modal_share": round(modal_count / len(candles), 6),
        "on_utc_grid": modal == 0 and len(offsets) == 1,
        "constant_offset": len(offsets) == 1,
    }


# Calendar dates (month, day) where gold/FX-CFD brokers commonly close
# entirely, regardless of weekday. Deliberately short: this is a supplement
# to the weekday-based weekly-close rule below, not a replacement for it, and
# an unlisted holiday still gets a fair chance to match that rule or the
# recurring-daily-pause rule before falling through to DATA_ERROR.
KNOWN_MARKET_HOLIDAYS_MONTH_DAY = frozenset({
    (12, 25),  # Christmas Day
    (1, 1),    # New Year's Day
})

# A Friday gap has to start in the afternoon/evening to plausibly be the
# weekly close — an outage that begins Friday at 09:00 UTC and happens to
# still be down at the weekend is not the same event as the market closing
# for the weekend, even though both end on a Saturday.
_FRIDAY_CLOSE_EARLIEST_HOUR_UTC = 12


def classify_gaps(
    candles: Sequence[Dict[str, Any]],
    timeframe: str,
    *,
    daily_pause_share_threshold: float = 0.4,
    suspicious_max_missing_bars: int = 2,
) -> Dict[str, Any]:
    """Classify every gap as an expected closure, a plausible thin patch, or
    an unexplained error — by calendar, not by expected duration.

    Matching an exact expected gap LENGTH is the wrong tool here: brokers
    differ on the exact minute their Friday close and Sunday open land, and
    an MT5 server clock that follows exchange DST drifts the UTC hour of
    every session boundary by an hour twice a year. Either one would make a
    duration-matching classifier misjudge real weekend closes for roughly
    half the year. Matching by day-of-week is immune to both, because it
    only asks *which day* a gap starts and ends on, never *what time*.

    Three categories, most permissive check tried first:

    * ``EXPECTED_MARKET_GAP`` — a gap that starts on Friday afternoon/evening
      and ends on Saturday, Sunday or Monday (the ordinary weekly close,
      including a holiday long weekend); or overlaps a known market holiday
      date; or recurs at the same UTC hour on enough other weekdays in this
      same series to be the broker's daily rollover pause rather than 150
      independent anomalies (checked against this series' own data, not a
      hard-coded hour, since that hour is broker- and DST-dependent too).
    * ``SUSPICIOUS_GAP`` — small (<= ``suspicious_max_missing_bars`` missing
      bars), does not recur, not blocking — plausible thin liquidity, worth a
      human glance.
    * ``DATA_ERROR`` — anything else. Fails closed: a gap this function
      cannot explain from the calendar or from its own recurrence is treated
      as broken, not whitelisted.

    Returns ``{"gaps": [...], "counts": {...}}``. Each gap entry carries its
    boundary timestamps, weekday, size and the reason it was classified the
    way it was, so a DATA_ERROR verdict can be audited row by row instead of
    trusted blindly.
    """
    span = duration(timeframe)
    gaps: List[Dict[str, Any]] = []

    for i in range(1, len(candles)):
        prev_t = float(candles[i - 1]["t"])
        t = float(candles[i]["t"])
        if t < prev_t:
            continue  # unsorted input is validate_series's problem, not this one's
        step = round((t - prev_t) / span)
        if step <= 1:
            continue
        start_dt = datetime.fromtimestamp(prev_t, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(t, tz=timezone.utc)
        gaps.append({
            "gap_start_t": prev_t,
            "gap_end_t": t,
            "gap_start_utc": start_dt.isoformat(),
            "gap_end_utc": end_dt.isoformat(),
            "start_weekday": start_dt.weekday(),  # Monday = 0 .. Sunday = 6
            "end_weekday": end_dt.weekday(),
            "start_hour_utc": start_dt.hour,
            "missing_bars": step - 1,
            "duration_hours": round((t - prev_t) / 3600.0, 2),
        })

    # Pass 1: the weekly close.
    for g in gaps:
        if (g["start_weekday"] == 4 and g["start_hour_utc"] >= _FRIDAY_CLOSE_EARLIEST_HOUR_UTC
                and g["end_weekday"] in (5, 6, 0)):
            g["category"] = "EXPECTED_MARKET_GAP"
            g["reason"] = "weekly close (Friday afternoon/evening -> weekend)"

    # Pass 2: named holidays, for the ones the weekly-close rule does not
    # already cover (a holiday landing on a weekday).
    for g in gaps:
        if "category" in g:
            continue
        day = datetime.fromtimestamp(g["gap_start_t"], tz=timezone.utc).date()
        end_day = datetime.fromtimestamp(g["gap_end_t"], tz=timezone.utc).date()
        overlap = False
        while day <= end_day:
            if (day.month, day.day) in KNOWN_MARKET_HOLIDAYS_MONTH_DAY:
                overlap = True
                break
            day += timedelta(days=1)
        if overlap:
            g["category"] = "EXPECTED_MARKET_GAP"
            g["reason"] = "known market holiday"

    # Pass 3: recurring daily pause, detected from this series' own data. A
    # broker that pauses briefly every day for the daily rollover produces
    # many small gaps clustered on the same UTC hour; that is one explained,
    # recurring event, not one anomaly per day.
    remaining = [g for g in gaps if "category" not in g]
    weekday_remaining = [g for g in remaining if g["start_weekday"] < 5]
    if weekday_remaining:
        hour_counts = Counter(g["start_hour_utc"] for g in weekday_remaining)
        dominant_hour, dominant_count = hour_counts.most_common(1)[0]
        # A share threshold alone is vacuous on a handful of gaps — one
        # isolated gap "dominates" its own hour 100% of the time. Recurrence
        # needs to actually recur.
        if (dominant_count >= 3
                and dominant_count / len(weekday_remaining) >= daily_pause_share_threshold):
            for g in weekday_remaining:
                if (g["start_hour_utc"] == dominant_hour
                        and g["missing_bars"] <= suspicious_max_missing_bars * 2):
                    g["category"] = "EXPECTED_MARKET_GAP"
                    g["reason"] = (
                        f"recurring daily pause at {dominant_hour:02d}:00 UTC "
                        f"({dominant_count}/{len(weekday_remaining)} weekday gaps land here)"
                    )

    # Pass 4: what is left is either small enough to be plausible thin
    # liquidity, or genuinely unexplained.
    for g in gaps:
        if "category" in g:
            continue
        if g["missing_bars"] <= suspicious_max_missing_bars:
            g["category"] = "SUSPICIOUS_GAP"
            g["reason"] = "small, non-recurring weekday gap — plausible thin liquidity"
        else:
            g["category"] = "DATA_ERROR"
            g["reason"] = (
                "not the weekly close, not a known holiday, not a recurring daily "
                "pause, and too large to be thin liquidity"
            )

    counts = Counter(g["category"] for g in gaps)
    return {"gaps": gaps, "counts": dict(counts)}
