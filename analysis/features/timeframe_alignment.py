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
import statistics
from collections import Counter
from datetime import date, datetime, timedelta, timezone
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


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher). Good Friday and
    Easter Monday closures are keyed off this, since neither has a fixed
    (month, day)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day_of_month = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day_of_month)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th occurrence of `weekday` (Monday=0) in `month`, 1-indexed."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        last_of_month = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_of_month = date(year, month + 1, 1) - timedelta(days=1)
    offset = (last_of_month.weekday() - weekday) % 7
    return last_of_month - timedelta(days=offset)


def us_market_holidays(year: int) -> set:
    """Dates this instrument's broker is evidenced to close for.

    Deliberately not a textbook exchange-holiday list — every entry here is
    backed by an actual gap seen in the real XAUUSD M30 export (see
    M30_GAP_VALIDATION_REPORT.md). MLK Day and Presidents' Day are common
    NYSE holidays but were NOT included because nothing in the observed gap
    pattern evidenced gold trading actually pausing for them; adding them
    on assumption would risk waving through a genuinely unexplained gap
    that only coincides with the date. An unlisted holiday still gets a
    fair chance at the weekly-close or broker-maintenance rules below
    before falling through to DATA_ERROR — the point of a short, evidenced
    list is that a truly novel unexplained gap stays unexplained.

    Computed per year, not hard-coded per date, so this holds across the
    full multi-year span rather than only whichever year prompted it.
    """
    easter = _easter_sunday(year)
    return {
        date(year, 1, 1),                    # New Year's Day
        easter - timedelta(days=2),          # Good Friday
        _last_weekday(year, 5, 0),            # Memorial Day: last Monday of May
        date(year, 6, 19),                    # Juneteenth
        date(year, 7, 4),                     # Independence Day
        _nth_weekday(year, 9, 0, 1),           # Labor Day: 1st Monday of September
        _nth_weekday(year, 11, 3, 4),          # Thanksgiving: 4th Thursday of November
        date(year, 12, 25),                   # Christmas Day
    }


def _overlaps_known_holiday(start_t: float, end_t: float) -> bool:
    day = datetime.fromtimestamp(start_t, tz=timezone.utc).date()
    end_day = datetime.fromtimestamp(end_t, tz=timezone.utc).date()
    holidays: set = set()
    years_loaded: set = set()
    while day <= end_day:
        if day.year not in years_loaded:
            holidays |= us_market_holidays(day.year)
            years_loaded.add(day.year)
        if day in holidays:
            return True
        day += timedelta(days=1)
    return False


_WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# A Friday close has to land in the afternoon/evening of the LAST PRESENT
# candle (`stop_hour_utc`) to plausibly be the weekly close at all — this is
# the one guard against a genuine multi-day outage that happens to start
# early Friday and end over the weekend being mistaken for the market simply
# closing on schedule. It is checked against when trading actually stopped,
# not the first missing bar's hour, so it survives a close landing exactly
# on a midnight boundary (stop 23:30 Friday, first missing bar dated
# Saturday 00:00) without needing a learned daily-close hour first.
_FRIDAY_CLOSE_EARLIEST_HOUR_UTC = 12


def _daily_close_hours(gaps: List[Dict[str, Any]], *, min_recurrence: int) -> set:
    """Which UTC hour(s) this series' broker consistently starts a daily
    close at.

    There can legitimately be more than one: an MT5 server clock that
    follows exchange DST drifts this hour by an hour twice a year, splitting
    a multi-year series into two (or more) equally legitimate clusters
    rather than one dominant hour — a classifier that only ever looks for a
    single "most common" hour silently fails half the year.

    Filtered on `stop_weekday` (the last PRESENT candle's weekday), not
    `start_weekday` (the first MISSING candle's) — a close that lands
    exactly on a midnight boundary (last trade Thursday 23:30, first
    missing bar technically dated Friday 00:00) must count toward
    Thursday's cluster, the day trading actually stopped, not get silently
    reassigned to the next calendar day by an artifact of the boundary.
    """
    hour_counts = Counter(g["start_hour_utc"] for g in gaps if g["stop_weekday"] < 5)
    return {hour for hour, n in hour_counts.items() if n >= min_recurrence}


def _pause_size_stats(gaps: List[Dict[str, Any]], hour: int,
                       *, max_bars_for_typical: int) -> Optional[Dict[str, float]]:
    """The routine size of the daily pause at `hour`, and how much it
    actually varies — from short instances only, since a rare multi-day
    holiday gap that happens to start at the same hour must not drag the
    median toward "anything is typical".

    Returns the median and its MAD (median absolute deviation), so the
    acceptance band in `classify_gaps` is driven by how consistent this
    broker's pause genuinely is, not a fixed percentage. A broker whose
    daily pause is always exactly N bars gets a tight band; one whose pause
    jitters gets a looser one, sized to the jitter actually observed rather
    than guessed.
    """
    sizes = [g["missing_bars"] for g in gaps
             if g["start_hour_utc"] == hour and g["stop_weekday"] < 5
             and g["missing_bars"] <= max_bars_for_typical]
    if not sizes:
        return None
    median = statistics.median(sizes)
    mad = statistics.median(abs(s - median) for s in sizes)
    return {"median": median, "mad": mad}


def classify_gaps(
    candles: Sequence[Dict[str, Any]],
    timeframe: str,
    *,
    min_daily_recurrence: int = 20,
    pause_tolerance_mad_multiple: float = 2.0,
    max_bars_for_typical_pause: int = 16,
    suspicious_max_missing_bars: int = 2,
) -> Dict[str, Any]:
    """Classify every gap as an expected closure, a plausible thin patch, or
    an unexplained error — by calendar and by this series' own recurring
    structure, never by matching a hard-coded expected duration.

    Matching an exact expected gap LENGTH is the wrong tool: brokers differ
    on the exact minute their close lands, and an MT5 server clock that
    follows exchange DST drifts the UTC hour of every session boundary by an
    hour twice a year. A duration-matching classifier would misjudge real
    closures for roughly half the year. This one instead:

    1. Learns this series' own recurring daily-close hour(s) and how
       consistent the pause size actually is at each one (`_daily_close_hours`,
       `_pause_size_stats`) — data-driven, not assumed, and not capped at an
       arbitrary bar count: a broker's daily maintenance window can
       legitimately run for hours, and the evidence that it is routine is
       that it recurs at a consistent hour with a consistent size, not that
       it is short. The acceptance band is `median ± max(pause_tolerance_mad_multiple
       * MAD, 1)` — a broker whose pause is always exactly N bars gets a tight
       band (so an irregular gap that happens to land on the same hour is NOT
       swallowed just because it is nearby in time), one with genuine jitter
       gets a band sized to the jitter actually observed, not a guessed
       percentage.
    2. ``weekly_close`` — starts on Friday at a recognized daily-close hour
       (or, if none has been established yet, the afternoon/evening
       fallback) and ends Saturday, Sunday or Monday.
    3. ``known_holiday`` — the gap's date range overlaps a US market holiday
       actually evidenced in this instrument's gap history (see
       `us_market_holidays`), including floating ones like Good Friday.
    4. ``broker_maintenance`` — starts at a recognized daily-close hour on a
       weekday, and its size is within tolerance of that hour's typical
       size — the same event recurring, not a fresh anomaly each day.
    5. ``thin_liquidity_or_irregular_session`` (``SUSPICIOUS_GAP``) — small
       (<= `suspicious_max_missing_bars`), not otherwise explained, not
       blocking.
    6. ``unexplained_missing_candles`` (``DATA_ERROR``) — anything else.
       Fails closed: nothing here whitelists a gap it cannot actually
       explain.

    Returns ``{"gaps": [...], "counts": {...}, "daily_close_hours_utc": [...]}``.
    Each gap entry carries its boundary timestamps (the *missing* window:
    `gap_start_t` is the first missing bar's open, `gap_end_t` is when data
    resumes), weekday, size, category and reason, so a DATA_ERROR verdict
    can be audited row by row instead of trusted blindly.
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
        missing = step - 1
        gap_start_t = prev_t + span  # first missing bar's open == last present bar's close
        gap_end_t = t                # data resumes here
        start_dt = datetime.fromtimestamp(gap_start_t, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(gap_end_t, tz=timezone.utc)
        # `stop_dt` is the last PRESENT candle (prev_t), used for weekday
        # classification instead of `start_dt` — see `_daily_close_hours`.
        # `start_weekday`/`start_hour_utc` stay start_dt-based because that
        # is how a gap is reported and read ("22:30 -> 01:00"); only the
        # *classification* logic needs the stop side.
        stop_dt = datetime.fromtimestamp(prev_t, tz=timezone.utc)
        gaps.append({
            "gap_start_t": gap_start_t,
            "gap_end_t": gap_end_t,
            "gap_start_utc": start_dt.isoformat(),
            "gap_end_utc": end_dt.isoformat(),
            "start_weekday": start_dt.weekday(),  # Monday = 0 .. Sunday = 6
            "end_weekday": end_dt.weekday(),
            "start_hour_utc": start_dt.hour,
            "stop_weekday": stop_dt.weekday(),
            "stop_hour_utc": stop_dt.hour,
            "missing_bars": missing,
            "duration_hours": round(missing * span / 3600.0, 2),
        })

    daily_close_hours = _daily_close_hours(gaps, min_recurrence=min_daily_recurrence)
    stats_at_hour = {
        hour: _pause_size_stats(gaps, hour, max_bars_for_typical=max_bars_for_typical_pause)
        for hour in daily_close_hours
    }

    # Pass 1: the weekly close. Keyed on `stop_weekday`/`stop_hour_utc` (the
    # last PRESENT candle), not `start_weekday` — a close landing exactly on
    # a midnight boundary (last trade Friday 23:30, first missing bar dated
    # Saturday 00:00) must still read as "stopped trading on Friday", or the
    # single most common real-world instance of this exact gap is missed.
    for g in gaps:
        checks = g.setdefault("rule_checks", {})
        is_friday_stop = g["stop_weekday"] == 4
        late_enough = g["stop_hour_utc"] >= _FRIDAY_CLOSE_EARLIEST_HOUR_UTC
        resumes_on_weekend = g["end_weekday"] in (5, 6, 0)
        checks["weekly_close"] = (
            f"last trade {_WEEKDAY_NAMES[g['stop_weekday']]} {g['stop_hour_utc']:02d}:xx UTC "
            f"({'Friday' if is_friday_stop else 'not Friday'}"
            f"{', late enough' if is_friday_stop and late_enough else ''}"
            f"{', too early in the day' if is_friday_stop and not late_enough else ''}), "
            f"resumes {_WEEKDAY_NAMES[g['end_weekday']]} "
            f"({'weekend/Monday' if resumes_on_weekend else 'not weekend/Monday'})"
        )
        if is_friday_stop and late_enough and resumes_on_weekend:
            g["category"] = "EXPECTED_MARKET_GAP"
            g["reason"] = "weekly_close"

    # Pass 2: known holidays — catches what the weekly-close rule does not
    # (a holiday landing on a weekday, or a closure that starts before
    # Friday, e.g. the Thursday evening ahead of Good Friday).
    for g in gaps:
        holiday_hit = _overlaps_known_holiday(g["gap_start_t"], g["gap_end_t"])
        g["rule_checks"]["known_holiday"] = (
            "overlaps an evidenced US market holiday" if holiday_hit
            else "no evidenced holiday in this date range")
        if "category" in g:
            continue
        if holiday_hit:
            g["category"] = "EXPECTED_MARKET_GAP"
            g["reason"] = "known_holiday"

    # Pass 3: the broker's routine daily maintenance/session pause.
    for g in gaps:
        hour = g["start_hour_utc"]
        if g["stop_weekday"] >= 5:
            g["rule_checks"]["broker_maintenance"] = "stopped trading on a weekend day"
        elif hour not in daily_close_hours:
            g["rule_checks"]["broker_maintenance"] = (
                f"{hour:02d}:00 UTC is not a recognized daily-close hour "
                f"(recognized: {sorted(daily_close_hours) or 'none established'})")
        else:
            stats = stats_at_hour.get(hour)
            if stats is None:
                g["rule_checks"]["broker_maintenance"] = (
                    f"{hour:02d}:00 UTC is recognized but has no typical size on record")
            else:
                median, mad = stats["median"], stats["mad"]
                band = max(pause_tolerance_mad_multiple * mad, 1.0)
                lo, hi = median - band, median + band
                in_band = lo <= g["missing_bars"] <= hi
                g["rule_checks"]["broker_maintenance"] = (
                    f"{hour:02d}:00 UTC typical size {median:.0f} bars (MAD {mad:.1f}), "
                    f"tolerance [{lo:.1f}, {hi:.1f}]; this gap is {g['missing_bars']} bars "
                    f"({'within' if in_band else 'outside'} tolerance)")
                if "category" not in g and in_band:
                    g["category"] = "EXPECTED_MARKET_GAP"
                    g["reason"] = "broker_maintenance"
                    g["detail"] = f"~{median:.0f} bars typical at {hour:02d}:00 UTC in this series"

    # Pass 4/5: what is left is either small enough to be plausible thin
    # liquidity, or genuinely unexplained.
    for g in gaps:
        if "category" in g:
            continue
        if g["missing_bars"] <= suspicious_max_missing_bars:
            g["category"] = "SUSPICIOUS_GAP"
            g["reason"] = "thin_liquidity_or_irregular_session"
        else:
            g["category"] = "DATA_ERROR"
            g["reason"] = "unexplained_missing_candles"

    counts = Counter(g["category"] for g in gaps)
    return {
        "gaps": gaps,
        "counts": dict(counts),
        "daily_close_hours_utc": sorted(daily_close_hours),
    }
