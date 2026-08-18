"""Forensic audit of specific XAUUSD M30 gaps against the live MT5 source.

Run on Windows, with MT5 running and logged in. This is READ-ONLY: it opens
no positions, sends no orders, writes no historical file, and does not
touch data/historical/XAUUSD_M30.json or any calendar rule. Its only output
is a report.

Why this exists
----------------
scripts/validate_m30_candles.py can only classify a gap from what is
already in the exported file plus a calendar model. It cannot tell the
difference between "the broker genuinely has no candles here" and "the
broker has candles here that the exporter failed to fetch" — that requires
asking MT5 directly. This script asks, and reports exactly what it found,
distinguishing observed evidence from inference at every step.

Two rounds of real runs against a live terminal (package 5.0.5735, build
6116) found two real bugs in this script, both fixed here:

1. It called the ``symbol_info_session_trade`` attribute directly, which
   does not exist on that package. Fixed by `detect_session_api` looking
   at `dir(mt5)` at runtime instead of assuming any specific name.
2. Its M1 query for old historical windows (2024/2025) returned a bar
   timestamped MONTHS after the requested range (e.g. 2026-05-01 for a
   2024-01-15 request) — a clear sign the terminal does not have M1 history
   that far back and `copy_rates_range` did not bound its result to the
   requested window the way this script assumed. Trusting that as
   "M1 = 0 bars" would have been actively wrong: it is not evidence of
   absence, it is evidence the query itself is unreliable for this window.
   Fixed by validating every returned timestamp against the requested
   range (`_validate_range`) for M30, M1 AND ticks — anything outside the
   window is excluded from classification and the query is flagged
   ``QUERY_RANGE_VIOLATION``; and by reporting ``M1_HISTORY_UNAVAILABLE``
   explicitly (not a silent "0") whenever the terminal has no valid M1
   coverage anywhere in the 48h audit window, as distinct from a genuine,
   corroborated absence inside just the gap.

API compatibility
------------------
Nothing here assumes an MT5 API beyond what `dir(mt5)` / `hasattr` confirm
is actually present on the installed package. No specific session function
name is hard-coded anywhere.

Initialization is explicit and self-contained (`mt5.initialize()` ->
`terminal_info()` -> `symbol_info()`/`symbol_select()` -> ... ->
`mt5.shutdown()` in a `finally`), matching how MT5 was actually verified to
work in this environment: `mt5.initialize()` alone, with no login call,
attaches to an already-running, already-authenticated terminal.

Usage::

    python scripts/audit_xauusd_m30_gaps.py
    python scripts/audit_xauusd_m30_gaps.py --symbol XAUUSD --candles data/historical
    python scripts/audit_xauusd_m30_gaps.py --start "2025-07-03T03:30:00+00:00" \
        --end "2025-07-03T12:00:00+00:00"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.features import timeframe_alignment as ta  # noqa: E402  (re-exported for reuse)

# The five gaps from the real Gate 1 report this tool was built to
# investigate. Each is (start, end) as reported — the missing window, not
# the surrounding present candles.
REPORTED_GAPS = [
    (datetime(2024, 1, 15, 22, 0, tzinfo=timezone.utc), datetime(2024, 1, 16, 1, 0, tzinfo=timezone.utc)),
    (datetime(2025, 1, 7, 4, 30, tzinfo=timezone.utc), datetime(2025, 1, 7, 7, 0, tzinfo=timezone.utc)),
    (datetime(2025, 7, 3, 3, 30, tzinfo=timezone.utc), datetime(2025, 7, 3, 12, 0, tzinfo=timezone.utc)),
    (datetime(2025, 10, 23, 20, 30, tzinfo=timezone.utc), datetime(2025, 10, 23, 22, 0, tzinfo=timezone.utc)),
    (datetime(2026, 2, 26, 22, 0, tzinfo=timezone.utc), datetime(2026, 2, 27, 3, 0, tzinfo=timezone.utc)),
]

MARGIN = timedelta(hours=24)
CONCLUSIONS = ("BROKER_SESSION_GAP", "FETCH_PIPELINE_FAILURE",
               "HISTORICAL_FILE_CORRUPTION", "CALENDAR_RULE_MISMATCH", "UNKNOWN")


def _naive_utc(dt: datetime) -> datetime:
    """MT5's copy_rates_range/copy_ticks_range want a plain datetime they
    treat as UTC — handing them a tz-aware one risks the library applying a
    local-time conversion on top. Always convert explicitly."""
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def number_of_grid_points(start: datetime, end: datetime, minutes: int) -> int:
    """Diagnostic reference only — how many bars a fully-populated grid
    would have in [start, end). NOT used for classification: a session can
    legitimately be closed for part or all of this range."""
    span = timedelta(minutes=minutes)
    if end <= start:
        return 0
    return int((end - start) // span)


def tick_epoch(tick) -> float:
    """MT5 tick structs carry both `time` (seconds) and `time_msc`
    (milliseconds); prefer the millisecond field when present for
    precision, fall back to seconds otherwise."""
    try:
        return float(tick["time_msc"]) / 1000.0
    except (KeyError, IndexError, TypeError):
        return float(tick["time"])


# ---------------------------------------------------------------------------
# Historical file (read-only)
# ---------------------------------------------------------------------------

def load_historical(symbol: str, directory: str) -> list:
    path = os.path.join(directory, f"{symbol}_M30.json")
    if not os.path.exists(path):
        print(f"  (no historical file at {path})")
        return []
    with open(path, encoding="utf-8") as fh:
        candles = json.load(fh)
    candles.sort(key=lambda c: c["t"])
    return candles


def candles_in_window(candles: list, start: datetime, end: datetime) -> list:
    lo, hi = start.timestamp(), end.timestamp()
    return [c for c in candles if lo <= float(c["t"]) < hi]


# ---------------------------------------------------------------------------
# MT5 connection — explicit, self-contained, read-only
# ---------------------------------------------------------------------------

def connect_mt5(mt5, symbol: str) -> dict:
    """Initialize MT5, verify the terminal and the symbol. Raises
    RuntimeError with a specific reason on any failure."""
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

    terminal = mt5.terminal_info()
    if terminal is None:
        raise RuntimeError(f"mt5.terminal_info() returned None: {mt5.last_error()}")

    info = mt5.symbol_info(symbol)
    if info is None:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(
                f"symbol {symbol} unavailable and symbol_select() failed: {mt5.last_error()}")
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"{symbol} still unavailable after symbol_select(): "
                                f"{mt5.last_error()}")

    return {
        "initialized": True,
        "terminal_connected": getattr(terminal, "connected", None),
        "terminal_company": getattr(terminal, "company", None),
        "terminal_path": getattr(terminal, "path", None),
        "terminal_maxbars": getattr(terminal, "maxbars", None),
        "package_version": getattr(mt5, "__version__", None),
        "mt5_version": mt5.version(),
        "symbol": symbol,
        "symbol_visible": getattr(info, "visible", None),
        "symbol_description": getattr(info, "description", None),
        "last_error": mt5.last_error(),
    }


# ---------------------------------------------------------------------------
# Range-validated MT5 data queries
# ---------------------------------------------------------------------------

def _validate_range(rows: list, window_start: datetime, window_end: datetime,
                     epoch_of) -> dict:
    """Every MT5 query in this script is checked against this: did the
    returned data actually stay inside [window_start, window_end)? A round
    of real testing found `copy_rates_range` returning a bar timestamped
    months after the requested historical window — silently trusting that
    as "0 bars in range" would have been wrong in the dangerous direction
    (mistaking an unreliable query for a confirmed absence).

    Returns {"valid": [...], "violation": bool, "returned_first": float|None,
    "returned_last": float|None, "raw_count": int}. `valid` contains only
    rows whose timestamp actually falls in the window — callers must use
    `valid`, never the raw MT5 response, for anything downstream.
    """
    if not rows:
        return {"valid": [], "violation": False, "returned_first": None,
                "returned_last": None, "raw_count": 0}
    epochs = sorted(epoch_of(r) for r in rows)
    returned_first, returned_last = epochs[0], epochs[-1]
    lo, hi = window_start.timestamp(), window_end.timestamp()
    violation = returned_first < lo or returned_last >= hi
    valid = [r for r in rows if lo <= epoch_of(r) < hi]
    return {"valid": valid, "violation": violation, "returned_first": returned_first,
            "returned_last": returned_last, "raw_count": len(rows)}


def fetch_mt5_range(mt5, symbol: str, mt5_timeframe, start: datetime, end: datetime) -> dict:
    """Range-validated M30/M1 query. Returns a `_validate_range` result plus
    `last_error` captured immediately after the call, per-query, not just
    once at connection time."""
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, _naive_utc(start), _naive_utc(end))
    error = mt5.last_error()
    rows = [] if rates is None else list(rates)
    result = _validate_range(rows, start, end, lambda r: float(r["time"]))
    result["last_error"] = error
    return result


def fetch_mt5_ticks(mt5, symbol: str, start: datetime, end: datetime) -> dict:
    """Range-validated tick query. Returns {"valid", "violation",
    "returned_first", "returned_last", "raw_count", "status", "error",
    "last_error"}. `status` is "OK", "ERROR", or "UNSUPPORTED" — a failed or
    absent query is never reinterpreted as "zero ticks exist"."""
    if not hasattr(mt5, "copy_ticks_range"):
        return {"valid": [], "violation": False, "returned_first": None, "returned_last": None,
                "raw_count": 0, "status": "UNSUPPORTED", "error": None, "last_error": None}
    flag = getattr(mt5, "COPY_TICKS_ALL", None)
    try:
        if flag is None:
            ticks = mt5.copy_ticks_range(symbol, _naive_utc(start), _naive_utc(end), 0)
        else:
            ticks = mt5.copy_ticks_range(symbol, _naive_utc(start), _naive_utc(end), flag)
        error = mt5.last_error()
    except Exception as exc:  # pragma: no cover - depends on live terminal
        return {"valid": [], "violation": False, "returned_first": None, "returned_last": None,
                "raw_count": 0, "status": "ERROR", "error": str(exc), "last_error": None}
    if ticks is None:
        return {"valid": [], "violation": False, "returned_first": None, "returned_last": None,
                "raw_count": 0, "status": "ERROR", "error": str(error), "last_error": error}
    rows = list(ticks)
    result = _validate_range(rows, start, end, tick_epoch)
    result["status"] = "OK"
    result["error"] = None
    result["last_error"] = error
    return result


def detect_session_api(mt5) -> list:
    """Every attribute on the installed module whose name mentions
    'session', found at runtime rather than assumed. On MetaTrader5
    5.0.5735 this is an empty list — there is no session accessor at all,
    which is reported honestly rather than papered over."""
    return sorted(name for name in dir(mt5) if "session" in name.lower())


def query_session_evidence(mt5, symbol: str, start: datetime, end: datetime) -> dict:
    """Session evidence, or an explicit statement that none exists.

    Absence of a session API is reported as SESSION_METADATA_UNAVAILABLE —
    this must never, by itself, be read as "market was closed" or feed a
    CALENDAR_RULE_MISMATCH conclusion on its own.

    If some session-named attribute DOES exist, it is called defensively
    and its raw return value is reported under "details" but never
    interpreted as proof of anything — this script has no reviewed
    interpreter for an API it has never seen documented behavior for.
    """
    candidates = detect_session_api(mt5)
    if not candidates:
        return {
            "status": "SESSION_METADATA_UNAVAILABLE",
            "source": None,
            "details": (
                "no attribute containing 'session' found on the installed MetaTrader5 "
                f"module (mt5.version()={mt5.version()}) — symbol_info_session_trade, "
                "in particular, does not exist in this package"
            ),
            "confirms_closed": None,
        }

    raw = {}
    day = start.date()
    while day <= end.date():
        for name in candidates:
            fn = getattr(mt5, name, None)
            if not callable(fn):
                raw[f"{day}:{name}"] = f"(not callable: {fn!r})"
                continue
            mt5_day_of_week = (day.weekday() + 1) % 7  # Python Mon=0 -> MT5 Sun=0
            try:
                raw[f"{day}:{name}"] = fn(symbol, mt5_day_of_week, 0)
            except Exception as exc:
                raw[f"{day}:{name}"] = f"ERROR: {exc}"
        day += timedelta(days=1)

    return {
        "status": "SESSION_METADATA_QUERIED",
        "source": candidates,
        "details": raw,
        # Deliberately None, not True/False: no interpreter for an unnamed,
        # never-before-seen API is written here. See docstring above.
        "confirms_closed": None,
    }


# ---------------------------------------------------------------------------
# In-memory-only tick -> synthetic 1-minute OHLC (diagnostic, never persisted)
# ---------------------------------------------------------------------------

def synthetic_ohlc_from_ticks(ticks: list, *, minute_span: int = 1) -> list:
    """Aggregate ticks into `minute_span`-minute OHLC bars, IN MEMORY ONLY,
    purely to answer "was there real price activity during the gap?" —
    never written to disk, never compared against or substituted into the
    historical file, never fed to classification, and never confused with
    broker-provided M1/M30 data. Every bar this returns must be labelled
    SYNTHETIC_FROM_TICKS by the caller; nothing here does that labelling
    implicitly, so a caller cannot forget it.
    """
    if not ticks:
        return []
    span = minute_span * 60.0
    buckets: dict = {}
    for t in ticks:
        try:
            price = float(t["bid"])
        except (KeyError, IndexError, TypeError, ValueError):
            try:
                price = float(t["last"])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
        if price <= 0:
            continue
        epoch = tick_epoch(t)
        bucket = int(epoch // span) * span
        bar = buckets.setdefault(bucket, {"open": price, "high": price, "low": price,
                                           "close": price, "tick_count": 0})
        bar["high"] = max(bar["high"], price)
        bar["low"] = min(bar["low"], price)
        bar["close"] = price
        bar["tick_count"] += 1
    return [
        {"t": bucket, "label": "SYNTHETIC_FROM_TICKS", **bar}
        for bucket, bar in sorted(buckets.items())
    ]


# ---------------------------------------------------------------------------
# Per-gap audit
# ---------------------------------------------------------------------------

def _boundary_bar(bars: list, *, closest_to: str):
    """The single bar in `bars` nearest the gap boundary."""
    if not bars:
        return None
    ordered = sorted(bars, key=lambda r: float(r["time"]))
    return ordered[-1] if closest_to == "start" else ordered[0]


def _bar_summary(bar) -> str:
    if bar is None:
        return "(none)"
    t = datetime.fromtimestamp(float(bar["time"]), tz=timezone.utc).isoformat()
    try:
        return f"{t}  O={bar['open']} H={bar['high']} L={bar['low']} C={bar['close']}"
    except (KeyError, IndexError, TypeError):
        return t


def _iso(epoch):
    return None if epoch is None else datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def audit_one_gap(mt5, symbol: str, start: datetime, end: datetime, historical: list) -> dict:
    window_start, window_end = start - MARGIN, end + MARGIN
    gap_seconds = (end - start).total_seconds()

    # ---- M30 --------------------------------------------------------
    m30 = fetch_mt5_range(mt5, symbol, mt5.TIMEFRAME_M30, window_start, window_end)
    m30_valid = m30["valid"]
    m30_before = [r for r in m30_valid if float(r["time"]) < start.timestamp()]
    m30_after = [r for r in m30_valid if float(r["time"]) >= end.timestamp()]
    m30_in_gap = [r for r in m30_valid if start.timestamp() <= float(r["time"]) < end.timestamp()]

    # ---- M1 ---------------------------------------------------------
    m1 = fetch_mt5_range(mt5, symbol, mt5.TIMEFRAME_M1, window_start, window_end)
    m1_valid = m1["valid"]
    m1_before = [r for r in m1_valid if float(r["time"]) < start.timestamp()]
    m1_after = [r for r in m1_valid if float(r["time"]) >= end.timestamp()]
    m1_in_gap = [r for r in m1_valid if start.timestamp() <= float(r["time"]) < end.timestamp()]
    # Genuine absence requires the terminal to actually have SOME valid M1
    # coverage somewhere in the 48h window; zero anywhere (or a range
    # violation on the raw response) means the query itself is unreliable
    # for this period, not that the market was quiet.
    m1_status = "M1_HISTORY_UNAVAILABLE" if (m1["violation"] or not m1_valid) else "OK"

    # ---- Ticks --------------------------------------------------------
    ticks = fetch_mt5_ticks(mt5, symbol, window_start, window_end)
    ticks_valid = ticks["valid"]
    ticks_before = [t for t in ticks_valid if tick_epoch(t) < start.timestamp()]
    ticks_after = [t for t in ticks_valid if tick_epoch(t) >= end.timestamp()]
    ticks_in_gap = sorted(
        [t for t in ticks_valid if start.timestamp() <= tick_epoch(t) < end.timestamp()],
        key=tick_epoch)

    tick_diagnostics = None
    if ticks_in_gap:
        first_t, last_t = tick_epoch(ticks_in_gap[0]), tick_epoch(ticks_in_gap[-1])
        hours = max((last_t - first_t) / 3600.0, 1e-9)
        tick_diagnostics = {
            "first_tick": first_t, "last_tick": last_t,
            "count": len(ticks_in_gap),
            "ticks_per_hour": len(ticks_in_gap) / hours if len(ticks_in_gap) > 1 else None,
            "offset_from_gap_start_s": first_t - start.timestamp(),
            "offset_from_gap_end_s": end.timestamp() - last_t,
            "sample_first_3": [_iso(tick_epoch(t)) for t in ticks_in_gap[:3]],
            "sample_last_3": [_iso(tick_epoch(t)) for t in ticks_in_gap[-3:]],
        }

    tick_activity_without_m30 = len(ticks_in_gap) > 0 and len(m30_in_gap) == 0

    synthetic = (synthetic_ohlc_from_ticks(ticks_in_gap) if tick_activity_without_m30 else [])

    # ---- Historical file comparison (exact timestamp SETS) -----------
    hist_window = candles_in_window(historical, window_start, window_end)
    hist_in_gap = candles_in_window(historical, start, end)

    mt5_ts_window = {round(float(r["time"])) for r in m30_valid}
    hist_ts_window = {round(float(c["t"])) for c in hist_window}
    mt5_only = sorted(mt5_ts_window - hist_ts_window)
    historical_only = sorted(hist_ts_window - mt5_ts_window)
    matching = mt5_ts_window & hist_ts_window

    mt5_ts_in_gap = {round(float(r["time"])) for r in m30_in_gap}
    hist_ts_in_gap = {round(float(c["t"])) for c in hist_in_gap}
    mt5_only_in_gap = sorted(mt5_ts_in_gap - hist_ts_in_gap)
    historical_only_in_gap = sorted(hist_ts_in_gap - mt5_ts_in_gap)

    session = query_session_evidence(mt5, symbol, start, end)

    # ---- Classification (conservative, evidence-gated) ----------------
    evidence = []
    conclusion, confidence = "UNKNOWN", "LOW"

    for label, q in (("M30", m30), ("M1", m1), ("ticks", ticks)):
        if q.get("violation"):
            evidence.append(
                f"{label}: QUERY_RANGE_VIOLATION — requested [{window_start.isoformat()}, "
                f"{window_end.isoformat()}), MT5 returned data spanning "
                f"[{_iso(q['returned_first'])}, {_iso(q['returned_last'])}]; "
                f"out-of-range rows excluded from every count below")
    if m1_status == "M1_HISTORY_UNAVAILABLE":
        evidence.append("M1: M1_HISTORY_UNAVAILABLE — no validly-in-range M1 bar anywhere in "
                         "the 48h window, so M1 provides no usable evidence for this gap "
                         "(NOT reported as M1=0)")
    if tick_activity_without_m30:
        evidence.append(
            f"TICK_ACTIVITY_WITHOUT_M30: {len(ticks_in_gap)} tick(s) fall inside the reported "
            f"gap while MT5 returns zero M30 bars there — evidence of market-data activity, "
            f"NOT proof of a fetch-pipeline failure (MT5 tick history can extend further back "
            f"than its generated M30 series) and NOT proof the market was open in the sense "
            f"session metadata would confirm")

    if mt5_only_in_gap:
        conclusion, confidence = "FETCH_PIPELINE_FAILURE", "HIGH"
        evidence.append(
            f"MT5 returns {len(mt5_only_in_gap)} M30 candle(s) inside the reported gap that "
            f"the historical file does not have: "
            f"{[_iso(t) for t in mt5_only_in_gap][:5]}" + (" ..." if len(mt5_only_in_gap) > 5 else ""))
    elif historical_only_in_gap:
        conclusion, confidence = "HISTORICAL_FILE_CORRUPTION", "MEDIUM"
        evidence.append(
            f"the historical file has {len(historical_only_in_gap)} M30 candle(s) inside the "
            f"reported gap that MT5 does not currently return — caveat: MT5's locally cached "
            f"history can itself be limited or have changed since export, so this is MEDIUM, "
            f"not HIGH, confidence")
    else:
        evidence.append(f"MT5 M30 bars inside the gap (range-valid): {len(m30_in_gap)}")
        evidence.append(f"MT5 M1 bars inside the gap (range-valid): {len(m1_in_gap)} "
                         f"[{m1_status}]")
        evidence.append(f"tick query status: {ticks['status']}"
                         + (f" ({ticks['error']})" if ticks["error"] else ""))
        evidence.append(f"ticks inside the gap (range-valid): {len(ticks_in_gap)}"
                         if ticks["status"] == "OK" else "ticks inside the gap: n/a")
        evidence.append(f"session evidence status: {session['status']}")

        if session["confirms_closed"] is True:
            conclusion, confidence = "BROKER_SESSION_GAP", "HIGH"
            evidence.append("session metadata directly confirms the market was closed")
        elif tick_activity_without_m30:
            # Explicit per the brief: this evidence state must NEVER be
            # auto-resolved to BROKER_SESSION_GAP or FETCH_PIPELINE_FAILURE.
            # It stays UNKNOWN unless stronger (session) evidence exists.
            conclusion, confidence = "UNKNOWN", "MEDIUM"
            evidence.append(
                "conclusion withheld: tick activity without M30 is a real, evidenced "
                "contradiction, not something this script may resolve into a specific root "
                "cause without session confirmation")
        elif (ticks["status"] == "OK" and len(ticks_before) > 0 and len(ticks_after) > 0
                and len(ticks_in_gap) == 0 and len(m1_in_gap) == 0 and len(m30_in_gap) == 0
                and m1_status == "OK"):
            conclusion, confidence = "CALENDAR_RULE_MISMATCH", "MEDIUM"
            evidence.append(
                f"MT5 has tick activity both before ({len(ticks_before)}) and after "
                f"({len(ticks_after)}) the gap, and validated M1 coverage elsewhere in the "
                f"window, proving MT5's history is not broken in this region generally, yet "
                f"zero ticks, M1 or M30 bars exist inside the gap itself — a genuine data "
                f"absence, but session metadata is unavailable ({session['status']}) so the "
                f"specific cause is not established; MEDIUM confidence")
        else:
            reasons = []
            if ticks["status"] != "OK":
                reasons.append(f"tick query did not succeed ({ticks['status']})")
            elif len(ticks_before) == 0 or len(ticks_after) == 0:
                reasons.append("no corroborating tick activity immediately around the gap")
            if m1_status != "OK":
                reasons.append("M1 history unavailable for this window")
            if session["status"] == "SESSION_METADATA_UNAVAILABLE":
                reasons.append("no session metadata available in this MT5 package")
            evidence.append("insufficient evidence to reach any conclusion above LOW "
                             "confidence: " + "; ".join(reasons))

    return {
        "start": start, "end": end,
        "connection_symbol": symbol,
        "m30_query": m30, "m1_query": m1, "tick_query": ticks,
        "m1_status": m1_status,
        "m30_bars_in_gap": len(m30_in_gap),
        "m30_bars_before": len(m30_before),
        "m30_bars_after": len(m30_after),
        "m30_first_before": _boundary_bar(m30_before, closest_to="start"),
        "m30_first_after": _boundary_bar(m30_after, closest_to="end"),
        "m1_bars_in_gap": len(m1_in_gap),
        "m1_bars_before": len(m1_before),
        "m1_bars_after": len(m1_after),
        "m1_first_before": _boundary_bar(m1_before, closest_to="start"),
        "m1_first_after": _boundary_bar(m1_after, closest_to="end"),
        "tick_status": ticks["status"],
        "tick_error": ticks["error"],
        "ticks_in_gap": len(ticks_in_gap),
        "ticks_before": len(ticks_before),
        "ticks_after": len(ticks_after),
        "tick_diagnostics": tick_diagnostics,
        "tick_activity_without_m30": tick_activity_without_m30,
        "synthetic_from_ticks": synthetic,
        "hist_bars_in_gap": len(hist_in_gap),
        "matching_timestamps": len(matching),
        "mt5_only": mt5_only,
        "historical_only": historical_only,
        "mt5_only_in_gap": mt5_only_in_gap,
        "historical_only_in_gap": historical_only_in_gap,
        "expected_m30_grid_points": number_of_grid_points(start, end, 30),
        "expected_m1_grid_points": number_of_grid_points(start, end, 1),
        "session": session,
        "evidence": evidence,
        "conclusion": conclusion,
        "confidence": confidence,
    }


def print_report(index: int, result: dict, connection: dict) -> None:
    print("=" * 60)
    print(f"GAP #{index}")
    print("=" * 60)
    print(f"\nGap:")
    print(f"  start: {result['start'].isoformat()}")
    print(f"  end:   {result['end'].isoformat()}")
    print(f"  duration: {(result['end'] - result['start'])}")
    print(f"  reference-only expected grid points: M30={result['expected_m30_grid_points']} "
          f"M1={result['expected_m1_grid_points']} (NOT used for classification)")

    print("\nMT5 connection:")
    print(f"  initialized: {connection['initialized']}")
    print(f"  terminal_connected: {connection['terminal_connected']}")
    print(f"  terminal_maxbars: {connection.get('terminal_maxbars')}")
    print(f"  symbol: {connection['symbol']} (visible={connection['symbol_visible']})")
    print(f"  last_error: {connection['last_error']}")

    for label, key_bars, key_before, key_after, key_first_before, key_first_after, query in (
        ("M30", "m30_bars_in_gap", "m30_bars_before", "m30_bars_after",
         "m30_first_before", "m30_first_after", result["m30_query"]),
        ("M1", "m1_bars_in_gap", "m1_bars_before", "m1_bars_after",
         "m1_first_before", "m1_first_after", result["m1_query"]),
    ):
        print(f"\n{label}:")
        print(f"  requested_start: {result['start'].isoformat() if label == 'M30' else result['start'].isoformat()}")
        print(f"  requested_end:   {result['end'].isoformat()}")
        print(f"  returned_first (raw, pre-validation): {_iso(query['returned_first'])}")
        print(f"  returned_last  (raw, pre-validation): {_iso(query['returned_last'])}")
        print(f"  QUERY_RANGE_VIOLATION: {query['violation']}")
        print(f"  last_error: {query.get('last_error')}")
        if label == "M1":
            print(f"  status: {result['m1_status']}")
        print(f"  bars_in_gap: {result[key_bars]}")
        print(f"  bars_before: {result[key_before]}")
        print(f"  bars_after:  {result[key_after]}")
        print(f"  first_before: {_bar_summary(result[key_first_before])}")
        print(f"  first_after:  {_bar_summary(result[key_first_after])}")

    print("\nTICKS:")
    tq = result["tick_query"]
    print(f"  status: {result['tick_status']}" + (f"  error={result['tick_error']}" if result['tick_error'] else ""))
    print(f"  returned_first (raw): {_iso(tq['returned_first'])}")
    print(f"  returned_last  (raw): {_iso(tq['returned_last'])}")
    print(f"  QUERY_RANGE_VIOLATION: {tq['violation']}")
    print(f"  ticks_in_gap: {result['ticks_in_gap']}")
    print(f"  ticks_before: {result['ticks_before']}")
    print(f"  ticks_after:  {result['ticks_after']}")
    if result["tick_diagnostics"]:
        d = result["tick_diagnostics"]
        print(f"  first_tick_in_gap: {_iso(d['first_tick'])}")
        print(f"  last_tick_in_gap:  {_iso(d['last_tick'])}")
        print(f"  ticks_per_hour: {d['ticks_per_hour']}")
        print(f"  offset_from_gap_start: {d['offset_from_gap_start_s']:.1f}s")
        print(f"  offset_from_gap_end:   {d['offset_from_gap_end_s']:.1f}s")
        print(f"  sample first 3: {d['sample_first_3']}")
        print(f"  sample last 3:  {d['sample_last_3']}")
    if result["tick_activity_without_m30"]:
        print("  TICK_ACTIVITY_WITHOUT_M30: True (evidence state, not a conclusion)")
        if result["synthetic_from_ticks"]:
            n = len(result["synthetic_from_ticks"])
            first, last = result["synthetic_from_ticks"][0], result["synthetic_from_ticks"][-1]
            print(f"  SYNTHETIC_FROM_TICKS (in-memory diagnostic only, never written to disk, "
                  f"never used as data): {n} synthetic 1-min bar(s), "
                  f"{_iso(first['t'])} .. {_iso(last['t'])}, "
                  f"price range [{min(b['low'] for b in result['synthetic_from_ticks']):.3f}, "
                  f"{max(b['high'] for b in result['synthetic_from_ticks']):.3f}]")

    print("\nHISTORICAL FILE:")
    print(f"  bars_in_gap: {result['hist_bars_in_gap']}")
    print(f"  matching_timestamps (whole 48h window): {result['matching_timestamps']}")
    print(f"  mt5_only (whole window, {len(result['mt5_only'])}): "
          f"{[_iso(t) for t in result['mt5_only'][:10]]}" + (" ..." if len(result['mt5_only']) > 10 else ""))
    print(f"  historical_only (whole window, {len(result['historical_only'])}): "
          f"{[_iso(t) for t in result['historical_only'][:10]]}" + (" ..." if len(result['historical_only']) > 10 else ""))
    print(f"  mt5_only_in_gap: {[_iso(t) for t in result['mt5_only_in_gap']]}")
    print(f"  historical_only_in_gap: {[_iso(t) for t in result['historical_only_in_gap']]}")

    print("\nSESSION EVIDENCE:")
    print(f"  status: {result['session']['status']}")
    print(f"  source: {result['session']['source']}")
    if result['session']['status'] != "SESSION_METADATA_UNAVAILABLE":
        print(f"  details: {result['session']['details']}")

    print("\nEVIDENCE:")
    for line in result["evidence"]:
        print(f"  - {line}")

    print(f"\nCONCLUSION:\n  {result['conclusion']}")
    print(f"\nCONFIDENCE:\n  {result['confidence']}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--candles", default=os.path.join("data", "historical"))
    parser.add_argument("--start", help="ISO8601 UTC; audits one gap instead of the reported five")
    parser.add_argument("--end", help="ISO8601 UTC; required with --start")
    args = parser.parse_args()

    print("READ-ONLY AUDIT")
    print("No historical file modified.")
    print("No calendar rule modified.")
    print("No dataset built.")
    print("No model trained.")
    print("No orders sent.")
    print("Any tick-derived aggregation is in-memory diagnostic only (SYNTHETIC_FROM_TICKS) "
          "and is never written to disk or treated as broker-provided data.\n")

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("ERROR: MetaTrader5 is not installed. Run this on the Windows machine "
              "with the MT5 terminal running and logged in.")
        return 1

    try:
        connection = connect_mt5(mt5, args.symbol)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        try:
            mt5.shutdown()
        except Exception:
            pass
        return 1

    print(f"Connected: package={connection['package_version']} "
          f"terminal={connection['mt5_version']} company={connection['terminal_company']} "
          f"path={connection['terminal_path']} maxbars={connection.get('terminal_maxbars')}")
    print(f"Symbol: {connection['symbol']} — {connection['symbol_description']} "
          f"(visible={connection['symbol_visible']})\n")

    try:
        if args.start:
            if not args.end:
                print("ERROR: --end is required with --start")
                return 1
            gaps = [(datetime.fromisoformat(args.start), datetime.fromisoformat(args.end))]
        else:
            gaps = REPORTED_GAPS

        historical = load_historical(args.symbol, args.candles)
        if historical:
            print(f"Loaded {len(historical)} historical {args.symbol} M30 bars from "
                  f"{args.candles} for comparison.\n")

        results = []
        for i, (start, end) in enumerate(gaps, 1):
            result = audit_one_gap(mt5, args.symbol, start, end, historical)
            print_report(i, result, connection)
            results.append(result)

        print("=" * 60)
        print("SUMMARY TABLE")
        print("=" * 60)
        header = (f"{'gap start':25s} {'M30':>5s} {'M1':>18s} {'ticks':>8s} {'hist':>5s} "
                  f"{'session':>26s}  {'conclusion':28s} confidence")
        print(header)
        print("-" * len(header))
        for r in results:
            ticks_display = str(r["ticks_in_gap"]) if r["tick_status"] == "OK" else r["tick_status"]
            m1_display = f"{r['m1_bars_in_gap']} [{r['m1_status']}]"
            print(f"{r['start'].isoformat():25s} {r['m30_bars_in_gap']:5d} {m1_display:>18s} "
                  f"{ticks_display:>8s} {r['hist_bars_in_gap']:5d} {r['session']['status']:>26s}  "
                  f"{r['conclusion']:28s} {r['confidence']}")

        counts = {c: 0 for c in CONCLUSIONS}
        for r in results:
            counts[r["conclusion"]] += 1
        print(f"\n{counts}")
        tick_without_m30 = sum(1 for r in results if r["tick_activity_without_m30"])
        print(f"TICK_ACTIVITY_WITHOUT_M30 (evidence state, not a conclusion): "
              f"{tick_without_m30}/{len(results)} gap(s)")
        print("\nNo file was modified. No calendar rule was changed. This is evidence only.")
        return 0
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
