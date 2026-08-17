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

API compatibility
------------------
Built and verified against MetaTrader5 Python package 5.0.5735 / terminal
build 6116. That package does NOT expose ``symbol_info_session_trade`` (an
earlier version of this script assumed it did, which is wrong) — no
session-related API is hard-coded here. `detect_session_api` looks at
`dir(mt5)` at runtime instead, so this script neither crashes nor silently
misreports on a package that has no session accessor at all, and picks one
up automatically if a future package version adds one under some other
name. Nothing here invents an API the installed package does not have.

Initialization is explicit and self-contained (`mt5.initialize()` ->
`terminal_info()` -> `symbol_info()`/`symbol_select()` -> ... ->
`mt5.shutdown()` in a `finally`), matching how MT5 was actually verified to
work in this environment: `mt5.initialize()` alone, with no login call,
attaches to an already-running, already-authenticated terminal. This
deliberately does NOT go through `data.market.mt5_session.ensure_session()`
(which performs a full `mt5.login()` with `.env` credentials) — that is a
different, heavier connection path than what this read-only audit needs or
was verified against.

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

from analysis.features import timeframe_alignment as ta  # noqa: E402

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
    local-time conversion on top, and this environment's Windows host has
    its own local timezone. Always convert explicitly, never rely on
    default handling, and never let a candle timestamp be silently
    reinterpreted in local time anywhere in this script."""
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def number_of_grid_points(start: datetime, end: datetime, minutes: int) -> int:
    """Diagnostic reference only — how many bars a fully-populated grid
    would have in [start, end). NOT used for classification: a session can
    legitimately be closed for part or all of this range, so a shortfall
    against this number proves nothing by itself."""
    span = timedelta(minutes=minutes)
    if end <= start:
        return 0
    return int((end - start) // span)


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
    RuntimeError with a specific reason on any failure rather than letting a
    later call fail more confusingly."""
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
        "package_version": getattr(mt5, "__version__", None),
        "mt5_version": mt5.version(),
        "symbol": symbol,
        "symbol_visible": getattr(info, "visible", None),
        "symbol_description": getattr(info, "description", None),
        "last_error": mt5.last_error(),
    }


# ---------------------------------------------------------------------------
# MT5 data queries
# ---------------------------------------------------------------------------

def fetch_mt5_range(mt5, symbol: str, mt5_timeframe, start: datetime, end: datetime) -> list:
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, _naive_utc(start), _naive_utc(end))
    return [] if rates is None else list(rates)


def fetch_mt5_ticks(mt5, symbol: str, start: datetime, end: datetime) -> dict:
    """Returns {"ticks": [...], "status": "OK"|"ERROR"|"UNSUPPORTED", "error": ...}.

    A tick-query failure is reported as exactly that — ERROR, with
    mt5.last_error() attached — never silently reinterpreted as "no ticks
    exist" or "market closed". copy_ticks_range is present in the installed
    package, but this still checks with hasattr rather than assuming it,
    consistent with not hard-coding any MT5 API surface.
    """
    if not hasattr(mt5, "copy_ticks_range"):
        return {"ticks": [], "status": "UNSUPPORTED", "error": None}
    flag = getattr(mt5, "COPY_TICKS_ALL", None)
    try:
        if flag is None:
            ticks = mt5.copy_ticks_range(symbol, _naive_utc(start), _naive_utc(end), 0)
        else:
            ticks = mt5.copy_ticks_range(symbol, _naive_utc(start), _naive_utc(end), flag)
    except Exception as exc:  # pragma: no cover - depends on live terminal
        return {"ticks": [], "status": "ERROR", "error": str(exc)}
    if ticks is None:
        return {"ticks": [], "status": "ERROR", "error": str(mt5.last_error())}
    return {"ticks": list(ticks), "status": "OK", "error": None}


def detect_session_api(mt5) -> list:
    """Every attribute on the installed module whose name mentions
    'session', found at runtime rather than assumed. On MetaTrader5
    5.0.5735 this is an empty list — there is no session accessor at all,
    which is reported honestly rather than papered over."""
    return sorted(name for name in dir(mt5) if "session" in name.lower())


def query_session_evidence(mt5, symbol: str, start: datetime, end: datetime) -> dict:
    """Session evidence, or an explicit statement that none exists.

    Absence of a session API is reported as SESSION_METADATA_UNAVAILABLE —
    this status must never, by itself, be read as "market was closed" or
    feed a CALENDAR_RULE_MISMATCH conclusion on its own; it only means this
    particular source of evidence isn't available in this environment.

    If some session-named attribute DOES exist (a different package
    version, a broker-specific extension), it is called defensively and its
    raw return value is reported under "details" — but not interpreted as
    proof of anything, since this script does not know that function's
    actual semantics. Turning a raw value into "closed" requires a named,
    reviewed interpreter for that specific API, which does not exist here
    because the API itself does not exist in the installed package.
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
# Per-gap audit
# ---------------------------------------------------------------------------

def _boundary_bar(bars: list, *, closest_to: str):
    """The single bar in `bars` nearest the gap boundary — the last one
    chronologically if `closest_to == "start"` (i.e. immediately BEFORE the
    gap), the first one if `closest_to == "end"` (immediately AFTER)."""
    if not bars:
        return None
    ordered = sorted(bars, key=lambda r: float(r["time"]))
    return ordered[-1] if closest_to == "start" else ordered[0]


def _bar_summary(bar) -> str:
    if bar is None:
        return "(none)"
    t = datetime.fromtimestamp(float(bar["time"]), tz=timezone.utc).isoformat()
    try:
        return (f"{t}  O={bar['open']} H={bar['high']} L={bar['low']} C={bar['close']}")
    except (KeyError, IndexError, TypeError):
        return t


def audit_one_gap(mt5, symbol: str, start: datetime, end: datetime, historical: list) -> dict:
    window_start, window_end = start - MARGIN, end + MARGIN

    m30_window = fetch_mt5_range(mt5, symbol, mt5.TIMEFRAME_M30, window_start, window_end)
    m30_before_raw = [r for r in m30_window if float(r["time"]) < start.timestamp()]
    m30_after_raw = [r for r in m30_window if float(r["time"]) >= end.timestamp()]
    m30_in_gap = [r for r in m30_window if start.timestamp() <= float(r["time"]) < end.timestamp()]

    m1_window = fetch_mt5_range(mt5, symbol, mt5.TIMEFRAME_M1, window_start, window_end)
    m1_before_raw = [r for r in m1_window if float(r["time"]) < start.timestamp()]
    m1_after_raw = [r for r in m1_window if float(r["time"]) >= end.timestamp()]
    m1_in_gap = [r for r in m1_window if start.timestamp() <= float(r["time"]) < end.timestamp()]

    tick_result = fetch_mt5_ticks(mt5, symbol, window_start, window_end)
    ticks = tick_result["ticks"]

    def tick_time(t):
        # MT5 tick structs carry both `time` (seconds) and `time_msc`
        # (milliseconds); prefer the millisecond field when present for
        # precision, fall back to seconds otherwise.
        try:
            return float(t["time_msc"]) / 1000.0
        except (KeyError, IndexError, TypeError):
            return float(t["time"])

    ticks_before = [t for t in ticks if tick_time(t) < start.timestamp()]
    ticks_after = [t for t in ticks if tick_time(t) >= end.timestamp()]
    ticks_in_gap = [t for t in ticks if start.timestamp() <= tick_time(t) < end.timestamp()]

    hist_window = candles_in_window(historical, window_start, window_end)
    hist_in_gap = candles_in_window(historical, start, end)

    # Exact timestamp-SET comparison, not counts — over the whole audited
    # window for full diagnostic visibility, and separately scoped to just
    # the reported gap interval, since that narrower set is what actually
    # drives the classification below.
    mt5_ts_window = {round(float(r["time"])) for r in m30_window}
    hist_ts_window = {round(float(c["t"])) for c in hist_window}
    mt5_only = sorted(mt5_ts_window - hist_ts_window)
    historical_only = sorted(hist_ts_window - mt5_ts_window)
    matching = mt5_ts_window & hist_ts_window

    mt5_ts_in_gap = {round(float(r["time"])) for r in m30_in_gap}
    hist_ts_in_gap = {round(float(c["t"])) for c in hist_in_gap}
    mt5_only_in_gap = sorted(mt5_ts_in_gap - hist_ts_in_gap)
    historical_only_in_gap = sorted(hist_ts_in_gap - mt5_ts_in_gap)

    session = query_session_evidence(mt5, symbol, start, end)

    evidence = []
    conclusion, confidence = "UNKNOWN", "LOW"

    if mt5_only_in_gap:
        conclusion, confidence = "FETCH_PIPELINE_FAILURE", "HIGH"
        evidence.append(
            f"MT5 returns {len(mt5_only_in_gap)} M30 candle(s) inside the reported gap that "
            f"the historical file does not have: {[datetime.fromtimestamp(t, tz=timezone.utc).isoformat() for t in mt5_only_in_gap][:5]}"
            + (" ..." if len(mt5_only_in_gap) > 5 else ""))
    elif historical_only_in_gap:
        conclusion, confidence = "HISTORICAL_FILE_CORRUPTION", "MEDIUM"
        evidence.append(
            f"the historical file has {len(historical_only_in_gap)} M30 candle(s) inside the "
            f"reported gap that MT5 does not currently return — caveat: MT5's locally cached "
            f"history can itself be limited or have changed since export, so this is MEDIUM, "
            f"not HIGH, confidence")
    else:
        evidence.append(f"MT5 M30 bars inside the gap: {len(m30_in_gap)}")
        evidence.append(f"MT5 M1 bars inside the gap: {len(m1_in_gap)}")
        evidence.append(f"tick query status: {tick_result['status']}"
                         + (f" ({tick_result['error']})" if tick_result["error"] else ""))
        evidence.append(f"ticks inside the gap: {len(ticks_in_gap)}"
                         if tick_result["status"] == "OK" else "ticks inside the gap: n/a")
        evidence.append(f"session evidence status: {session['status']}")

        if session["confirms_closed"] is True:
            conclusion, confidence = "BROKER_SESSION_GAP", "HIGH"
            evidence.append("session metadata directly confirms the market was closed")
        elif tick_result["status"] == "OK" and len(ticks_before) > 0 and len(ticks_after) > 0 \
                and len(ticks_in_gap) == 0 and len(m1_in_gap) == 0 and len(m30_in_gap) == 0:
            conclusion, confidence = "CALENDAR_RULE_MISMATCH", "MEDIUM"
            evidence.append(
                f"MT5 has tick activity both before ({len(ticks_before)}) and after "
                f"({len(ticks_after)}) the gap, proving MT5's history coverage is not broken "
                f"in this region generally, yet zero ticks, M1 or M30 bars exist inside the "
                f"gap itself — a genuine data absence, but session metadata is unavailable "
                f"({session['status']}) so the specific cause (holiday, broker maintenance, "
                f"halt) is not established; MEDIUM confidence, not a proven session closure")
        else:
            reasons = []
            if tick_result["status"] != "OK":
                reasons.append(f"tick query did not succeed ({tick_result['status']})")
            if tick_result["status"] == "OK" and (len(ticks_before) == 0 or len(ticks_after) == 0):
                reasons.append("no corroborating tick activity immediately around the gap, "
                                "so a broader connectivity/history problem cannot be ruled out")
            if session["status"] == "SESSION_METADATA_UNAVAILABLE":
                reasons.append("no session metadata available in this MT5 package")
            evidence.append("insufficient evidence to reach any conclusion above HIGH/MEDIUM "
                             "confidence: " + "; ".join(reasons))

    return {
        "start": start, "end": end,
        "connection_symbol": symbol,
        "m30_bars_in_gap": len(m30_in_gap),
        "m30_bars_before": len(m30_before_raw),
        "m30_bars_after": len(m30_after_raw),
        "m30_first_before": _boundary_bar(m30_before_raw, closest_to="start"),
        "m30_first_after": _boundary_bar(m30_after_raw, closest_to="end"),
        "m1_bars_in_gap": len(m1_in_gap),
        "m1_bars_before": len(m1_before_raw),
        "m1_bars_after": len(m1_after_raw),
        "m1_first_before": _boundary_bar(m1_before_raw, closest_to="start"),
        "m1_first_after": _boundary_bar(m1_after_raw, closest_to="end"),
        "tick_status": tick_result["status"],
        "tick_error": tick_result["error"],
        "ticks_in_gap": len(ticks_in_gap),
        "ticks_before": len(ticks_before),
        "ticks_after": len(ticks_after),
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
          f"M1={result['expected_m1_grid_points']} (NOT used for classification — a closed "
          f"session legitimately has fewer)")

    print("\nMT5 connection:")
    print(f"  initialized: {connection['initialized']}")
    print(f"  terminal_connected: {connection['terminal_connected']}")
    print(f"  symbol: {connection['symbol']} (visible={connection['symbol_visible']})")
    print(f"  last_error: {connection['last_error']}")

    print("\nM30:")
    print(f"  bars_in_gap: {result['m30_bars_in_gap']}")
    print(f"  bars_before: {result['m30_bars_before']}")
    print(f"  bars_after:  {result['m30_bars_after']}")
    print(f"  first_before: {_bar_summary(result['m30_first_before'])}")
    print(f"  first_after:  {_bar_summary(result['m30_first_after'])}")

    print("\nM1:")
    print(f"  bars_in_gap: {result['m1_bars_in_gap']}")
    print(f"  bars_before: {result['m1_bars_before']}")
    print(f"  bars_after:  {result['m1_bars_after']}")
    print(f"  first_before: {_bar_summary(result['m1_first_before'])}")
    print(f"  first_after:  {_bar_summary(result['m1_first_after'])}")

    print("\nTICKS:")
    print(f"  status: {result['tick_status']}" + (f"  error={result['tick_error']}" if result['tick_error'] else ""))
    print(f"  ticks_in_gap: {result['ticks_in_gap']}")
    print(f"  ticks_before: {result['ticks_before']}")
    print(f"  ticks_after:  {result['ticks_after']}")

    print("\nHISTORICAL FILE:")
    print(f"  bars_in_gap: {result['hist_bars_in_gap']}")
    print(f"  matching_timestamps (whole 48h window): {result['matching_timestamps']}")
    print(f"  mt5_only (whole window): {len(result['mt5_only'])}"
          + (f"  in-gap: {len(result['mt5_only_in_gap'])}" if result['mt5_only_in_gap'] else ""))
    print(f"  historical_only (whole window): {len(result['historical_only'])}"
          + (f"  in-gap: {len(result['historical_only_in_gap'])}" if result['historical_only_in_gap'] else ""))

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
    print("No orders sent.\n")

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
          f"path={connection['terminal_path']}")
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
        header = (f"{'gap start':25s} {'M30':>5s} {'M1':>5s} {'ticks':>6s} {'hist':>5s} "
                  f"{'session':>26s}  {'conclusion':28s} confidence")
        print(header)
        print("-" * len(header))
        for r in results:
            ticks_display = str(r["ticks_in_gap"]) if r["tick_status"] == "OK" else r["tick_status"]
            print(f"{r['start'].isoformat():25s} {r['m30_bars_in_gap']:5d} {r['m1_bars_in_gap']:5d} "
                  f"{ticks_display:>6s} {r['hist_bars_in_gap']:5d} {r['session']['status']:>26s}  "
                  f"{r['conclusion']:28s} {r['confidence']}")

        counts = {c: 0 for c in CONCLUSIONS}
        for r in results:
            counts[r["conclusion"]] += 1
        print(f"\n{counts}")
        print("\nNo file was modified. No calendar rule was changed. This is evidence only.")
        return 0
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
