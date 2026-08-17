"""Forensic audit of specific XAUUSD M30 gaps against the live MT5 source.

Run on Windows, with MT5 running and logged in. This is read-only: it opens
no positions, writes no historical file, and does not touch
data/historical/XAUUSD_M30.json. Its only output is a report.

Why this exists
----------------
scripts/validate_m30_candles.py can only classify a gap from what is already
in the exported file plus a calendar model. It cannot tell the difference
between "the broker genuinely has no candles here" (an evidence-based
EXPECTED_MARKET_GAP) and "the broker has candles here that the exporter
failed to fetch" (a FETCH_PIPELINE_FAILURE) — that distinction requires
asking MT5 directly, which only a machine with the terminal installed can
do. This script asks.

For each gap it queries, independently:

  * M30 candles from MT5 over [start - 24h, end + 24h]
  * M1 candles from MT5 over the same window, since M1 presence during an
    M30 hole distinguishes "no ticks at all" from "M30 aggregation itself
    is what's missing"
  * the symbol's published trade session (mt5.symbol_info_session_trade),
    if the broker exposes one, for every day the gap touches
  * the exported historical file over the same window, for comparison

and prints a report plus one conclusion per gap:

  BROKER_SESSION_GAP        MT5 itself has no candles, and/or a published
                             session says the market was closed
  FETCH_PIPELINE_FAILURE    MT5 has candles here that the exported file
                             does not — the exporter, not the market, is
                             responsible
  HISTORICAL_FILE_CORRUPTION the file's candle count/timestamps in this
                             window do not match MT5's own raw bars
  CALENDAR_RULE_MISMATCH    MT5 has no candles and no session data, but
                             surrounding evidence does not support a fetch
                             failure either — a legitimate closure the
                             calendar model does not have a rule for yet
  UNKNOWN                   the evidence does not cleanly support any of
                             the above; needs a human look

None of this changes classify_gaps() or any calendar rule. Per the brief:
only proven, evidence-based conclusions may become a narrow new rule, added
separately and only after this report exists.

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


def _naive_utc(dt: datetime) -> datetime:
    """MT5's copy_rates_range wants a plain datetime it treats as UTC —
    handing it a tz-aware one risks the library applying a local-time
    conversion on top. Always convert explicitly, never rely on default."""
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


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


def fetch_mt5_range(mt5, symbol: str, mt5_timeframe, start: datetime, end: datetime):
    from data.market.mt5_session import mt5_call

    with mt5_call():
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, _naive_utc(start), _naive_utc(end))
    return [] if rates is None else list(rates)


def session_info(mt5, symbol: str, day: datetime) -> list:
    """Every published trade-session segment for `day`'s weekday, or an
    empty list if the broker exposes none / the call is unsupported.

    MT5's day_of_week is Sunday=0..Saturday=6; Python's date.weekday() is
    Monday=0..Sunday=6. Converted explicitly rather than assumed equal —
    that mismatch would silently query the wrong day.
    """
    mt5_dow = (day.weekday() + 1) % 7
    segments = []
    for session_index in range(4):  # a handful of sub-sessions is generous
        try:
            result = mt5.symbol_info_session_trade(symbol, mt5_dow, session_index)
        except Exception as exc:  # pragma: no cover - depends on live terminal
            if session_index == 0:
                segments.append(("ERROR", str(exc)))
            break
        if result is None:
            break
        segments.append(result)
    return segments


def audit_one_gap(mt5, symbol: str, start: datetime, end: datetime, historical: list) -> dict:
    window_start, window_end = start - MARGIN, end + MARGIN

    m30_window = fetch_mt5_range(mt5, symbol, mt5.TIMEFRAME_M30, window_start, window_end)
    m30_in_gap = [r for r in m30_window if start.timestamp() <= float(r["time"]) < end.timestamp()]

    m1_window = fetch_mt5_range(mt5, symbol, mt5.TIMEFRAME_M1, window_start, window_end)
    m1_in_gap = [r for r in m1_window if start.timestamp() <= float(r["time"]) < end.timestamp()]

    hist_in_gap = candles_in_window(historical, start, end)
    hist_before = candles_in_window(historical, start - MARGIN, start)
    hist_after = candles_in_window(historical, end, end + MARGIN)

    sessions = []
    day = start.date()
    while day <= end.date():
        segs = session_info(mt5, symbol, datetime(day.year, day.month, day.day))
        sessions.append((str(day), segs))
        day += timedelta(days=1)
    market_closed_by_session = bool(sessions) and all(not segs for _, segs in sessions)

    # Decision. Order matters: a mismatch between MT5's own raw bars and the
    # historical file is the most actionable finding, so it is checked first
    # — it is never ambiguous with a broker closure, because if MT5 has bars
    # here, the market was plainly not closed.
    if len(m30_in_gap) > 0 and len(hist_in_gap) < len(m30_in_gap):
        conclusion = "FETCH_PIPELINE_FAILURE"
    elif len(m30_in_gap) == 0 and len(hist_in_gap) > 0:
        # The file has candles MT5 no longer serves for this window — only
        # plausible if the file was hand-edited or corrupted after export.
        conclusion = "HISTORICAL_FILE_CORRUPTION"
    elif len(m30_in_gap) == 0 and len(m1_in_gap) == 0 and market_closed_by_session:
        conclusion = "BROKER_SESSION_GAP"
    elif len(m30_in_gap) == 0 and len(m1_in_gap) == 0:
        # No candles at any resolution, but no published session confirms a
        # closure either (or the broker does not expose session data at
        # all) — a real absence with no calendar rule to explain it yet.
        conclusion = "CALENDAR_RULE_MISMATCH"
    else:
        conclusion = "UNKNOWN"

    return {
        "start": start, "end": end,
        "m30_before": m30_window[:1] and m30_window[0], "m30_after": m30_window[-1:] and m30_window[-1],
        "m30_bars_in_gap": len(m30_in_gap),
        "m1_bars_in_gap": len(m1_in_gap),
        "sessions": sessions,
        "market_closed_by_session": market_closed_by_session,
        "hist_bars_in_gap": len(hist_in_gap),
        "hist_bars_before": len(hist_before),
        "hist_bars_after": len(hist_after),
        "conclusion": conclusion,
    }


def print_report(result: dict) -> None:
    print("=" * 60)
    print("XAUUSD M30 GAP AUDIT")
    print("=" * 60)
    print(f"\nGap:\n{result['start'].isoformat()} -> {result['end'].isoformat()}\n")

    print("M30 (live MT5):")
    print(f"  bars returned by MT5 during gap: {result['m30_bars_in_gap']}")

    print("\nM1 (live MT5):")
    print(f"  bars during gap: {result['m1_bars_in_gap']}")

    print("\nMT5 session (symbol_info_session_trade):")
    if not result["sessions"]:
        print("  no session data queried")
    for day, segs in result["sessions"]:
        if not segs:
            print(f"  {day}: no published trading session (closed, or broker does not expose one)")
        else:
            for seg in segs:
                print(f"  {day}: session segment {seg}")
    print(f"  market closed (inferred): {'YES' if result['market_closed_by_session'] else 'NO / UNKNOWN'}")

    print("\nHistorical file:")
    print(f"  bars during gap: {result['hist_bars_in_gap']}")
    print(f"  bars in the 24h before: {result['hist_bars_before']}")
    print(f"  bars in the 24h after: {result['hist_bars_after']}")

    print(f"\nConclusion:\n  {result['conclusion']}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--candles", default=os.path.join("data", "historical"))
    parser.add_argument("--start", help="ISO8601 UTC; audits one gap instead of the reported five")
    parser.add_argument("--end", help="ISO8601 UTC; required with --start")
    args = parser.parse_args()

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("ERROR: MetaTrader5 is not installed. Run this on the Windows machine "
              "with the MT5 terminal running and logged in.")
        return 1

    from data.market.mt5_session import ensure_session, ensure_symbol

    if not ensure_session():
        print("ERROR: could not establish an MT5 session. Check .env and that the "
              "terminal is running and logged in.")
        return 1
    if not ensure_symbol(args.symbol):
        print(f"ERROR: {args.symbol} is not available in Market Watch.")
        return 1

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
    for start, end in gaps:
        result = audit_one_gap(mt5, args.symbol, start, end, historical)
        print_report(result)
        results.append(result)

    print("=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    header = f"{'gap start':25s} {'MT5 M30':>8s} {'MT5 M1':>8s} {'file':>6s} {'closed?':>8s}  conclusion"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['start'].isoformat():25s} {r['m30_bars_in_gap']:8d} {r['m1_bars_in_gap']:8d} "
              f"{r['hist_bars_in_gap']:6d} {('YES' if r['market_closed_by_session'] else 'NO'):>8s}  "
              f"{r['conclusion']}")

    counts: dict = {}
    for r in results:
        counts[r["conclusion"]] = counts.get(r["conclusion"], 0) + 1
    print(f"\n{dict(counts)}")
    print("\nNo file was modified. No calendar rule was changed. This is evidence only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
