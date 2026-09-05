"""Fetch raw OHLC candles from MT5 for entry-model training. Run on Windows.

Why this exists
---------------
Training the entry model requires the *same* indicator values the live path
computes. The live path derives them from raw candles with its own arithmetic
(simple-average RSI, SMA-based MACD — see data/market/mt5_client.get_indicators),
which differs materially from the standard Wilder/EMA formulas used to build the
older entry_v2 parquet datasets. Measured on a realistic series:

    RSI   |live - wilder|   mean 6.6 points, p90 13.2   (thresholds sit at 30/45/55/70)
    MACD  live/EMA scale    3.5x, sign agreement 86%
    ATR   live/wilder       ratio 1.00  (equivalent in practice)

Training on the parquet's values would therefore hand the model a feature
distribution it never sees in production. This script captures the raw candles
so indicators can be recomputed with the live formulas instead.

It also captures enough history to label **both** BUY and SELL trades. The
existing parquet has no `direction` column, so every row in it was labelled as a
BUY — a model trained on that cannot tell the two apart.

Usage (on the Windows machine with MT5 running and logged in)::

    python scripts/fetch_training_candles.py
    python scripts/fetch_training_candles.py --years 3 --symbols EURUSD XAUUSD

Writes ``data/historical/<SYMBOL>_<TF>.json`` — OHLC plus ``spread`` and
``real_volume``, no indicators, no labels. Read-only: it opens no positions
and modifies no bot state.

Each row also carries ``spread`` and ``real_volume`` — fields MT5's rate
struct returns alongside OHLC that earlier exports discarded. Both are
properties of the closed bar itself, known at the same instant as its
open/high/low/close, so capturing them adds no look-ahead risk beyond what
this script already carries. ``real_volume`` is frequently 0 for FX/CFD
symbols, since most brokers do not report true traded volume for
over-the-counter instruments — this is reported, not assumed, by
``analysis/features/microstructure_features.py``.

Files fetched before this field was added do not have these keys.
``load_candles`` in ``scripts/train_entry_model.py`` and the microstructure
feature builder both treat their absence as "not available", not as zero.

Only completed candles: ``fetch()`` filters by actual close time against real
UTC "now" (see ``_closed_by``), not by dropping a fixed array position. A run
whose manifest's batch-level ``fetched_at`` was captured well before a
particular file's own MT5 round trip actually happened is exactly the
scenario a position-based drop cannot defend against — the filter is
timestamp-based specifically so it stays correct regardless of how long the
batch takes or what MT5 hands back. Each manifest file entry also carries its
own ``fetched_at`` and ``forming_candles_dropped`` for that reason: a
completed-candle check must compare a file's newest candle against *that
file's* pull time, not the run's start time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.features import timeframe_alignment as ta  # noqa: E402
from analysis.data import phase1  # noqa: E402
from config import SYMBOLS  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("fetch_training_candles")

OUT_DIR = os.path.join("data", "historical")

# H4 drives labelling for the established research track (the horizon is
# counted in H4 bars, matching how the live trade-age layer counts). H1
# supplies rsi/macd exactly as the live snapshot does. Neither is changed by
# the M30 addition below.
#
# M30 is fetched for an INDEPENDENT research track. It does not feed, alter or
# reinterpret the H4/H1 track, and no existing result derived from H4/H1
# changes because of it. It is fetched for whatever symbols are requested, so
# `--symbols XAUUSD` keeps the download to the one instrument under study.
TIMEFRAMES = ("H4", "H1", "M30")

# Bars per year, following the convention already used here: bars-per-session
# x sessions-per-year, rounded up so holidays and gaps cannot truncate the
# request. The existing entries are H4 = 6 bars/day and H1 = 24 bars/day over
# ~267 sessions; M30 is 48 bars/day on the same basis, giving ~12,800. M15 is
# 96 bars/day -> ~25,600. NOTE: five years of M15 (~128,000 bars) exceeds what
# copy_rates_from_pos accepts, so deep M15 fetches go through fetch_range()
# (--start), which is bounded by the terminal's own history depth instead of
# this estimate.
_BARS_PER_YEAR = {"H4": 1600, "H1": 6400, "M30": 12800, "M15": 25600}


def count_for(timeframe: str, years: float) -> int:
    """Bars to request for `years` of history on `timeframe` (see _BARS_PER_YEAR)."""
    return int(_BARS_PER_YEAR[timeframe] * years)


def _mt5_timeframe(mt5, tf: str):
    return {
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
    }[tf]


def _closed_by(row: dict, timeframe: str, now: float) -> bool:
    """Has this bar's window actually elapsed, by wall-clock UTC?

    Requesting one extra bar and dropping array position -1 assumes MT5
    always returns exactly one still-forming bar in a known array slot.
    That held until it didn't: a run whose manifest claimed `fetched_at`
    hours before the file's own newest candle had actually closed proved the
    assumption isn't safe to rely on blindly — whatever the exact cause (a
    slow multi-symbol run, a retry, an MT5 ordering edge case), the fix is to
    stop trusting position and check the thing that actually matters: has
    `open_time + span` passed real UTC "now" or not. That question answers
    itself correctly regardless of how many rows MT5 handed back or in what
    order.
    """
    return float(row["time"]) + ta.duration(timeframe) <= now


def _rows_from_rates(rates, timeframe: str, now: float) -> tuple:
    """Filter MT5's rate struct to completed candles and build the stored rows.

    Shared by fetch() and fetch_range() so both write the exact same row
    schema and apply the exact same completion test. Returns
    ``(rows, forming_dropped)`` with rows sorted oldest-first.
    """
    closed = [r for r in rates if _closed_by(r, timeframe, now)]
    rows = [
        {
            "t": float(r["time"]),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r["tick_volume"]),
            "spread": float(r["spread"]),
            "real_volume": float(r["real_volume"]),
        }
        for r in closed
    ]
    rows.sort(key=lambda c: c["t"])
    return rows, len(rates) - len(closed)


def fetch_range(symbol: str, timeframe: str, start_dt: datetime,
                end_dt: datetime = None) -> dict:
    """Completed candles whose open falls in [start_dt, end_dt], via
    mt5.copy_rates_range.

    Why a second fetch path when fetch() exists: fetch() is count-based from
    the newest bar, and the count argument has a hard ceiling — a five-year
    M15 request (~128,000 bars) is rejected outright with
    ``Terminal: Invalid params``. Range-based fetching is bounded instead by
    the terminal's own history depth for that timeframe, which is exactly the
    quantity Phase 1 needs to measure and report honestly.

    When the terminal answers ``Invalid params`` — its history does not reach
    back to the requested start — the earliest usable start is found by binary
    search (the failure is monotonic in the start date: too old always fails)
    and the run is flagged ``depth_limited``. Nothing is invented to fill the
    shortfall; the caller records what the terminal actually has.

    Returns ``{"candles", "forming_dropped", "fetched_at", "requested_start",
    "actual_start", "depth_limited"}``.
    """
    import MetaTrader5 as mt5  # imported here so the module loads off-Windows

    from data.market.mt5_session import ensure_symbol, mt5_call

    if not ensure_symbol(symbol):
        raise RuntimeError(f"symbol {symbol} not available in Market Watch")

    end = end_dt or datetime.now(timezone.utc)
    requested_start = start_dt

    def _try_range(lo: datetime):
        with mt5_call():
            return mt5.copy_rates_range(
                symbol, _mt5_timeframe(mt5, timeframe), lo, end)

    rates = _try_range(start_dt)
    depth_limited = False
    if rates is None or len(rates) == 0:
        # Monotone failure: start predates the terminal's depth for this
        # timeframe. Binary search the earliest start the terminal answers.
        # Sanity bound: if the terminal has nothing within TARGET_YEARS of
        # `end`, no start date will make it answer — fail honestly instead
        # of walking forever.
        lo = start_dt                                    # known-failing
        hi = end - timedelta(days=1)                     # assumed-working
        probe = _try_range(hi)
        if probe is None or len(probe) == 0:
            raise RuntimeError(
                f"no candles for {symbol} {timeframe} even from "
                f"{hi.date()} (last_error={mt5.last_error()})")
        depth_limited = True
        while (hi - lo) > timedelta(days=1):
            mid = lo + (hi - lo) / 2
            probe = _try_range(mid)
            if probe is None or len(probe) == 0:
                lo = mid
            else:
                hi = mid
        rates = _try_range(hi)
        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"copy_rates_range stopped answering for {symbol} {timeframe} "
                f"at start={hi.isoformat()}: {mt5.last_error()}")

    fetched_at = datetime.now(timezone.utc)
    rows, forming_dropped = _rows_from_rates(rates, timeframe, fetched_at.timestamp())
    if not rows:
        raise RuntimeError(
            f"no completed candles for {symbol} {timeframe} in "
            f"[{requested_start.date()}, {end.date()}]: {mt5.last_error()}")

    return {
        "candles": rows,
        "forming_dropped": forming_dropped,
        "fetched_at": fetched_at.isoformat(),
        "requested_start": requested_start.isoformat(),
        "actual_start": datetime.fromtimestamp(
            rows[0]["t"], tz=timezone.utc).isoformat(),
        "depth_limited": depth_limited,
    }


def fetch(symbol: str, timeframe: str, count: int) -> dict:
    """Completed candles only, newest last.

    Returns ``{"candles": [...], "forming_dropped": N, "fetched_at": iso}``
    — the drop count and the real fetch timestamp go into the manifest per
    file (see main()) rather than only once for the whole run, so a later
    completed-candle check compares against when THIS file was actually
    pulled, not when the batch started.
    """
    import MetaTrader5 as mt5  # imported here so the module loads off-Windows

    from data.market.mt5_session import ensure_symbol, mt5_call

    if not ensure_symbol(symbol):
        raise RuntimeError(f"symbol {symbol} not available in Market Watch")

    # Ask for a few extra bars: the filter below decides what is actually
    # closed, so over-requesting costs nothing and guards against needing
    # more than the traditional "exactly one forming bar" assumption.
    with mt5_call():
        rates = mt5.copy_rates_from_pos(symbol, _mt5_timeframe(mt5, timeframe), 0, count + 3)
    fetched_at = datetime.now(timezone.utc)
    now = fetched_at.timestamp()

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"no candles returned for {symbol} {timeframe}: {mt5.last_error()}")

    # MT5's rate struct also carries `spread` and `real_volume` per bar — both
    # properties of the closed bar, known at the same instant as OHLC, so
    # capturing them adds no look-ahead risk (see _rows_from_rates).
    rows, forming_dropped = _rows_from_rates(rates, timeframe, now)
    if len(rows) > count:
        rows = rows[-count:]
    return {"candles": rows, "forming_dropped": forming_dropped,
            "fetched_at": fetched_at.isoformat()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=list(SYMBOLS))
    parser.add_argument("--years", type=float, default=3.0,
                        help="how much history to request per timeframe (default 3). "
                             "Count-based fetch from the newest bar; large M15 "
                             "requests hit the terminal's count ceiling -- use "
                             "--start for deep history instead.")
    parser.add_argument("--timeframes", nargs="*", default=None,
                        help="timeframes to fetch (default: %s)" % (TIMEFRAMES,))
    parser.add_argument("--start", default=None,
                        help="ISO date (UTC) to fetch FROM, e.g. 2021-09-05 -- "
                             "range-based deep fetch via copy_rates_range. When "
                             "the terminal's history does not reach this start, "
                             "the earliest usable start is found by binary search "
                             "and the shortfall is recorded (depth_limited), never "
                             "filled with synthetic candles.")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    timeframes = tuple(args.timeframes) if args.timeframes else TIMEFRAMES
    start_dt = None
    if args.start:
        try:
            start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"ERROR: --start must be an ISO date like 2021-09-05, got {args.start!r}")
            return 1

    try:
        import MetaTrader5  # noqa: F401
    except ImportError:
        print("ERROR: MetaTrader5 is not installed. Run this on the Windows machine.")
        return 1

    from data.market.mt5_session import ensure_session

    if not ensure_session():
        print("ERROR: could not establish an MT5 session. Check .env and that the "
              "terminal is running and logged in.")
        return 1

    os.makedirs(args.out, exist_ok=True)
    batch_started_at = datetime.now(timezone.utc).isoformat()

    # Merge into any existing manifest instead of replacing it: a partial run
    # (one symbol failing, an interrupted batch) must not erase the provenance
    # of files already on disk. Entries this run fetches are replaced wholesale;
    # everything else is kept exactly as it was. Deterministic (phase1.merge_manifest).
    manifest_path = os.path.join(args.out, "manifest.json")
    existing_manifest = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                existing_manifest = json.load(fh)
        except (OSError, ValueError) as exc:
            logger.warning("could not read existing manifest (%s); starting a fresh one", exc)

    manifest_files: dict = {}
    failures = []

    for symbol in args.symbols:
        for tf in timeframes:
            try:
                result = (fetch_range(symbol, tf, start_dt)
                          if start_dt is not None else fetch(symbol, tf, count_for(tf, args.years)))
            except Exception as exc:
                logger.error("fetch failed for %s %s: %s", symbol, tf, exc)
                failures.append(f"{symbol} {tf}: {exc}")
                continue

            candles = result["candles"]
            path = os.path.join(args.out, f"{symbol}_{tf}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(candles, fh)

            first = datetime.fromtimestamp(candles[0]["t"], tz=timezone.utc).date()
            last = datetime.fromtimestamp(candles[-1]["t"], tz=timezone.utc).date()
            last_close = datetime.fromtimestamp(
                ta.close_time(candles[-1]["t"], tf), tz=timezone.utc)
            entry = {
                "path": path,
                "symbol": symbol,
                "timeframe": tf,
                "bars": len(candles),
                "row_count": len(candles),
                "first": str(first),
                "last": str(last),
                # Provenance the Phase 1 contract requires: where the data came
                # from and in what shape, so no consumer has to guess.
                "source": "mt5",
                "schema_version": phase1.DATA_SCHEMA_VERSION,
                "timezone": "UTC",
                # This file's own pull, not the batch start — the check that
                # a candle wasn't still forming when exported must compare
                # against the moment THIS file was actually fetched.
                "fetched_at": result["fetched_at"],
                "forming_candles_dropped": result["forming_dropped"],
                "last_completed_candle_close_utc": last_close.isoformat(),
            }
            if start_dt is not None:
                entry["requested_start"] = result["requested_start"]
                entry["actual_start"] = result["actual_start"]
                entry["depth_limited"] = result["depth_limited"]
            manifest_files[f"{symbol}_{tf}"] = entry
            depth_note = ""
            if start_dt is not None and result["depth_limited"]:
                depth_note = "  [DEPTH-LIMITED by terminal history]"
            print(f"  {symbol:8s} {tf:3s}  {len(candles):6d} bars  {first} -> {last}"
                  f"  (dropped {result['forming_dropped']} forming){depth_note}")

    manifest = phase1.merge_manifest(existing_manifest, manifest_files,
                                     batch_started_at)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        return 1

    print(f"\nDone. Wrote {len(manifest['files'])} files to {args.out}/")
    print("Next: python scripts/train_entry_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
