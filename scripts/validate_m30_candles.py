"""Gate 1 (Data Integrity) for an M30 candle file — nothing past that gate.

Scope, deliberately narrow: this checks one raw candle series for one
(symbol, timeframe) pair — provenance, timestamp grid, completed-candle
policy, field usability, and calendar-aware gap classification. It does NOT
build a dataset, does NOT label a trade, and does NOT touch H4 or H1.

Why a separate script from validate_real_dataset.py rather than an extension
of it: that script's dataset-build/leakage/label sections (5-9) are wired to
`train_entry_model.build_dataset`, which is H4-decision + H1-context by
construction — entry price is the next H4 bar's open, the horizon is counted
in H4 bars. None of that applies to an M30 series considered on its own, and
forcing M30 through it would mean either breaking that H4/H1 contract or
silently no-op'ing half the script. Sections 1-4 of that script (provenance,
timestamp integrity, completed-candle policy) ARE timeframe-agnostic, so this
script mirrors them rather than duplicating their logic blind: `diagnose_grid`
and the field-usability check are imported, not reimplemented.

Usage (Windows machine, where data/historical lives)::

    python scripts/validate_m30_candles.py
    python scripts/validate_m30_candles.py --symbols XAUUSD --candles data/historical

Exit code 0 = Gate 1 passes for every requested (symbol, M30) series.
Exit code 1 = something must be fixed or understood first. No training
follows from this script either way — it only reports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.features import microstructure_features as micro  # noqa: E402
from analysis.features import timeframe_alignment as ta  # noqa: E402

import train_entry_model as trainer  # noqa: E402

TIMEFRAME = "M30"

failures: list = []
warnings: list = []


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def fail(message: str) -> None:
    failures.append(message)
    print(f"  FAIL  {message}")


def warn(message: str) -> None:
    warnings.append(message)
    print(f"  WARN  {message}")


def ok(message: str) -> None:
    print(f"  ok    {message}")


def utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def check_provenance(directory: str, symbols) -> dict:
    section("1. PROVENANCE")
    result = trainer.check_provenance(directory)
    if not result["real"]:
        fail(result["reason"])
        return {}

    manifest = result["manifest"]
    ok(f"manifest.json present, fetched_at = {manifest.get('fetched_at')}")

    declared = manifest.get("files", {})
    expected = {f"{s}_{TIMEFRAME}" for s in symbols}
    missing = expected - set(declared)
    if missing:
        fail(f"manifest does not declare: {sorted(missing)} — "
             f"run scripts/fetch_training_candles.py before this")

    for key in sorted(expected & set(declared)):
        entry = declared[key]
        path = os.path.join(directory, f"{key}.json")
        if not os.path.exists(path):
            fail(f"{key}: manifest lists {entry.get('path')} but it is missing")
            continue
        on_disk = len(json.load(open(path, encoding="utf-8")))
        if on_disk != entry.get("bars"):
            fail(f"{key}: manifest says {entry.get('bars')} bars, file holds {on_disk} "
                 f"— the file changed after it was fetched")
        else:
            ok(f"{key}: {on_disk} bars, {entry.get('first')} -> {entry.get('last')}")

    return manifest


def check_grid_and_structure(series, symbol: str) -> None:
    section("2. TIMESTAMP INTEGRITY")
    name = f"{symbol}_{TIMEFRAME}"

    issues = ta.validate_series(series, TIMEFRAME)
    for key in ("unsorted", "duplicate_timestamps", "bad_ohlc", "non_finite"):
        if issues[key]:
            fail(f"{name}: {issues[key]} {key}")
    if issues["misaligned_to_grid"]:
        fail(f"{name}: {issues['misaligned_to_grid']} candles not on the {TIMEFRAME} grid")

    grid = ta.diagnose_grid(series, TIMEFRAME)
    if grid["on_utc_grid"]:
        ok(f"{name}: on the UTC {TIMEFRAME} grid, {issues['gaps']} raw gaps "
           f"(largest {issues['largest_gap_bars']} bars)")
    elif grid["constant_offset"]:
        warn(f"{name}: every bar is offset {grid['modal_offset_hours']}h from the UTC "
             f"{TIMEFRAME} grid — broker server time. classify_gaps() below reasons about "
             f"weekday only, never an exact hour, so this offset does not skew it, but any "
             f"future session-hour feature built from these timestamps must correct for it.")
    else:
        fail(f"{name}: {grid['distinct_offsets']} different grid offsets (modal "
             f"{grid['modal_offset_hours']}h covers only {grid['modal_share']:.1%}) — "
             f"this series is not a clean single timeframe")


def check_completed(series, symbol: str, manifest: dict) -> None:
    section("3. COMPLETED CANDLES ONLY")
    name = f"{symbol}_{TIMEFRAME}"

    # Prefer this file's own fetch timestamp over the batch-level one: a
    # multi-symbol, multi-timeframe run can take long enough that comparing
    # against when the WHOLE run started, rather than when this particular
    # file was actually pulled, produces a false "still forming" verdict on
    # a candle that legitimately closed between the two.
    entry = manifest.get("files", {}).get(f"{symbol}_{TIMEFRAME}", {})
    fetched_at_raw = entry.get("fetched_at") or manifest.get("fetched_at")
    fetched_at = None
    if fetched_at_raw:
        try:
            fetched_at = datetime.fromisoformat(fetched_at_raw).timestamp()
        except ValueError:
            warn(f"cannot parse fetched_at: {fetched_at_raw!r}")

    now = datetime.now(timezone.utc).timestamp()
    newest = float(series[-1]["t"])
    closes_at = ta.close_time(newest, TIMEFRAME)

    if closes_at > now:
        fail(f"{name}: newest bar opens {utc(newest)} and does not close until "
             f"{utc(closes_at)} — a forming candle reached the export")
        return

    if fetched_at is not None and closes_at > fetched_at:
        fail(f"{name}: newest bar closes {utc(closes_at)}, after the fetch at "
             f"{utc(fetched_at)} — it was still forming when exported")
    else:
        ok(f"{name}: newest bar closed {utc(closes_at)}, before the fetch")

    dropped = entry.get("forming_candles_dropped")
    if dropped is not None:
        ok(f"{name}: fetch_training_candles.py dropped {dropped} forming candle(s) "
           f"at export time")


def check_fields(series, symbol: str) -> None:
    section("4. FIELD USABILITY (spread / real_volume / tick_volume)")
    name = f"{symbol}_{TIMEFRAME}"
    availability = micro.field_availability(series)
    for field in ("spread", "real_volume", "volume"):
        status = availability.get(field, "NOT AVAILABLE")
        label = "tick_volume" if field == "volume" else field
        if field == "real_volume" and status.startswith("AVAILABLE (constant"):
            # Explicitly not a failure and not even a warning: most FX/CFD
            # brokers do not report true traded volume, so an all-zero
            # real_volume is the expected reading, not a defect. Reported
            # here so it stays visible without being treated as an error.
            ok(f"{name}: {label} = {status} — unavailable from this broker, not a defect")
        elif status == "NOT AVAILABLE":
            warn(f"{name}: {label} = NOT AVAILABLE — this file predates spread/real_volume "
                 f"capture in fetch_training_candles.py, or the broker never sent it")
        else:
            ok(f"{name}: {label} = {status}")


def check_gaps(series, symbol: str) -> None:
    section("5. GAP CLASSIFICATION")
    name = f"{symbol}_{TIMEFRAME}"

    result = ta.classify_gaps(series, TIMEFRAME)
    gaps = result["gaps"]
    counts = result["counts"]

    total_span_days = (float(series[-1]["t"]) - float(series[0]["t"])) / 86400.0
    print(f"  {name}: {len(gaps)} total gaps over {total_span_days:.0f} days "
          f"({len(series)} bars)")
    hours = result.get("daily_close_hours_utc") or []
    print(f"  recognized daily-close hour(s) UTC: "
          f"{[f'{h:02d}:00' for h in hours] or '(none established — series too short/sparse)'}")
    for category in ("EXPECTED_MARKET_GAP", "SUSPICIOUS_GAP", "DATA_ERROR"):
        print(f"    {category:20s} {counts.get(category, 0)}")

    reason_counts = Counter((g["category"], g["reason"]) for g in gaps)
    if reason_counts:
        print("\n  by reason:")
        for (category, reason), n in sorted(reason_counts.items()):
            print(f"    {category:20s} reason={reason:32s} {n}")

    error_gaps = [g for g in gaps if g["category"] == "DATA_ERROR"]
    if error_gaps:
        fail(f"{name}: {len(error_gaps)} DATA_ERROR gaps — genuinely unexplained, "
             f"listing every one below for audit:")
        weekday_names = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
        for g in error_gaps:
            print(f"    {g['gap_start_utc']} ({weekday_names[g['start_weekday']]}) -> "
                  f"{g['gap_end_utc']} ({weekday_names[g['end_weekday']]})  "
                  f"missing={g['missing_bars']} bars  {g['duration_hours']:.1f}h  "
                  f"reason={g['reason']}")
    else:
        ok(f"{name}: 0 DATA_ERROR gaps — every gap explained by the weekly close, a known "
           f"holiday, the broker's own recurring daily pause, or small enough to be "
           f"plausible thin liquidity")

    suspicious = counts.get("SUSPICIOUS_GAP", 0)
    if suspicious:
        share = suspicious / max(len(gaps), 1)
        (warn if share <= 0.5 else fail)(
            f"{name}: {suspicious} SUSPICIOUS_GAP ({share:.1%} of all gaps) — not blocking "
            f"unless this share is unreasonably high")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", default=os.path.join("data", "historical"))
    parser.add_argument("--symbols", nargs="*", default=["XAUUSD"])
    args = parser.parse_args()

    print(f"Validating {args.candles} for {args.symbols} {TIMEFRAME} — Gate 1 (data "
          f"integrity) only. No dataset is built, nothing is trained, nothing is written.")

    manifest = check_provenance(args.candles, args.symbols)
    if failures:
        print("\nStopping: the data cannot be identified.")
        return 1

    for symbol in args.symbols:
        try:
            series = trainer.load_candles(symbol, TIMEFRAME, args.candles)
        except FileNotFoundError as exc:
            fail(str(exc))
            continue

        check_grid_and_structure(series, symbol)
        check_completed(series, symbol, manifest)
        check_fields(series, symbol)
        check_gaps(series, symbol)

    section("VERDICT")
    if failures:
        print(f"  GATE 1 FAILED — {len(failures)} blocking problem(s):\n")
        for item in failures:
            print(f"    - {item}")
        if warnings:
            print(f"\n  plus {len(warnings)} warning(s):")
            for item in warnings:
                print(f"    - {item}")
        print("\n  Do not proceed to Gate 2. Fix these first.")
        return 1

    print("  GATE 1 PASSED")
    if warnings:
        print(f"\n  {len(warnings)} warning(s) — not blocking, but read them:")
        for item in warnings:
            print(f"    - {item}")
    print("\n  Gate 1 (Data Integrity) is clean. Gate 2 (Leakage Detection) is next, and is "
          "not run by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
