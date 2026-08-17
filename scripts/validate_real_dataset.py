"""Phase 4 gate: does the real broker data survive every integrity check?

Trains nothing, writes no model, touches no production file. It answers one
question — may this data be used to train — and refuses to answer it vaguely.

Everything in PHASE3_REPORT.md was verified on synthetic candles. Synthetic
candles prove the code is correct; they say nothing about whether a real broker
feed is clean. The checks most likely to fire here are the ones synthetic data
cannot reproduce:

  * a broker serving candles in server time rather than UTC, which shifts every
    session label by the offset while leaving the alignment arithmetic intact,
  * a forming candle that reached the export,
  * H1 history that does not span the H4 history, silently dropping rows,
  * real weekend and holiday gaps.

Usage (Windows machine, where data/historical lives)::

    python scripts/validate_real_dataset.py
    python scripts/validate_real_dataset.py --candles data/historical --horizon 24

Exit code 0 = the gate passes and baseline training may proceed.
Exit code 1 = something must be fixed or understood first.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.features import live_parity_features as lpf  # noqa: E402
from analysis.features import timeframe_alignment as ta  # noqa: E402
from analysis.models import entry_feature_spec as spec  # noqa: E402

import train_entry_model as trainer  # noqa: E402

TIMEFRAMES = ("H4", "H1")

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


# ---------------------------------------------------------------------------
# 1. Provenance
# ---------------------------------------------------------------------------

def check_provenance(directory: str, symbols) -> dict:
    section("1. PROVENANCE")

    result = trainer.check_provenance(directory)
    if not result["real"]:
        fail(result["reason"])
        return {}

    manifest = result["manifest"]
    ok(f"manifest.json present, fetched_at = {manifest.get('fetched_at')}")

    declared = manifest.get("files", {})
    expected = {f"{s}_{tf}" for s in symbols for tf in TIMEFRAMES}
    missing = expected - set(declared)
    if missing:
        fail(f"manifest does not declare: {sorted(missing)}")

    for key, entry in sorted(declared.items()):
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


# ---------------------------------------------------------------------------
# 2. Timestamp integrity, including the broker-offset diagnosis
# ---------------------------------------------------------------------------

diagnose_grid = ta.diagnose_grid  # moved to timeframe_alignment.py so classify_gaps can share it


def check_series(candles_by_symbol) -> dict:
    section("2. TIMESTAMP INTEGRITY")

    grids = {}
    for symbol, tf_map in sorted(candles_by_symbol.items()):
        for timeframe, series in tf_map.items():
            name = f"{symbol}_{timeframe}"
            issues = ta.validate_series(series, timeframe)
            grid = diagnose_grid(series, timeframe)
            grids[name] = grid

            for key in ("unsorted", "duplicate_timestamps", "bad_ohlc", "non_finite"):
                if issues[key]:
                    fail(f"{name}: {issues[key]} {key}")

            if grid["on_utc_grid"]:
                ok(f"{name}: on the UTC {timeframe} grid, "
                   f"{issues['gaps']} gaps (largest {issues['largest_gap_bars']} bars)")
            elif grid["constant_offset"]:
                warn(f"{name}: every bar is offset {grid['modal_offset_hours']}h from the "
                     f"UTC {timeframe} grid — broker server time. Alignment is unaffected "
                     f"(it compares timestamps to each other) but SESSION LABELS would be "
                     f"shifted by {grid['modal_offset_hours']}h")
            else:
                fail(f"{name}: {grid['distinct_offsets']} different grid offsets "
                     f"(modal {grid['modal_offset_hours']}h covers only "
                     f"{grid['modal_share']:.1%}) — the series is not a single timeframe")

            if issues["largest_gap_bars"] > 100:
                warn(f"{name}: largest gap is {issues['largest_gap_bars']} bars "
                     f"(~{issues['largest_gap_bars'] * ta.duration(timeframe) / 86400:.1f} days)")

    return grids


# ---------------------------------------------------------------------------
# 3. Completed candles only
# ---------------------------------------------------------------------------

def check_completed(candles_by_symbol, manifest) -> None:
    section("3. COMPLETED CANDLES ONLY")

    fetched_at = None
    if manifest.get("fetched_at"):
        try:
            fetched_at = datetime.fromisoformat(manifest["fetched_at"]).timestamp()
        except ValueError:
            warn(f"cannot parse fetched_at: {manifest['fetched_at']!r}")

    now = datetime.now(timezone.utc).timestamp()

    for symbol, tf_map in sorted(candles_by_symbol.items()):
        for timeframe, series in tf_map.items():
            name = f"{symbol}_{timeframe}"
            newest = float(series[-1]["t"])
            closes_at = ta.close_time(newest, timeframe)

            if closes_at > now:
                fail(f"{name}: newest bar opens {utc(newest)} and does not close until "
                     f"{utc(closes_at)} — a forming candle reached the export")
                continue

            if fetched_at is not None and closes_at > fetched_at:
                fail(f"{name}: newest bar closes {utc(closes_at)}, after the fetch at "
                     f"{utc(fetched_at)} — it was still forming when exported")
            else:
                ok(f"{name}: newest bar closed {utc(closes_at)}, before the fetch")


# ---------------------------------------------------------------------------
# 4. Cross-timeframe coverage
# ---------------------------------------------------------------------------

def check_coverage(candles_by_symbol) -> None:
    section("4. CROSS-TIMEFRAME COVERAGE")

    for symbol, tf_map in sorted(candles_by_symbol.items()):
        h4, h1 = tf_map["H4"], tf_map["H1"]
        h4_start, h4_end = float(h4[0]["t"]), float(h4[-1]["t"])
        h1_start, h1_end = float(h1[0]["t"]), float(h1[-1]["t"])

        if h1_start > h4_start:
            lost = sum(1 for c in h4 if float(c["t"]) < h1_start)
            warn(f"{symbol}: H1 starts {utc(h1_start)}, after H4's {utc(h4_start)} — "
                 f"the first {lost} H4 bars have no H1 history and will be dropped")
        else:
            ok(f"{symbol}: H1 covers the start of H4")

        if h1_end < h4_end:
            warn(f"{symbol}: H1 ends {utc(h1_end)}, before H4's {utc(h4_end)}")

        # How many H1 bars does a decision actually see? Four is the maximum
        # inside one H4 bar; consistently fewer means H1 has holes.
        counts = []
        for i in range(len(h4) - 200, len(h4) - 1, 7):
            if i < 100:
                continue
            at = ta.decision_time(h4, i, "H4")
            visible = ta.closed_slice(h1, "H1", at)
            recent = sum(1 for c in visible if float(c["t"]) >= float(h4[i]["t"]))
            counts.append(recent)
        if counts:
            distribution = Counter(counts)
            ok(f"{symbol}: H1 bars available inside the decision H4 bar: "
               f"{dict(sorted(distribution.items()))} (4 is complete)")
            if statistics.mean(counts) < 3.0:
                warn(f"{symbol}: on average only {statistics.mean(counts):.2f} of 4 H1 "
                     f"bars are present per H4 bar — H1 history has holes")


# ---------------------------------------------------------------------------
# 5-6. Build the dataset and run the gate
# ---------------------------------------------------------------------------

def build_and_gate(candles_by_symbol, horizon: int):
    section("5. DATASET BUILD")

    X, y, meta = trainer.build_dataset(candles_by_symbol, horizon)
    if not X:
        fail("no rows were built")
        return None, None, None, None

    wins = sum(1 for v in y if v == 1.0)
    print(f"  rows       {len(X)}")
    print(f"  features   {len(X[0])}  ({', '.join(spec.FEATURE_NAMES)})")
    print(f"  win rate   {wins / len(y):.4f}")
    print(f"  BUY/SELL   {sum(1 for m in meta if m['direction'] == 'BUY')} / "
          f"{sum(1 for m in meta if m['direction'] == 'SELL')}")
    print(f"  span       {utc(min(m['t'] for m in meta))} -> {utc(max(m['t'] for m in meta))}")
    print(f"  reasons    {dict(Counter(m['reason'] for m in meta))}")

    per_symbol = defaultdict(lambda: [0, 0])
    for label, info in zip(y, meta):
        per_symbol[info["symbol"]][0] += 1
        per_symbol[info["symbol"]][1] += int(label == 1.0)
    for symbol, (n, w) in sorted(per_symbol.items()):
        print(f"    {symbol:8s} {n:6d} rows  win rate {w / n:.4f}")

    section("6. DATASET GATE")
    report = trainer.validate_dataset(X, y, meta)

    checks = [
        ("non-finite values", report.get("non_finite_values", 0) == 0,
         report.get("non_finite_values")),
        ("rows of wrong width", report.get("wrong_width_rows", 0) == 0,
         report.get("wrong_width_rows")),
        ("out-of-order rows", report.get("out_of_order_rows", 0) == 0,
         report.get("out_of_order_rows")),
        ("duplicate decisions", report.get("duplicate_decisions", 0) == 0,
         report.get("duplicate_decisions")),
        ("impossible entry prices", report.get("impossible_entry_prices", 0) == 0,
         report.get("impossible_entry_prices")),
        ("class balance 25-75%", report.get("overall", {}).get("balance_ok", False),
         report.get("overall", {}).get("win_rate")),
        ("symbol balance >= 0.2", report.get("symbol_balance_ratio", 0) >= 0.2,
         report.get("symbol_balance_ratio")),
        ("no leaky features", not report.get("leaky_features"),
         report.get("leaky_features")),
        ("no leakage suspects", not report.get("leakage_suspects"),
         report.get("leakage_suspects")),
    ]
    for label, passed, value in checks:
        (ok if passed else fail)(f"{label}: {value}")

    unexpected = set(report.get("constant_features", [])) - set(spec.LIVE_CONSTANT_FEATURES)
    (ok if not unexpected else fail)(f"no unexpected constant features: {sorted(unexpected)}")

    print("\n  single-feature AUC (tie-corrected; >=0.90 would mean leakage):")
    for name, auc in sorted(report.get("single_feature_auc", {}).items(),
                            key=lambda kv: -kv[1]):
        print(f"    {name:20s} {auc:.4f}")

    print("\n  duplicate rows: "
          f"{report.get('duplicate_rows')} ({report.get('duplicate_pct', 0):.4%})")
    if report.get("duplicate_pct", 0) > 0.02:
        warn(f"{report['duplicate_pct']:.2%} of rows are exact duplicates")

    return X, y, meta, report


# ---------------------------------------------------------------------------
# 7. Leakage, measured on the real series
# ---------------------------------------------------------------------------

def check_leakage(candles_by_symbol, horizon: int) -> None:
    section("7. LEAKAGE (measured on this data)")

    symbol = sorted(candles_by_symbol)[0]
    h4 = candles_by_symbol[symbol]["H4"]
    h1 = candles_by_symbol[symbol]["H1"]
    cut = int(len(h4) * 0.7)

    def vectors(series):
        X, _, meta = trainer.build_dataset({symbol: {"H4": series, "H1": h1}}, horizon)
        boundary = ta.decision_time(h4, cut - 1, "H4")
        return {(m["t"], m["direction"]): tuple(x)
                for x, m in zip(X, meta) if m["t"] <= boundary}

    base = vectors(h4)

    future = [dict(c) for c in h4]
    for j in range(cut, len(future)):
        for key in ("open", "high", "low", "close"):
            future[j][key] *= 7.0
    mutated_future = vectors(future)

    common = set(base) & set(mutated_future)
    differing = [k for k in common if base[k] != mutated_future[k]]
    print(f"  {symbol}: {len(common)} decisions at or before the mutation point")
    if differing:
        fail(f"{len(differing)} feature vectors changed when the FUTURE was multiplied "
             f"by 7 — something reads past the decision")
    else:
        ok("0 feature vectors changed when the future was multiplied by 7")

    past = [dict(c) for c in h4]
    for j in range(cut):
        for key in ("open", "high", "low", "close"):
            past[j][key] *= 7.0
    mutated_past = vectors(past)
    common_past = set(base) & set(mutated_past)
    changed = [k for k in common_past if base[k] != mutated_past[k]]
    if not changed:
        fail("mutating the PAST changed nothing either — the leakage check is vacuous "
             "and proves nothing")
    else:
        ok(f"{len(changed)}/{len(common_past)} vectors changed when the past was "
           f"mutated (the check is not vacuous)")


# ---------------------------------------------------------------------------
# 8. Entry price
# ---------------------------------------------------------------------------

def check_entry_price(candles_by_symbol, meta, horizon: int) -> None:
    section("8. ENTRY PRICE")

    by_symbol = defaultdict(list)
    for row in meta:
        by_symbol[row["symbol"]].append(row)

    for symbol, rows in sorted(by_symbol.items()):
        h4 = candles_by_symbol[symbol]["H4"]
        index_of = {float(c["t"]): i for i, c in enumerate(h4)}

        mismatched = 0
        equals_close = 0
        gaps = []
        checked = 0
        for row in rows[::7]:
            decision_open = row["t"] - ta.duration("H4")
            i = index_of.get(decision_open)
            if i is None or i + 1 >= len(h4):
                continue
            checked += 1
            expected = float(h4[i + 1]["open"])
            if row["entry_price"] != expected:
                mismatched += 1
            if row["entry_price"] == float(h4[i]["close"]):
                equals_close += 1
            gaps.append(abs(expected - float(h4[i]["close"])))

        if not checked:
            fail(f"{symbol}: could not verify any entry price")
            continue

        (ok if not mismatched else fail)(
            f"{symbol}: entry price == next bar's open in {checked - mismatched}/{checked} "
            f"sampled rows")

        # An overnight/weekend gap is normal; if the next open ALWAYS equals the
        # previous close the broker is stitching bars, and next-open vs
        # this-close stops being a meaningful distinction.
        share = equals_close / checked
        median_gap = statistics.median(gaps) if gaps else 0.0
        if share > 0.98:
            warn(f"{symbol}: next open equals previous close in {share:.1%} of bars "
                 f"(median gap {median_gap:.6f}) — this feed has no bar-to-bar gaps, so "
                 f"the executable-entry fix changes nothing here in practice")
        else:
            ok(f"{symbol}: next open differs from previous close in {1 - share:.1%} of "
               f"bars, median gap {median_gap:.6f}")


# ---------------------------------------------------------------------------
# 9. Labels
# ---------------------------------------------------------------------------

def check_labels(candles_by_symbol, meta, horizon: int) -> None:
    section("9. LABELS")

    reasons = Counter(row["reason"] for row in meta)
    total = len(meta)
    print(f"  resolutions: {dict(reasons)}")

    both = reasons.get("both_same_bar", 0)
    if total and both / total > 0.15:
        warn(f"{both / total:.1%} of trades touched both barriers in one bar and were "
             f"labelled losses — the barriers are close relative to an H4 bar's range, "
             f"so the label is pessimistic by that much")
    else:
        ok(f"both-barriers-in-one-bar: {both} rows ({both / max(total, 1):.2%})")

    holding = [row["bars"] for row in meta]
    if holding:
        print(f"  holding bars: min {min(holding)}, median "
              f"{statistics.median(holding):.0f}, max {max(holding)}")
        if max(holding) > horizon:
            fail(f"a trade held {max(holding)} bars, beyond the horizon of {horizon}")
        else:
            ok(f"no trade exceeded the {horizon}-bar horizon")

    # Resolution rate: how many decisions produced a usable row at all.
    decisions = 0
    for symbol, tf_map in candles_by_symbol.items():
        h4 = tf_map["H4"]
        decisions += 2 * max(0, len(h4) - trainer.WARMUP_BARS - horizon - 1)
    if decisions:
        rate = total / decisions
        print(f"  resolved {total} of ~{decisions} candidate decisions ({rate:.1%})")
        if rate < 0.5:
            warn(f"only {rate:.1%} of decisions resolved inside {horizon} bars — "
                 f"consider whether the horizon is long enough")
        else:
            ok(f"{rate:.1%} of candidate decisions resolved")

    per_direction = defaultdict(lambda: [0, 0])
    for row, label in zip(meta, [1.0 if r["reason"] == "tp_first" else 0.0 for r in meta]):
        per_direction[row["direction"]][0] += 1
        per_direction[row["direction"]][1] += int(label == 1.0)
    for direction, (n, w) in sorted(per_direction.items()):
        print(f"    {direction}: {n} rows, win rate {w / n:.4f}")


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", default=os.path.join("data", "historical"))
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--horizon", type=int, default=trainer.DEFAULT_HORIZON)
    args = parser.parse_args()

    from config import SYMBOLS
    symbols = args.symbols or list(SYMBOLS)

    print(f"Validating {args.candles} for {symbols}, horizon {args.horizon} H4 bars")
    print(f"Nothing will be trained, written or installed.")

    manifest = check_provenance(args.candles, symbols)
    if failures:
        print("\nStopping: the data cannot be identified.")
        return 1

    candles_by_symbol = {}
    for symbol in symbols:
        try:
            candles_by_symbol[symbol] = {
                tf: trainer.load_candles(symbol, tf, args.candles) for tf in TIMEFRAMES
            }
        except FileNotFoundError as exc:
            fail(str(exc))
            return 1

    check_series(candles_by_symbol)
    check_completed(candles_by_symbol, manifest)
    check_coverage(candles_by_symbol)

    X, y, meta, report = build_and_gate(candles_by_symbol, args.horizon)
    if X is None:
        return 1

    check_leakage(candles_by_symbol, args.horizon)
    check_entry_price(candles_by_symbol, meta, args.horizon)
    check_labels(candles_by_symbol, meta, args.horizon)

    section("VERDICT")
    if failures:
        print(f"  DATASET GATE FAILED — {len(failures)} blocking problem(s):\n")
        for item in failures:
            print(f"    - {item}")
        if warnings:
            print(f"\n  plus {len(warnings)} warning(s):")
            for item in warnings:
                print(f"    - {item}")
        print("\n  Do not train. Fix these first.")
        return 1

    print("  DATASET GATE PASSED")
    if warnings:
        print(f"\n  {len(warnings)} warning(s) — not blocking, but read them:")
        for item in warnings:
            print(f"    - {item}")
    print("\n  Baseline training may proceed:")
    print("    python scripts/train_entry_model.py --dry-run --folds 5")
    print("\n  --dry-run trains and validates but installs nothing. Installing "
          "additionally requires KAIROS_ALLOW_MODEL_INSTALL=1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
