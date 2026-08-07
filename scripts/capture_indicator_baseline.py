#!/usr/bin/env python3
"""Capture a numeric baseline of indicator values before the QuantDinger removal.

Why this exists
---------------
The migration swaps the *source* of market data (QuantDinger -> MT5) but must
not change the *maths*. The entry model was trained on features produced by
data/market/client.py's formulas; if the new module computes RSI or MACD even
slightly differently, the feature distribution shifts and p_win changes for
reasons nobody can trace back.

This script records what both paths return for the same symbols/timeframes at
the same moment. Run it BEFORE the migration, then again AFTER, and diff the
two files. Identical numbers mean the source changed and nothing else.

Usage
-----
    python scripts/capture_indicator_baseline.py --out baseline_before.json
    # ... perform migration ...
    python scripts/capture_indicator_baseline.py --out baseline_after.json
    python scripts/capture_indicator_baseline.py --compare baseline_before.json baseline_after.json

Read-only: makes data requests, touches no trading logic and no positions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TIMEFRAMES = ["H4", "H1", "M15"]
# Fields that feed the entry model; these are the ones that must not drift.
TRACKED = ["rsi", "atr", "macd", "ma_trend", "close"]
# Floating point noise tolerance when comparing two captures.
TOLERANCE = 1e-9


def _capture_one(getter, symbol: str, timeframe: str) -> dict:
    try:
        data = getter(symbol, timeframe) or {}
        return {k: data.get(k) for k in TRACKED}
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


def capture(symbols) -> dict:
    """Snapshot both data paths for every symbol/timeframe pair."""
    result = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "symbols": {},
    }

    # QuantDinger path (the source of truth the entry model was trained against)
    try:
        from data.market.client import get_indicators as qd_get_indicators
    except Exception as exc:
        qd_get_indicators = None
        result["quantdinger_import_error"] = str(exc)

    # MT5 path — after the migration this import resolves to the new module.
    try:
        from data.market.hybrid_client import get_indicators_hybrid as mt5_get_indicators
    except Exception as exc:
        mt5_get_indicators = None
        result["mt5_import_error"] = str(exc)

    for symbol in symbols:
        result["symbols"][symbol] = {}
        for tf in TIMEFRAMES:
            entry = {}
            if qd_get_indicators is not None:
                entry["quantdinger"] = _capture_one(qd_get_indicators, symbol, tf)
            if mt5_get_indicators is not None:
                entry["mt5"] = _capture_one(mt5_get_indicators, symbol, tf)
            result["symbols"][symbol][tf] = entry
            print(f"  {symbol:8s} {tf:4s}  {entry}")

    return result


def _values_match(a, b) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= TOLERANCE
    return a == b


def compare(before_path: str, after_path: str) -> int:
    """Diff two captures. Returns a non-zero exit code when anything drifted."""
    with open(before_path, encoding="utf-8") as fh:
        before = json.load(fh)
    with open(after_path, encoding="utf-8") as fh:
        after = json.load(fh)

    drift = []
    for symbol, tfs in before.get("symbols", {}).items():
        for tf, sources in tfs.items():
            # The migration's contract: what MT5 returns afterwards must match
            # what QuantDinger returned before, field for field.
            baseline = sources.get("quantdinger") or {}
            current = (
                after.get("symbols", {}).get(symbol, {}).get(tf, {}).get("mt5")
                or {}
            )
            for field in TRACKED:
                old, new = baseline.get(field), current.get(field)
                if not _values_match(old, new):
                    drift.append((symbol, tf, field, old, new))

    print(f"\n=== Baseline comparison: {before_path} -> {after_path} ===")
    if not drift:
        print("PASS: every tracked field matches. The source changed, the maths did not.")
        return 0

    print(f"DRIFT DETECTED in {len(drift)} field(s):\n")
    print(f"{'symbol':10s} {'tf':5s} {'field':10s} {'before':>18s} {'after':>18s}")
    print("-" * 68)
    for symbol, tf, field, old, new in drift:
        print(f"{symbol:10s} {tf:5s} {field:10s} {str(old):>18s} {str(new):>18s}")
    print(
        "\nThe migration was supposed to change only the data source. Any drift "
        "here means a formula changed too, which shifts the entry model's input "
        "distribution. Investigate before merging."
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="baseline.json")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--symbols", nargs="*", default=None)
    args = parser.parse_args()

    if args.compare:
        return compare(args.compare[0], args.compare[1])

    symbols = args.symbols
    if not symbols:
        from config import SYMBOLS

        symbols = SYMBOLS

    print(f"=== Capturing indicator baseline for {symbols} ===")
    data = capture(symbols)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
