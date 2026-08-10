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

Writes ``data/historical/<SYMBOL>_<TF>.json`` — plain OHLC, no indicators, no
labels. Read-only: it opens no positions and modifies no bot state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SYMBOLS  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("fetch_training_candles")

OUT_DIR = os.path.join("data", "historical")

# H4 drives labelling (the horizon is counted in H4 bars, matching how the
# live trade-age layer counts). H1 supplies rsi/macd exactly as the live
# snapshot does.
TIMEFRAMES = ("H4", "H1")

# Bars per year, generous upper bounds (forex ~6 sessions/week).
_BARS_PER_YEAR = {"H4": 1600, "H1": 6400}


def _mt5_timeframe(mt5, tf: str):
    return {"H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}[tf]


def fetch(symbol: str, timeframe: str, count: int) -> list:
    """Completed candles only, newest last."""
    import MetaTrader5 as mt5  # imported here so the module loads off-Windows

    from data.market.mt5_session import ensure_symbol, mt5_call

    if not ensure_symbol(symbol):
        raise RuntimeError(f"symbol {symbol} not available in Market Watch")

    with mt5_call():
        # +1 then drop the last: the newest bar is still forming, and including
        # it would leak partial future information into a training row.
        rates = mt5.copy_rates_from_pos(symbol, _mt5_timeframe(mt5, timeframe), 0, count + 1)

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"no candles returned for {symbol} {timeframe}: {mt5.last_error()}")

    rows = [
        {
            "t": float(r["time"]),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r["tick_volume"]),
        }
        for r in rates[:-1]
    ]
    rows.sort(key=lambda c: c["t"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=list(SYMBOLS))
    parser.add_argument("--years", type=float, default=3.0,
                        help="how much history to request per timeframe (default 3)")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

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
    manifest = {"fetched_at": datetime.now(timezone.utc).isoformat(), "files": {}}
    failures = []

    for symbol in args.symbols:
        for tf in TIMEFRAMES:
            count = int(_BARS_PER_YEAR[tf] * args.years)
            try:
                candles = fetch(symbol, tf, count)
            except Exception as exc:
                logger.error("fetch failed for %s %s: %s", symbol, tf, exc)
                failures.append(f"{symbol} {tf}: {exc}")
                continue

            path = os.path.join(args.out, f"{symbol}_{tf}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(candles, fh)

            first = datetime.fromtimestamp(candles[0]["t"], tz=timezone.utc).date()
            last = datetime.fromtimestamp(candles[-1]["t"], tz=timezone.utc).date()
            manifest["files"][f"{symbol}_{tf}"] = {
                "path": path, "bars": len(candles),
                "first": str(first), "last": str(last),
            }
            print(f"  {symbol:8s} {tf:3s}  {len(candles):6d} bars  {first} -> {last}")

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
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
