#!/usr/bin/env python3
"""Build the shared candle fixtures the golden-parity tests run on.

    python scripts/build_research_fixtures.py

Why a fixture and not the live history
--------------------------------------
Golden parity asks one question: given IDENTICAL input, does KAIROS compute
the same feature vector and the same probability as the research repo that
trained the model? Answering it needs a single candle file both repositories
can read, small enough to commit and frozen so the answer cannot drift.

The `spread` column
-------------------
KAIROS's stored candles (``data/historical/*.json``) carry no spread, and every
research model requires ``spread_relative``. The fixtures therefore add a
spread column computed by a fixed, documented function of the bar itself. It
is written to the manifest as ``spread_source: SYNTHETIC_FIXTURE_ONLY`` and it
is NOT a claim about any broker's spreads.

That is legitimate for a parity test — both sides receive the same numbers, so
any disagreement is an implementation difference — and it is NOT legitimate as
a way to serve a model on real data. Against a real source with no spread
column, the pipeline correctly refuses to predict; see
``tests/test_research_missing_policy.py``.

Provenance per timeframe
------------------------
* H1 / H4 — REAL OHLC, sliced from ``data/historical/<SYM>_<TF>.json``.
* M15     — SYNTHETIC. KAIROS stores no M15 candles at all, so the M15 stack
            is a deterministic random walk. It exercises the M15 contract and
            the two-context-timeframe merge; it says nothing about markets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.research.candles import KairosHistoricalSource  # noqa: E402

OUT = Path("tests/fixtures/research/candles")

# Enough history that every lookback window (the longest is spread_relative's
# 200 bars) is complete well before the scored window begins.
H1_BARS, H4_BARS = 700, 340
M15_BARS, M15_H1_BARS, M15_H4_BARS = 1400, 700, 340

SYMBOLS = ("EURUSD", "GBPUSD", "XAUUSD")


def fixture_spread(df: pd.DataFrame, symbol: str) -> pd.Series:
    """A deterministic, reproducible spread. NOT a broker measurement.

    Derived from the bar's own range so it is a pure function of the candle
    file: no RNG state to carry, identical in both repositories, and stable
    across runs and machines. Quantised to whole points and floored at zero,
    which reproduces the shape real feeds have — including the literal zeros
    that the research repo's ``UNIT_WHEN_ALSO_ZERO`` policy exists to handle.
    """
    point = 0.01 if symbol == "XAUUSD" else 0.00001
    rng = (df["high"] - df["low"]) / point
    return np.floor(rng * 0.015).clip(0, 40).astype(float)


#: Price precision the fixtures are stored at. Rounding here rather than at
#: read time means BOTH repositories load the identical float from the file,
#: which is what makes a bit-exact parity claim possible at all.
PRICE_DECIMALS = 5


def _records(df: pd.DataFrame) -> list:
    out = df.copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].round(PRICE_DECIMALS)
    return out.to_dict("records")


def write(symbol: str, timeframe: str, df: pd.DataFrame, provenance: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{symbol}_{timeframe}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_records(df), fh, separators=(",", ":"))
    return {
        "path": str(path), "bars": len(df), "ohlc_provenance": provenance,
        "spread_source": "SYNTHETIC_FIXTURE_ONLY",
        "first": str(df["timestamp"].iloc[0]), "last": str(df["timestamp"].iloc[-1]),
    }


def synthetic_stack(symbol: str) -> dict:
    """A deterministic M15/H1/H4 stack, aligned so the merges are meaningful."""
    seed = abs(hash(symbol)) % (2 ** 31)
    rng = np.random.default_rng(seed)
    base = {"XAUUSD": 2000.0, "EURUSD": 1.10, "GBPUSD": 1.27}[symbol]
    vol = base * 0.0006

    end = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    out = {}
    for tf, minutes, n in (("M15", 15, M15_BARS), ("H1", 60, M15_H1_BARS),
                           ("H4", 240, M15_H4_BARS)):
        ts = pd.date_range(end=end, periods=n, freq=f"{minutes}min", tz="UTC")
        step = vol * np.sqrt(minutes / 15.0)
        close = base + np.cumsum(rng.normal(0, step, n))
        open_ = np.concatenate([[close[0]], close[:-1]])
        wick = np.abs(rng.normal(0, step, n))
        high = np.maximum(open_, close) + wick
        low = np.minimum(open_, close) - np.abs(rng.normal(0, step, n))
        df = pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                           "low": low, "close": close})
        df["spread"] = fixture_spread(df, symbol)
        out[tf] = df
    return out


def main() -> int:
    source = KairosHistoricalSource("data/historical")
    manifest = {
        "note": ("Fixtures for research golden-parity tests. The `spread` column is "
                 "SYNTHETIC in every file (see scripts/build_research_fixtures.py). "
                 "H1/H4 OHLC is real; the M15 stack is entirely synthetic because "
                 "KAIROS stores no M15 candles."),
        "files": {},
    }

    for symbol in SYMBOLS:
        for timeframe, n in (("H1", H1_BARS), ("H4", H4_BARS)):
            df = source.load(symbol, timeframe).tail(n).reset_index(drop=True)
            df["spread"] = fixture_spread(df, symbol)
            manifest["files"][f"{symbol}_{timeframe}"] = write(
                symbol, timeframe, df,
                f"REAL, tail({n}) of data/historical/{symbol}_{timeframe}.json")
            print(f"{symbol} {timeframe}: {len(df)} real bars")

    # A separate synthetic namespace so a synthetic file can never be mistaken
    # for the real one: SYNTH_<SYMBOL>.
    for symbol in SYMBOLS:
        for timeframe, df in synthetic_stack(symbol).items():
            manifest["files"][f"SYNTH_{symbol}_{timeframe}"] = write(
                f"SYNTH_{symbol}", timeframe, df,
                "SYNTHETIC deterministic random walk (KAIROS stores no M15 candles)")
        print(f"SYNTH_{symbol}: synthetic M15/H1/H4 stack")

    path = OUT / "manifest.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"\nwrote {path} ({len(manifest['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
