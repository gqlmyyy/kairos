"""Phase 1/2 forensic audit: what the deployed 65-feature entry model was trained on.

Every claim in ENTRY_PIPELINE_AUDIT.md is produced here, so it can be re-run and
disputed rather than taken on trust.

Four checks, each independent:

  1. entry_price provenance   — is the price the TP/SL barriers are built from
                                actually the candle close?
  2. direction coverage       — can the label distinguish BUY from SELL?
  3. H4 indicator provenance  — are the "H4" indicators computed over H4 candles?
  4. look-ahead               — is the H4 candle attached at decision time t
                                already closed at t?

Check 4 is the decisive one. The other three are severe but ordinary bugs; a
feature that reads the future invalidates every metric ever measured on this
dataset.

The inputs are the artifacts the entry_v2 pipeline itself wrote:

    data/entry_v2/entry_v2_dataset_*.csv   (dataset_builder)
    data/entry_v2/features_dataset.parquet (feature_engineering)
    data/entry_v2/labeled_dataset.parquet  (entry_labels)

These are gitignored, so this script reports what is missing instead of failing
when they are absent.

Usage::

    python scripts/audit_entry_pipeline.py
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

DATA = os.path.join("data", "entry_v2")


def _load():
    unified = sorted(glob.glob(os.path.join(DATA, "entry_v2_dataset_*.csv")))
    feats = os.path.join(DATA, "features_dataset.parquet")
    labels = os.path.join(DATA, "labeled_dataset.parquet")
    missing = [p for p in (feats, labels) if not os.path.exists(p)]
    if not unified:
        missing.append(os.path.join(DATA, "entry_v2_dataset_*.csv"))
    if missing:
        print("Cannot audit — these entry_v2 artifacts are absent:")
        for m in missing:
            print(f"  {m}")
        print("\nThey are gitignored. Regenerate with "
              "`python -m analysis.entry_v2.run_pipeline` on a machine with MT5.")
        return None
    return pd.read_csv(unified[-1]), pd.read_parquet(feats), pd.read_parquet(labels)


def _wilder_rsi(closes, period: int = 14):
    c = np.asarray(closes, dtype=float)
    d = np.diff(c, prepend=c[0])
    gains, losses = np.clip(d, 0, None), np.clip(-d, 0, None)
    out = np.full(len(c), np.nan)
    avg_gain = avg_loss = None
    for i in range(len(c)):
        if i < period:
            continue
        if avg_gain is None:
            avg_gain = gains[i - period + 1:i + 1].mean()
            avg_loss = losses[i - period + 1:i + 1].mean()
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def check_entry_price(unified, labeled) -> None:
    print("=" * 74)
    print("1. Is entry_price the candle close?")
    print("=" * 74)
    if "entry_price" not in labeled.columns:
        print("  no entry_price column\n")
        return

    same = (labeled["entry_price"] - labeled["h4_ema_50"]).abs()
    print(f"  entry_price == h4_ema_50 in {100 * (same == 0).mean():.4f}% of "
          f"{len(labeled)} rows")

    cols = ["symbol", "t", "h4_close"]
    merged = labeled.merge(unified[cols], on=["symbol", "t"], how="left")
    ok = merged["h4_close"].notna()
    err = (merged["entry_price"] - merged["h4_close"]).abs()[ok]
    print(f"  |entry_price - h4_close|: mean={err.mean():.6g} "
          f"median={err.median():.6g} max={err.max():.6g}")

    # The number that matters: the barriers are 1.0 ATR (SL) and 1.5 ATR (TP),
    # so an offset measured in ATR is directly comparable to the barrier width.
    if "entry_atr" in merged.columns:
        in_atr = (err / merged["entry_atr"][ok])
        print(f"  in units of entry_atr:    mean={in_atr.mean():.3f} "
              f"p95={in_atr.quantile(0.95):.3f}")
        print("  (barriers are SL 1.0 ATR / TP 1.5 ATR — an offset of the same "
              "size\n   means the label describes a trade nobody could place)")
    print()


def check_direction(labeled) -> None:
    print("=" * 74)
    print("2. Can the label tell BUY from SELL?")
    print("=" * 74)
    has = "direction" in labeled.columns
    print(f"  'direction' column present: {has}")
    if not has:
        print("  -> entry_labels._direction_from_row() returns 1 (BUY) for every row,")
        print(f"     so all {len(labeled)} rows are labelled as BUY trades.")
    print(f"  win rate: {labeled['label'].mean():.4f}")
    print(f"  label_reason: {labeled['label_reason'].value_counts().to_dict()}")
    print()


def check_h4_indicators(unified, feats) -> None:
    print("=" * 74)
    print("3. Are the 'H4' indicators computed over H4 candles?")
    print("=" * 74)
    for sym in sorted(unified["symbol"].unique()):
        s = unified[unified["symbol"] == sym].sort_values("t")
        s = s[s["h4_close"].notna()].reset_index(drop=True)
        fe = feats[feats["symbol"] == sym].sort_values("t")
        if s.empty or fe.empty:
            continue

        changes = s["h4_close"].ne(s["h4_close"].shift()).mean()

        # Same RSI formula, two different input series.
        oversampled = pd.DataFrame({"t": s["t"].values,
                                    "rsi": _wilder_rsi(s["h4_close"].values)})
        distinct = s.loc[s["h4_close"].ne(s["h4_close"].shift())].copy()
        distinct["rsi"] = _wilder_rsi(distinct["h4_close"].values)
        true_h4 = pd.merge_asof(s[["t"]].sort_values("t"),
                                distinct[["t", "rsi"]].sort_values("t"),
                                on="t", direction="backward")

        m = (fe.merge(oversampled, on="t", how="inner", suffixes=("", "_over"))
               .merge(true_h4, on="t", how="inner", suffixes=("_over", "_true")))
        if m.empty:
            continue

        e_over = (m["h4_rsi_14"] - m["rsi_over"]).abs()
        e_true = (m["h4_rsi_14"] - m["rsi_true"]).abs()
        print(f"  {sym}: h4_close changes on {100 * changes:.2f}% of hourly rows "
              f"(25% => each H4 candle repeated 4x)")
        print(f"     shipped h4_rsi_14 vs RSI over the H1 grid : "
              f"mean|err|={e_over.mean():7.4f}")
        print(f"     shipped h4_rsi_14 vs RSI over true H4     : "
              f"mean|err|={e_true.mean():7.4f}")
    print("  -> a match against the H1-grid series means the lookback is 14 hours,")
    print("     not 14 H4 candles; 'h4_sma_200' spans 50 H4 candles, not 200.")
    print()


def check_lookahead(unified) -> None:
    print("=" * 74)
    print("4. Is the H4 candle attached at time t already closed at t?")
    print("=" * 74)
    for sym in sorted(unified["symbol"].unique()):
        s = unified[unified["symbol"] == sym].sort_values("t").reset_index(drop=True)
        s = s[s["h4_close"].notna() & s["h1_close"].notna()].reset_index(drop=True)
        if len(s) < 100:
            continue
        new = (s["h4_close"].ne(s["h4_close"].shift())
               | s["h4_open"].ne(s["h4_open"].shift()))
        idx = np.flatnonzero(new.values)

        opens = np.mean(np.abs(s["h4_open"].values[idx] - s["h1_open"].values[idx]) < 1e-9)

        covers_h = covers_l = n = 0
        for k in idx:
            fut = s.iloc[k:k + 4]
            if len(fut) < 4:
                continue
            n += 1
            covers_h += s["h4_high"].values[k] >= fut["h1_high"].max() - 1e-9
            covers_l += s["h4_low"].values[k] <= fut["h1_low"].min() + 1e-9

        closes = [abs(s["h4_close"].values[k] - s["h1_close"].values[k + 3]) < 1e-9
                  for k in idx if k + 3 < len(s)]

        print(f"  {sym} ({len(idx)} distinct H4 candles)")
        print(f"     H4 open == H1 open at first appearance : {100 * opens:6.2f}%"
              "   (candle OPENS at t)")
        print(f"     H4 high covers hours t..t+3            : {100 * covers_h / n:6.2f}%")
        print(f"     H4 low  covers hours t..t+3            : {100 * covers_l / n:6.2f}%")
        print(f"     H4 close == H1 close at t+3h           : {100 * np.mean(closes):6.2f}%")
    print("  -> the attached candle is the one still forming, so its close, high")
    print("     and low are up to 3 hours of future price. Every H4 feature, and")
    print("     entry_price via h4_ema_50, is contaminated.")
    print()


def main() -> int:
    loaded = _load()
    if loaded is None:
        return 1
    unified, feats, labeled = loaded
    print(f"unified={len(unified)} rows  features={len(feats)} rows  "
          f"labeled={len(labeled)} rows\n")

    check_entry_price(unified, labeled)
    check_direction(labeled)
    check_h4_indicators(unified, feats)
    check_lookahead(unified)

    print("=" * 74)
    print("Any single one of these invalidates the dataset. Check 4 invalidates")
    print("every metric ever measured on it, including the model's own test score.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
