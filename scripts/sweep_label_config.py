"""Phase 2/3: is the entry label learnable at all, and at which TP/SL/horizon?

Run before touching a single feature. The question this answers is not "which
config scores highest" — a maximum over 40+ configurations on one dataset is
mostly noise. It is:

  1. Does ANY (SL, TP, horizon) produce a label with conditional structure the
     current features can see?
  2. Is that structure stable across walk-forward folds and across symbols?

Two diagnostics carry the answer:

**Conditional win rate spread.** For each feature, the win rate is measured
inside each of its buckets with a binomial confidence interval. If no bucket
deviates from the base rate by more than sampling noise, no model can separate
this label with these features — and that is a feature problem, not a
hyper-parameter problem. This is computed without training anything, so it
cannot be confounded by model capacity or regularisation.

**Walk-forward AUC across folds.** Reported as mean *and* spread. A config whose
folds swing between 0.44 and 0.58 has no edge; it has variance.

Efficiency note: features do not depend on the label configuration, so they are
built once and reused for every config. Only the labels are recomputed.

Usage::

    python scripts/sweep_label_config.py                     # full sweep
    python scripts/sweep_label_config.py --quick             # fewer configs
    python scripts/sweep_label_config.py --out sweep.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.features import live_parity_features as lpf  # noqa: E402
from analysis.models import entry_feature_spec as spec  # noqa: E402

import train_entry_model as trainer  # noqa: E402
from analysis.features import timeframe_alignment as ta  # noqa: E402

CANDLE_DIR = os.path.join("data", "historical")

# Deliberately small and defensible. Every ratio here is one a trader would
# actually run; this is not a grid search for the maximum.
TP_SL_GRID = [
    (1.0, 1.5),
    (1.0, 2.0),
    (1.5, 2.0),
    (1.5, 2.5),   # current
    (1.5, 3.0),
    (2.0, 3.0),
]
HORIZONS = [4, 8, 12, 16, 20, 24, 32]

QUICK_TP_SL = [(1.0, 1.5), (1.5, 2.5), (1.5, 3.0)]
QUICK_HORIZONS = [8, 16, 24]


# ---------------------------------------------------------------------------
# Feature cache — built once, independent of the label configuration
# ---------------------------------------------------------------------------

def build_feature_cache(candles_by_symbol: Dict[str, Dict[str, List]]) -> List[Dict[str, Any]]:
    """One entry per (symbol, H4 bar) that live could actually have traded.

    Holds the feature vector minus `direction` (which the label loop varies),
    plus the ATR the label needs and the bar index for the forward walk.
    """
    rows: List[Dict[str, Any]] = []

    for symbol, tf_map in candles_by_symbol.items():
        h4, h1 = tf_map["H4"], tf_map["H1"]

        for i in range(trainer.WARMUP_BARS, len(h4) - max(HORIZONS) - 1):
            # Decision at the H4 bar's close; only closed candles are visible.
            bar_t = ta.decision_time(h4, i, "H4")

            h4_visible = ta.closed_slice(h4, "H4", bar_t)
            h1_visible = ta.closed_slice(h1, "H1", bar_t)
            if len(h4_visible) < trainer.WARMUP_BARS or len(h1_visible) < trainer.WARMUP_BARS:
                continue
            h4_ind = lpf.live_indicators(h4_visible)
            if h4_ind is None:
                continue
            h1_ind = lpf.live_indicators(h1_visible)
            if h1_ind is None:
                continue

            trend_score, trend_dir = lpf.trend_score_from_indicators(h4_ind)
            momentum, mom_dir = lpf.momentum_score_from_indicators(h1_ind)
            vol = lpf.volatility_score_from_indicators(h1_ind)
            regime = lpf.regime_from_scores(trend_dir, vol)

            rows.append({
                "symbol": symbol, "idx": i, "t": bar_t,
                "atr": float(h4_ind["atr"]),
                "regime": regime,
                "kwargs": dict(
                    rsi=h1_ind["rsi"], atr=h4_ind["atr"], macd=h1_ind["macd"],
                    trend_strength=lpf.mtf_strength_from_directions(
                        trend_dir, mom_dir, mom_dir),
                    trend_score=trend_score, momentum_score=momentum,
                    volatility_score=vol, market_regime=regime,
                    session=spec.session_from_timestamp(bar_t),
                ),
            })

    rows.sort(key=lambda r: r["t"])
    return rows


def label_dataset(cache, candles_by_symbol, sl_mult, tp_mult, horizon):
    """Relabel the cached features under one (SL, TP, horizon) configuration."""
    X, y, meta = [], [], []
    unresolved = 0
    attempts = 0

    # simulate_trade reads the module-level multipliers; set them for this pass.
    prev_sl, prev_tp = trainer.ATR_SL_BASE_MULTIPLIER, trainer.ATR_TP_BASE_MULTIPLIER
    trainer.ATR_SL_BASE_MULTIPLIER = sl_mult
    trainer.ATR_TP_BASE_MULTIPLIER = tp_mult
    try:
        for row in cache:
            h4 = candles_by_symbol[row["symbol"]]["H4"]
            for direction in ("BUY", "SELL"):
                attempts += 1
                out = trainer.simulate_trade(h4, row["idx"], direction,
                                             row["atr"], horizon)
                if out is None:
                    unresolved += 1
                    continue
                X.append(spec.build_feature_vector(direction=direction, **row["kwargs"]))
                y.append(out["label"])
                meta.append({"symbol": row["symbol"], "t": row["t"],
                             "direction": direction, "regime": row["regime"],
                             "reason": out["reason"], "bars": out["bars"]})
    finally:
        trainer.ATR_SL_BASE_MULTIPLIER = prev_sl
        trainer.ATR_TP_BASE_MULTIPLIER = prev_tp

    return X, y, meta, attempts, unresolved


# ---------------------------------------------------------------------------
# Model-free signal diagnostic
# ---------------------------------------------------------------------------

def conditional_signal(X, y, min_n: int = 200) -> Dict[str, Any]:
    """Win rate inside each feature bucket, with a binomial confidence interval.

    Trains nothing. If every bucket's interval covers the base rate, the label
    carries no structure these features can express, and no amount of model
    tuning will find any.
    """
    n = len(y)
    if n == 0:
        return {"error": "empty"}
    base = sum(y) / n

    findings = []
    for idx, name in enumerate(spec.FEATURE_NAMES):
        col = [row[idx] for row in X]
        uniq = sorted(set(col))
        # Bucket continuous features by quintile; use levels for categoricals.
        if len(uniq) > 12:
            qs = [statistics.quantiles(col, n=5)[k] for k in range(4)]
            def bucket(v, qs=qs):
                return sum(v > q for q in qs)
            keys = range(5)
        else:
            def bucket(v):
                return v
            keys = uniq

        groups: Dict[Any, List[float]] = defaultdict(list)
        for v, label in zip(col, y):
            groups[bucket(v)].append(label)

        for k in keys:
            vals = groups.get(k, [])
            if len(vals) < min_n:
                continue
            w = sum(vals)
            m = len(vals)
            rate = w / m
            se = math.sqrt(max(rate * (1 - rate), 1e-12) / m)
            z = (rate - base) / se if se > 0 else 0.0
            findings.append({
                "feature": name, "bucket": (k if not isinstance(k, float)
                                            else round(k, 4)),
                "n": m, "win_rate": round(rate, 4),
                "delta_vs_base": round(rate - base, 4),
                "z": round(z, 2),
            })

    findings.sort(key=lambda f: -abs(f["z"]))
    # |z| > 3 on a single bucket is the threshold for "worth a model's time".
    strong = [f for f in findings if abs(f["z"]) >= 3.0]
    return {
        "base_win_rate": round(base, 4),
        "buckets_examined": len(findings),
        "buckets_beyond_3_sigma": len(strong),
        "max_abs_z": findings[0]["z"] if findings else 0.0,
        "top": findings[:8],
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(X, y, meta, folds: int) -> Dict[str, Any]:
    wf = trainer.walk_forward(X, y, meta, folds=folds)
    if wf.get("skipped"):
        return {"skipped": wf["skipped"]}

    aucs = [f["overall"]["roc_auc"] for f in wf["folds"]
            if f["overall"]["roc_auc"] is not None]
    praucs = [f["overall"]["pr_auc"] for f in wf["folds"]
              if f["overall"]["pr_auc"] is not None]
    briers = [f["overall"].get("brier_skill_score") for f in wf["folds"]
              if f["overall"].get("brier_skill_score") is not None]

    per_symbol_auc: Dict[str, List[float]] = defaultdict(list)
    for f in wf["folds"]:
        for sym, m in f.get("per_symbol", {}).items():
            if m.get("roc_auc") is not None:
                per_symbol_auc[sym].append(m["roc_auc"])

    return {
        "mean_roc_auc": round(statistics.mean(aucs), 4) if aucs else None,
        "min_fold_auc": round(min(aucs), 4) if aucs else None,
        "max_fold_auc": round(max(aucs), 4) if aucs else None,
        "auc_spread": round(max(aucs) - min(aucs), 4) if len(aucs) > 1 else None,
        "folds_above_0.5": sum(1 for a in aucs if a > 0.5),
        "n_folds": len(aucs),
        "mean_pr_auc": round(statistics.mean(praucs), 4) if praucs else None,
        "mean_brier_skill": round(statistics.mean(briers), 4) if briers else None,
        "per_symbol_mean_auc": {
            s: round(statistics.mean(v), 4) for s, v in per_symbol_auc.items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candles", default=CANDLE_DIR)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join("models", "entry", "label_sweep.json"))
    args = ap.parse_args()

    from config import SYMBOLS
    symbols = args.symbols or list(SYMBOLS)

    print("Loading candles...")
    candles = {}
    for s in symbols:
        candles[s] = {"H4": trainer.load_candles(s, "H4", args.candles),
                      "H1": trainer.load_candles(s, "H1", args.candles)}
        print(f"  {s}: H4={len(candles[s]['H4'])} H1={len(candles[s]['H1'])}")

    print("\nBuilding features once (they do not depend on the label config)...")
    cache = build_feature_cache(candles)
    print(f"  {len(cache)} candidate bars")

    grid = QUICK_TP_SL if args.quick else TP_SL_GRID
    horizons = QUICK_HORIZONS if args.quick else HORIZONS
    total = len(grid) * len(horizons)
    print(f"\nSweeping {total} configurations "
          f"({len(grid)} TP/SL x {len(horizons)} horizons)...\n")

    header = (f"{'SL:TP':>9} {'H':>3} {'rows':>7} {'win%':>6} {'unres%':>7} "
              f"{'AUC':>7} {'spread':>7} {'>0.5':>5} {'PR-AUC':>7} {'sig':>4}")
    print(header)
    print("-" * len(header))

    results = []
    for sl_mult, tp_mult in grid:
        for horizon in horizons:
            X, y, meta, attempts, unresolved = label_dataset(
                cache, candles, sl_mult, tp_mult, horizon)
            if len(y) < 500:
                print(f"{sl_mult:>4}:{tp_mult:<4} {horizon:>3} "
                      f"{len(y):>7}  (too few rows, skipped)")
                continue

            win_rate = sum(y) / len(y)
            breakeven = sl_mult / (sl_mult + tp_mult)
            signal = conditional_signal(X, y)
            ev = evaluate(X, y, meta, args.folds)

            row = {
                "sl_mult": sl_mult, "tp_mult": tp_mult, "horizon": horizon,
                "rows": len(y),
                "win_rate": round(win_rate, 4),
                "breakeven_win_rate": round(breakeven, 4),
                "edge_vs_breakeven": round(win_rate - breakeven, 4),
                "unresolved_pct": round(unresolved / max(attempts, 1), 4),
                "per_symbol": {},
                "signal": signal,
                **ev,
            }
            by_sym: Dict[str, List[float]] = defaultdict(list)
            for label, m in zip(y, meta):
                by_sym[m["symbol"]].append(label)
            row["per_symbol"] = {
                s: {"n": len(v), "win_rate": round(sum(v) / len(v), 4)}
                for s, v in by_sym.items()
            }
            results.append(row)

            print(f"{sl_mult:>4}:{tp_mult:<4} {horizon:>3} {len(y):>7} "
                  f"{win_rate*100:>5.1f}% {row['unresolved_pct']*100:>6.1f}% "
                  f"{str(ev.get('mean_roc_auc')):>7} "
                  f"{str(ev.get('auc_spread')):>7} "
                  f"{ev.get('folds_above_0.5', 0)}/{ev.get('n_folds', 0):<3} "
                  f"{str(ev.get('mean_pr_auc')):>7} "
                  f"{signal['buckets_beyond_3_sigma']:>4}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(),
                   "symbols": symbols, "results": results}, fh, indent=2)

    print(f"\nFull results -> {args.out}")

    if results:
        print("\n" + "=" * 70)
        print("READING THIS TABLE")
        print("=" * 70)
        print("  AUC       mean walk-forward ROC-AUC. 0.50 = coin flip.")
        print("  spread    max fold AUC - min fold AUC. A large spread with a")
        print("            mean near 0.5 is variance, not edge.")
        print("  >0.5      how many folds beat chance. 5/5 matters; 3/5 does not.")
        print("  sig       feature buckets whose win rate deviates from the base")
        print("            rate by more than 3 sigma. THIS IS THE KEY COLUMN:")
        print("            0 means no model can separate this label with these")
        print("            features, whatever the hyper-parameters.")
        print()
        best = max(results, key=lambda r: r["signal"]["buckets_beyond_3_sigma"])
        print(f"  Most learnable label config by that measure: "
              f"SL {best['sl_mult']} TP {best['tp_mult']} horizon {best['horizon']} "
              f"-> {best['signal']['buckets_beyond_3_sigma']} significant buckets, "
              f"AUC {best.get('mean_roc_auc')}")
        if best["signal"]["buckets_beyond_3_sigma"] == 0:
            print("\n  NO configuration produced a single significant bucket.")
            print("  The bottleneck is the FEATURES, not the labelling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
