"""Phase 7: add feature groups one at a time and measure what each one buys.

Adding thirty features at once and reporting the total is how overfitting
hides. Each group is added cumulatively and the *out-of-sample* delta is
measured, so a group that only helps in-sample shows up as a flat or negative
contribution and can be dropped.

Reported per step:

  ROC-AUC, PR-AUC lift over base rate, Brier skill, fold spread, folds above
  0.5, and per-symbol AUC.

The last two matter as much as the mean. A group that lifts the mean while
widening the fold spread has added variance, not signal; a group that helps one
symbol and hurts the other two has found something instrument-specific that
will not generalise.

Nothing is installed and no model is written. This produces evidence for the
promotion decision; it does not make it.

Usage::

    python scripts/evaluate_feature_groups.py
    python scripts/evaluate_feature_groups.py --sl 1.5 --tp 2.5 --horizon 24
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.features import entry_features as ef  # noqa: E402

import train_entry_model as trainer  # noqa: E402

# Cumulative build-up. Each step is the previous plus one group.
STEPS: List[List[str]] = [
    ["baseline"],
    ["baseline", "trend"],
    ["baseline", "trend", "volatility"],
    ["baseline", "trend", "volatility", "momentum"],
    ["baseline", "trend", "volatility", "momentum", "structure"],
    ["baseline", "trend", "volatility", "momentum", "structure", "mtf"],
]


def build_rows(candles_by_symbol, sl_mult, tp_mult, horizon):
    """All features for every group, plus the label, once.

    Computing the full 36-feature vector once and slicing per step is far
    cheaper than rebuilding per step, and guarantees every step sees exactly
    the same rows — otherwise a step's "gain" could just be a different sample.
    """
    all_names = ef.group_names()
    rows: List[Dict[str, Any]] = []

    prev_sl, prev_tp = trainer.ATR_SL_BASE_MULTIPLIER, trainer.ATR_TP_BASE_MULTIPLIER
    trainer.ATR_SL_BASE_MULTIPLIER = sl_mult
    trainer.ATR_TP_BASE_MULTIPLIER = tp_mult
    try:
        for symbol, tf in candles_by_symbol.items():
            h4, h1 = tf["H4"], tf["H1"]
            h1_times = [c["t"] for c in h1]

            for i in range(ef.MIN_BARS, len(h4) - horizon - 1):
                bar_t = h4[i]["t"]
                pos = bisect.bisect_right(h1_times, bar_t)
                if pos < ef.MIN_BARS:
                    continue

                from analysis.features import live_parity_features as lpf
                h4_ind = lpf.live_indicators(h4[: i + 1])
                if h4_ind is None:
                    continue
                atr = float(h4_ind["atr"])

                for direction in ("BUY", "SELL"):
                    outcome = trainer.simulate_trade(h4, i, direction, atr, horizon)
                    if outcome is None:
                        continue
                    named = ef.build_entry_features(
                        h4[: i + 1], h1[:pos],
                        direction=direction, timestamp=bar_t)
                    if named is None:
                        continue
                    rows.append({
                        "x": [float(named[n]) for n in all_names],
                        "y": outcome["label"],
                        "meta": {"symbol": symbol, "t": bar_t,
                                 "direction": direction,
                                 "regime": named["market_regime"],
                                 "reason": outcome["reason"]},
                    })
    finally:
        trainer.ATR_SL_BASE_MULTIPLIER = prev_sl
        trainer.ATR_TP_BASE_MULTIPLIER = prev_tp

    rows.sort(key=lambda r: r["meta"]["t"])
    return rows, all_names


def evaluate_step(rows, all_names, groups, folds):
    """Walk-forward on just this step's columns."""
    wanted = ef.group_names(groups)
    idx = [all_names.index(n) for n in wanted]

    X = [[r["x"][k] for k in idx] for r in rows]
    y = [r["y"] for r in rows]
    meta = [r["meta"] for r in rows]

    # walk_forward names its columns from the spec; patch to this step's names
    # so feature importance and any error message stay readable.
    from analysis.models import entry_feature_spec as spec
    prev = spec.FEATURE_NAMES
    spec.FEATURE_NAMES = tuple(wanted)
    try:
        wf = trainer.walk_forward(X, y, meta, folds=folds)
    finally:
        spec.FEATURE_NAMES = prev

    if wf.get("skipped"):
        return {"skipped": wf["skipped"]}

    aucs, praucs, briers, bases = [], [], [], []
    per_symbol = defaultdict(list)
    for f in wf["folds"]:
        o = f["overall"]
        if o["roc_auc"] is not None:
            aucs.append(o["roc_auc"])
        if o["pr_auc"] is not None:
            praucs.append(o["pr_auc"])
            bases.append(o["base_win_rate"])
        if o.get("brier_skill_score") is not None:
            briers.append(o["brier_skill_score"])
        for s, m in f.get("per_symbol", {}).items():
            if m.get("roc_auc") is not None:
                per_symbol[s].append(m["roc_auc"])

    return {
        "n_features": len(wanted),
        "rows": len(y),
        "mean_roc_auc": round(statistics.mean(aucs), 4) if aucs else None,
        "fold_spread": round(max(aucs) - min(aucs), 4) if len(aucs) > 1 else None,
        "folds_above_half": sum(1 for a in aucs if a > 0.5),
        "n_folds": len(aucs),
        "mean_pr_auc": round(statistics.mean(praucs), 4) if praucs else None,
        "pr_auc_lift": (round(statistics.mean(praucs) - statistics.mean(bases), 4)
                        if praucs else None),
        "mean_brier_skill": round(statistics.mean(briers), 4) if briers else None,
        "per_symbol_auc": {s: round(statistics.mean(v), 4)
                           for s, v in per_symbol.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candles", default=os.path.join("data", "historical"))
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--sl", type=float, default=1.5)
    ap.add_argument("--tp", type=float, default=2.5)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join("models", "entry", "feature_groups.json"))
    args = ap.parse_args()

    from config import SYMBOLS
    symbols = args.symbols or list(SYMBOLS)

    print("Loading candles...")
    candles = {}
    for s in symbols:
        candles[s] = {"H4": trainer.load_candles(s, "H4", args.candles),
                      "H1": trainer.load_candles(s, "H1", args.candles)}
        print(f"  {s}: H4={len(candles[s]['H4'])} H1={len(candles[s]['H1'])}")

    print(f"\nLabel config: SL {args.sl} TP {args.tp} horizon {args.horizon}")
    print(f"Building all {len(ef.group_names())} features once "
          f"(this is the slow part)...")
    rows, all_names = build_rows(candles, args.sl, args.tp, args.horizon)
    print(f"  {len(rows)} rows")
    if len(rows) < 1000:
        print("  too few rows to evaluate")
        return 1

    wins = sum(r["y"] for r in rows)
    print(f"  win rate {wins/len(rows):.4f}   "
          f"BUY {sum(1 for r in rows if r['meta']['direction']=='BUY')} / "
          f"SELL {sum(1 for r in rows if r['meta']['direction']=='SELL')}")

    header = (f"\n{'step':<44} {'nf':>3} {'AUC':>7} {'dAUC':>7} {'spread':>7} "
              f"{'>0.5':>5} {'PRlift':>7} {'Brier':>7}")
    print(header)
    print("-" * (len(header) - 1))

    results = []
    prev_auc = None
    for groups in STEPS:
        res = evaluate_step(rows, all_names, groups, args.folds)
        if res.get("skipped"):
            print(f"{'+'.join(groups):<44} skipped: {res['skipped']}")
            continue
        auc = res["mean_roc_auc"]
        delta = None if prev_auc is None else round(auc - prev_auc, 4)
        res["groups"] = groups
        res["delta_auc"] = delta
        results.append(res)

        print(f"{'+'.join(groups):<44} {res['n_features']:>3} "
              f"{auc:>7.4f} "
              f"{('     --' if delta is None else f'{delta:+7.4f}')} "
              f"{res['fold_spread']:>7.4f} "
              f"{res['folds_above_half']}/{res['n_folds']:<3} "
              f"{res['pr_auc_lift']:>+7.4f} "
              f"{res['mean_brier_skill']:>+7.4f}")
        prev_auc = auc

    print("\nPer-symbol AUC by step:")
    for res in results:
        print(f"  {'+'.join(res['groups']):<44} {res['per_symbol_auc']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(),
                   "label": {"sl": args.sl, "tp": args.tp, "horizon": args.horizon},
                   "steps": results}, fh, indent=2)
    print(f"\nFull results -> {args.out}")

    print("\n" + "=" * 72)
    print("HOW TO READ THIS")
    print("=" * 72)
    print("  dAUC     out-of-sample gain from adding that group. Under ~+0.01 is")
    print("           noise at this sample size — do not keep a group for it.")
    print("  spread   max fold AUC - min fold AUC. If a group raises the mean but")
    print("           widens the spread, it added variance, not signal.")
    print("  >0.5     folds beating chance. 5/5 is the bar; 3/5 is a coin.")
    print("  PRlift   PR-AUC above the base rate. This is the one that maps to")
    print("           'does filtering by p_win actually raise the win rate'.")
    print("  Brier    skill vs predicting the base rate. Negative = worse than")
    print("           saying 'every trade has the average chance'.")
    if results:
        best = max(results, key=lambda r: r["mean_roc_auc"])
        print(f"\n  Best step: {'+'.join(best['groups'])}  AUC {best['mean_roc_auc']}")
        print(f"  vs baseline alone: {results[0]['mean_roc_auc']}")
        gain = best["mean_roc_auc"] - results[0]["mean_roc_auc"]
        print(f"  total gain from all feature engineering: {gain:+.4f}")
        if best["mean_roc_auc"] < 0.55:
            print("\n  Even the best step is under 0.55. If that holds, the honest")
            print("  reading is that H4/H1 indicator features do not predict this")
            print("  label, and the promotion gate should keep rejecting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
