"""Do the SR_Mapping_NN feature ideas carry signal on KAIROS's own data?

This is a feasibility test, not a training run. It answers one question about
XAUUSD — whether support/resistance structure, candle anatomy and fractal age
add information the existing ten features do not — and answers it with the
same apparatus that produced the RED verdict in FEASIBILITY_REPORT.md:
stratified AUC, block-permutation nulls, decorrelation-aware effective sample
size, direction-free evaluation throughout.

Why not just port the reference model
--------------------------------------
Because its own artifacts show it does not work. Measured directly from
https://github.com/Mrizalfahlepi/SR_Mapping_NN at commit c0115aa:

  * seven of its 26 features read two bars into the future (Williams Fractal
    centred on bar i uses bars i+1 and i+2) — reproduced and fixed in
    analysis/features/sr_structure_features.py
  * its headline "72.4% precision" comes from scanning 55 thresholds against
    y_test and reporting the winner: 21 of 29 signals, one-sided binomial
    p=0.017 before selection, p=0.95 after correcting for 55 tries
  * its probabilities span 0.4901 to 0.5116 — every prediction it has ever
    made lies inside a 0.02-wide band
  * its own equity report shows the filter turning +$1,946 into +$504

So the model is discarded and only the feature *ideas* are carried across, to
be judged here on KAIROS data against KAIROS's own noise floor.

Nothing is trained for production, nothing is installed, no Optuna. Holdout
isolation is verified before any number is reported.

Usage (Windows machine)::

    python scripts/evaluate_sr_features.py --verify-holdout-isolation
    python scripts/evaluate_sr_features.py --permutations 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.features import live_parity_features as lpf  # noqa: E402
from analysis.features import sr_structure_features as sr  # noqa: E402
from analysis.features import timeframe_alignment as ta  # noqa: E402
from analysis.models import entry_feature_spec as spec  # noqa: E402

import train_entry_model as trainer  # noqa: E402
import feasibility_gate as fg  # noqa: E402
import information_discovery as idisc  # noqa: E402

OUT_DIR = os.path.join("research", "sr_features")


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def build_sr_rows(candles_by_symbol, meta):
    """SR feature vectors aligned row-for-row with `meta`, None where unbuildable."""
    rows = []
    for row in meta:
        h4 = candles_by_symbol[row["symbol"]]["H4"]
        visible = ta.closed_slice(h4, "H4", row["t"])
        if len(visible) < trainer.WARMUP_BARS:
            rows.append(None)
            continue
        indicators = lpf.live_indicators(visible)
        if indicators is None:
            rows.append(None)
            continue
        atr = float(indicators["atr"])
        rows.append(sr.build_sr_features(visible, timestamp=row["t"], atr=atr))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", default=os.path.join("data", "historical"))
    parser.add_argument("--symbols", nargs="*", default=["XAUUSD"],
                        help="XAUUSD only by default — prove it there first")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--research-frac", type=float, default=0.70)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--verify-holdout-isolation", action="store_true")
    parser.add_argument("--json", default=os.path.join(OUT_DIR, "sr_feature_audit.json"))
    args = parser.parse_args()

    rng = np.random.default_rng(20260817)

    provenance = trainer.check_provenance(args.candles)
    if not provenance["real"]:
        print(f"REFUSING: {provenance['reason']}")
        return 1

    candles = {s: {tf: trainer.load_candles(s, tf, args.candles)
                   for tf in ("H4", "H1")} for s in args.symbols}

    if not idisc.verify_holdout_isolation(candles, args.horizon, args.research_frac):
        return 1
    if args.verify_holdout_isolation:
        print("\nHoldout isolation verified. Exiting.")
        return 0

    head("1. DATASET")
    X_old, y, meta, holdout_n = idisc.build_research_rows(
        candles, args.horizon, args.research_frac)
    if len(y) < 500:
        print(f"  only {len(y)} research rows — too few to evaluate")
        return 1
    X_old = np.asarray(X_old, dtype=float)
    times = [m["t"] for m in meta]
    print(f"  symbols        : {args.symbols}")
    print(f"  research rows  : {len(y)}")
    print(f"  holdout rows   : {holdout_n}  (UNREAD)")
    print(f"  win rate       : {float(np.mean(y)):.4f}")
    print(f"  SL/TP          : {trainer.ATR_SL_BASE_MULTIPLIER} / "
          f"{trainer.ATR_TP_BASE_MULTIPLIER} x ATR  (KAIROS production values)")

    head("2. SR FEATURE AVAILABILITY")
    sr_rows = build_sr_rows(candles, meta)
    available = [i for i, r in enumerate(sr_rows) if r is not None]
    print(f"  built for {len(available)}/{len(meta)} research rows")
    if len(available) < 500:
        print("  too few — aborting")
        return 1

    names_sr = list(sr.FEATURE_NAMES)
    X_sr = np.asarray([[sr_rows[i][n] for n in names_sr] for i in available], dtype=float)
    X_old_sub = X_old[available]
    y_sub = np.asarray([y[i] for i in available], dtype=float)
    meta_sub = [meta[i] for i in available]
    times_sub = [times[i] for i in available]
    strata = [(m["symbol"], m["direction"]) for m in meta_sub]

    head("3. INFORMATION AUDIT (stratified — base-rate effects removed)")
    lag = fg.decorrelation_length(y_sub.tolist(), [m["symbol"] for m in meta_sub])
    n_eff = len(y_sub) / max(lag, 1)
    print(f"  decorrelation length {lag} rows -> effective n ~{n_eff:.0f} "
          f"of {len(y_sub)} nominal\n")

    print(f"  {'feature':24s} {'unique':>7} {'std':>10} {'strat AUC':>10}")
    audit = {}
    for j, name in enumerate(names_sr):
        col = X_sr[:, j]
        value = fg.stratified_auc(y_sub, col, strata)
        audit[name] = {
            "unique": int(len(np.unique(col))),
            "std": round(float(col.std()), 6),
            "stratified_auc": None if math.isnan(value) else round(value, 4),
        }
        print(f"  {name:24s} {len(np.unique(col)):7d} {col.std():10.4f} "
              f"{(value if not math.isnan(value) else float('nan')):10.4f}")

    block = max(2 * args.horizon, lag)
    owner, observed = fg.best_stratified_deviation(y_sub, X_sr, strata, names_sr)
    null = [fg.best_stratified_deviation(
        fg.block_permute(y_sub, block, rng), X_sr, strata, names_sr)[1]
        for _ in range(args.permutations)]
    pct = fg.permutation_percentile(observed, null)
    beats = pct >= 0.95
    print(f"\n  best stratified deviation: {owner} = {observed:.4f}")
    print(f"  null (block={block}, {args.permutations} perms): "
          f"median {statistics.median(null):.4f}, "
          f"p95 {sorted(null)[int(0.95 * len(null))]:.4f}")
    print(f"  percentile {pct:.1%}  ->  {'BEATS' if beats else 'NO EVIDENCE:'} "
          f"the noise floor")

    head("4. OLD vs SR vs OLD+SR (direction-free, walk-forward)")
    dir_index = spec.FEATURE_NAMES.index("direction")
    old_nodir = np.delete(X_old_sub, dir_index, axis=1)
    combined = np.concatenate([old_nodir, X_sr], axis=1)

    comparison = {}
    print(f"  {'view':34s} {'nf':>3} {'mean AUC':>9} {'folds':>6} {'spread':>8}")
    for label, matrix in [
        ("OLD FEATURES ONLY (direction-free)", old_nodir),
        ("SR FEATURES ONLY", X_sr),
        ("OLD + SR (direction-free)", combined),
    ]:
        scores = fg.walk_forward_scores(matrix, y_sub, times_sub,
                                        fg.logistic_factory, args.folds)
        if not scores:
            comparison[label] = None
            print(f"  {label:34s} {matrix.shape[1]:3d}      n/a")
            continue
        comparison[label] = {
            "mean": round(statistics.mean(scores), 4), "folds": len(scores),
            "spread": round(max(scores) - min(scores), 4),
            "per_fold": [round(s, 4) for s in scores],
            "folds_above_half": sum(1 for s in scores if s > 0.5),
        }
        print(f"  {label:34s} {matrix.shape[1]:3d} {statistics.mean(scores):9.4f} "
              f"{len(scores):6d} {max(scores) - min(scores):8.4f} "
              f"{[round(s, 3) for s in scores]}")

    # A small nonlinear model too: the reference used XGBoost, and a linear
    # probe alone would not settle whether these features interact.
    forest = fg.walk_forward_scores(X_sr, y_sub, times_sub,
                                    fg.small_forest_factory, args.folds)
    if forest:
        comparison["SR FEATURES ONLY (small forest)"] = {
            "mean": round(statistics.mean(forest), 4),
            "spread": round(max(forest) - min(forest), 4),
            "per_fold": [round(s, 4) for s in forest],
        }
        print(f"  {'SR FEATURES ONLY (small forest)':34s} {X_sr.shape[1]:3d} "
              f"{statistics.mean(forest):9.4f} {len(forest):6d} "
              f"{max(forest) - min(forest):8.4f} {[round(s, 3) for s in forest]}")

    head("5. VERDICT")
    sr_only = comparison.get("SR FEATURES ONLY")
    old_only = comparison.get("OLD FEATURES ONLY (direction-free)")
    both = comparison.get("OLD + SR (direction-free)")

    stable = bool(sr_only and sr_only["spread"] < 0.10
                  and sr_only["folds_above_half"] >= sr_only["folds"] - 1)
    improves = bool(both and old_only and both["mean"] > old_only["mean"] + 0.01)
    best_model_auc = max(
        [v["mean"] for v in comparison.values() if v] or [0.5])

    if beats and stable and improves and best_model_auc >= 0.55:
        verdict = "PRODUCTION CANDIDATE"
    elif beats and (stable or improves):
        verdict = "NOT PRODUCTION READY"
    elif best_model_auc > 0.52 or beats:
        verdict = "NO EDGE"
    else:
        verdict = "FAILED"

    print(f"  beats permutation null        : {beats}")
    print(f"  SR-only AUC                   : {sr_only['mean'] if sr_only else 'n/a'}")
    print(f"  stable across folds           : {stable}")
    print(f"  OLD+SR improves over OLD alone: {improves}")
    print(f"  best cheap-model AUC          : {best_model_auc:.4f}")
    print(f"\n  VERDICT: {verdict}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": args.symbols, "horizon": args.horizon,
        "sl_multiplier": trainer.ATR_SL_BASE_MULTIPLIER,
        "tp_multiplier": trainer.ATR_TP_BASE_MULTIPLIER,
        "research_rows": len(y), "holdout_rows_unread": holdout_n,
        "sr_rows_built": len(available),
        "decorrelation_lag": lag, "effective_n": round(n_eff),
        "per_feature": audit,
        "permutation": {"owner": owner, "observed": round(observed, 4),
                        "null_median": round(statistics.median(null), 4),
                        "null_p95": round(sorted(null)[int(0.95 * len(null))], 4),
                        "percentile": round(pct, 4), "beats_noise": beats},
        "comparison": comparison,
        "verdict": verdict,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  wrote {args.json}")

    if verdict in {"FAILED", "NO EDGE"}:
        print("\n  No model will be trained. models/entry/entry_model.json unchanged.")
    else:
        print("\n  Next: a single fixed-hyperparameter XGBoost fit compared against")
        print("  the logistic baseline above. Still no Optuna.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
