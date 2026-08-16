"""Pre-training feasibility gate: is there information here worth training on?

A clean pipeline and a predictive one are different claims. Phase 3 established
the first. This establishes — or refuses — the second, before any model fitting
larger than a few milliseconds.

The central question is not "what AUC can we reach" but "is 0.5109
distinguishable from noise at all". Those need different tools. AUC on
autocorrelated data has a much wider null distribution than the textbook
formula suggests: adjacent H4 decisions share ~96 of their 100-bar indicator
window, and overlapping horizons make consecutive labels dependent. A naive
binomial or Hanley-McNeil standard error is therefore far too narrow, and every
"significant" result read off one is suspect.

So the noise floor here is empirical: labels are permuted in contiguous BLOCKS
long enough to preserve their autocorrelation, and the statistic is recomputed
many times to build the null it must beat. A result inside that null is noise,
however good it looks against a textbook error bar.

Everything runs on the first `--research-frac` of the data in time order. The
remaining tail is never read — not for feature selection, not for horizon
choice, not for the verdict.

Usage (Windows machine)::

    python scripts/feasibility_gate.py
    python scripts/feasibility_gate.py --permutations 200 --json out.json

Trains no production model, writes no model, touches nothing live.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.features import timeframe_alignment as ta  # noqa: E402
from analysis.models import entry_feature_spec as spec  # noqa: E402

import train_entry_model as trainer  # noqa: E402

H4 = ta.duration("H4")

# Feature groups for the ablation. Names must match spec.FEATURE_NAMES.
FEATURE_GROUPS = {
    "momentum": ["rsi", "momentum_score"],
    "trend": ["trend_score", "trend_strength"],
    "volatility": ["atr", "volatility_score"],
    "regime": ["market_regime"],
    "session": ["session"],
    "direction": ["direction"],
    "price_derived": ["macd"],
}

HORIZONS = [4, 8, 12, 16, 20, 24]

results: dict = {}


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def sub(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Statistics that respect autocorrelation
# ---------------------------------------------------------------------------

def auc(y, score) -> float:
    """Tie-corrected Mann-Whitney AUC."""
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    pos = int((y == 1.0).sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return trainer._auc_with_ties(y, score, pos, neg)


def stratified_auc(y, score, strata) -> float:
    """AUC computed within strata and pooled, so a base-rate shift cannot pose
    as discrimination.

    This distinction decides the whole analysis. A binary column like
    `direction` gets an AUC away from 0.5 whenever its two levels simply have
    different win rates — SELL winning 35.5% against BUY's 30.3% is enough. But
    that is a constant, exploitable only by "always prefer SELL", and it says
    nothing about telling a good SELL from a bad one. Pooling within
    (symbol, direction) removes those offsets and leaves only the question that
    matters: does this feature rank outcomes *inside* a group where the base
    rate is already fixed?

    A feature that is constant within every stratum — `direction` itself — has
    no within-stratum ranking to measure and is correctly excluded.
    """
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    buckets = defaultdict(list)
    for i, key in enumerate(strata):
        buckets[key].append(i)

    numerator, denominator = 0.0, 0.0
    for idx in buckets.values():
        if len(idx) < 30:
            continue
        ys = y[idx]
        ss = score[idx]
        pos = int((ys == 1.0).sum())
        neg = len(ys) - pos
        if pos == 0 or neg == 0 or len(np.unique(ss)) < 2:
            continue
        value = trainer._auc_with_ties(ys, ss, pos, neg)
        if math.isnan(value):
            continue
        weight = pos * neg
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator else float("nan")


def best_stratified_deviation(y, X, strata, names):
    """Largest within-stratum discrimination across features, and its owner."""
    best_name, best_dev = None, 0.0
    for j, name in enumerate(names):
        value = stratified_auc(y, X[:, j], strata)
        if math.isnan(value):
            continue
        if abs(value - 0.5) > best_dev:
            best_name, best_dev = name, abs(value - 0.5)
    return best_name, best_dev


def decorrelation_length(y, groups, max_lag: int = 60) -> int:
    """Lag at which the label series stops correlating with itself.

    Consecutive rows are not independent observations: two trades opened one bar
    apart share almost all of the price path that decides them. This measures
    how far apart two rows must be before their labels are effectively
    unrelated, which is what sets the honest sample size.
    """
    by_group = defaultdict(list)
    for label, key in zip(y, groups):
        by_group[key].append(float(label))

    for lag in range(1, max_lag + 1):
        correlations = []
        for series in by_group.values():
            if len(series) <= lag + 30:
                continue
            a = np.asarray(series[:-lag])
            b = np.asarray(series[lag:])
            if a.std() == 0 or b.std() == 0:
                continue
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
        if correlations and abs(statistics.mean(correlations)) < 0.05:
            return lag
    return max_lag


def block_permute(y, block: int, rng) -> np.ndarray:
    """Shuffle labels in contiguous blocks, preserving their autocorrelation.

    A plain shuffle destroys the dependence between neighbouring labels and so
    produces a null that is far too tight — which is exactly how autocorrelated
    financial data manufactures significance.
    """
    y = np.asarray(y)
    n = len(y)
    blocks = [y[i:i + block] for i in range(0, n, block)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])[:n]


def permutation_percentile(observed: float, null: list) -> float:
    """Where the observed statistic falls in its own null distribution."""
    if not null:
        return float("nan")
    return float(sum(1 for v in null if v <= observed) / len(null))


# ---------------------------------------------------------------------------
# Cheap models
# ---------------------------------------------------------------------------

def walk_forward_scores(X, y, timestamps, model_factory, folds: int = 4):
    """Expanding-window walk-forward. Returns per-fold AUC."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(timestamps, kind="stable")
    X, y = X[order], y[order]

    n = len(y)
    aucs = []
    for k in range(1, folds + 1):
        train_end = int(n * k / (folds + 1))
        test_end = int(n * (k + 1) / (folds + 1))
        if train_end < 200 or test_end - train_end < 100:
            continue
        xtr, ytr = X[:train_end], y[:train_end]
        xte, yte = X[train_end:test_end], y[train_end:test_end]
        if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
            continue
        try:
            model = model_factory()
            model.fit(xtr, ytr)
            proba = model.predict_proba(xte)[:, 1]
        except Exception:
            continue
        aucs.append(auc(yte, proba))
    return [a for a in aucs if not math.isnan(a)]


def logistic_factory():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=400, C=1.0))


def tiny_tree_factory():
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(max_depth=3, min_samples_leaf=200,
                                  random_state=0)


def small_forest_factory():
    """A deliberately small nonlinear model — the cheap stand-in for XGBoost.

    If this finds nothing, a tuned gradient booster on the same columns is
    unlikely to. Hyper-parameters move a score; they do not create information.
    """
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=120, max_depth=4,
                                  min_samples_leaf=100, n_jobs=-1,
                                  random_state=0)


# ---------------------------------------------------------------------------
# Alternative targets
# ---------------------------------------------------------------------------

def alternative_targets(h4, decision_idx: int, direction: str, atr: float,
                        horizon: int) -> dict:
    """Five economically distinct questions about the same future path.

    All share one entry: the open of the bar after the decision. They differ
    only in what counts as success, which is the point — if every framing is
    equally unpredictable, the target design is not what is limiting us.
    """
    entry_idx = decision_idx + 1
    if entry_idx + horizon >= len(h4) or atr <= 0:
        return {}

    entry = float(h4[entry_idx]["open"])
    sign = 1.0 if direction == "BUY" else -1.0
    window = h4[entry_idx:entry_idx + horizon]

    final = float(window[-1]["close"])
    forward = sign * (final - entry)

    favorable, adverse = 0.0, 0.0
    for candle in window:
        high, low = float(candle["high"]), float(candle["low"])
        best = sign * ((high - entry) if sign > 0 else (entry - low))
        worst = sign * ((low - entry) if sign > 0 else (entry - high))
        favorable = max(favorable, best)
        adverse = min(adverse, worst)

    # E: symmetric barrier — same distance either way, removing the 1.5:2.5
    # asymmetry so a 37% base rate is not itself the difficulty.
    symmetric = None
    for candle in window:
        high, low = float(candle["high"]), float(candle["low"])
        up = (high - entry) if sign > 0 else (entry - low)
        down = (entry - low) if sign > 0 else (high - entry)
        if up >= atr and down >= atr:
            symmetric = 0.0
            break
        if up >= atr:
            symmetric = 1.0
            break
        if down >= atr:
            symmetric = 0.0
            break

    return {
        "B_forward_sign": 1.0 if forward > 0 else 0.0,
        "C_forward_return_atr": forward / atr,
        "D_favorable_excursion_atr": favorable / atr,
        "E_symmetric_barrier": symmetric,
    }


# ---------------------------------------------------------------------------

def self_test(rng) -> int:
    """Prove the gate can find signal before trusting it when it finds none.

    A test that only ever says RED is as useless as one that only says GREEN.
    This plants a known effect on one column of pure noise and checks that the
    stratified statistic recovers it, names the right column, and stays quiet
    when nothing is planted. It also establishes the sensitivity floor: the
    smallest true effect this design can distinguish from its own null at this
    sample size.
    """
    head("SELF-TEST — can this gate detect signal that is really there?")

    n, n_features = 5000, 10
    names = [f"f{i}" for i in range(n_features)]
    strata = [("S", "BUY" if i % 2 else "SELL") for i in range(n)]

    print(f"  {'planted effect':30s} {'owner':>6} {'dev':>8} {'pct':>7} "
          f"{'logistic':>9}  verdict")
    detected_floor = None
    for strength, description in [(0.00, "none (control)"),
                                  (0.05, "very weak (true AUC ~0.514)"),
                                  (0.10, "weak (true AUC ~0.528)"),
                                  (0.20, "moderate (true AUC ~0.556)"),
                                  (0.40, "strong (true AUC ~0.611)")]:
        X = rng.normal(size=(n, n_features))
        probability = 1.0 / (1.0 + np.exp(-strength * X[:, 3]))
        y = (rng.random(n) < probability).astype(float)

        owner, dev = best_stratified_deviation(y, X, strata, names)
        null = [best_stratified_deviation(block_permute(y, 48, rng), X, strata, names)[1]
                for _ in range(60)]
        pct = permutation_percentile(dev, null)
        scores = walk_forward_scores(X, y, list(range(n)), logistic_factory, 4)
        found = pct >= 0.95
        if found and strength > 0 and detected_floor is None:
            detected_floor = strength
        print(f"  {description:30s} {str(owner):>6} {dev:8.4f} {pct:6.1%} "
              f"{statistics.mean(scores):9.4f}  "
              f"{'DETECTED' if found else 'not detected'}"
              f"{' (correct column)' if found and owner == 'f3' else ''}")

    print("\n  Reading: the control must NOT be detected, and the moderate and")
    print("  strong effects must be, on the correct column. The smallest")
    print("  detected effect is this gate's sensitivity floor — a real edge")
    print("  weaker than that would not be visible at this sample size, and")
    print("  neither would it be tradeable.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="verify the gate can detect planted signal, then exit")
    parser.add_argument("--candles", default=os.path.join("data", "historical"))
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--research-frac", type=float, default=0.70,
                        help="fraction of history, in time order, that may be analysed")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(20260816)
    if args.self_test:
        return self_test(rng)

    from config import SYMBOLS
    symbols = args.symbols or list(SYMBOLS)

    head("0. SETUP")
    provenance = trainer.check_provenance(args.candles)
    if not provenance["real"]:
        print(f"  REFUSING: {provenance['reason']}")
        return 1
    print(f"  provenance ok, fetched {provenance['manifest'].get('fetched_at')}")

    candles = {s: {tf: trainer.load_candles(s, tf, args.candles)
                   for tf in ("H4", "H1")} for s in symbols}

    print(f"  building features once at horizon {args.horizon}...")
    X_all, y_all, meta_all = trainer.build_dataset(candles, args.horizon)
    if not X_all:
        print("  no rows built")
        return 1

    order = sorted(range(len(y_all)), key=lambda i: meta_all[i]["t"])
    X_all = [X_all[i] for i in order]
    y_all = [y_all[i] for i in order]
    meta_all = [meta_all[i] for i in order]

    cut = int(len(y_all) * args.research_frac)
    X, y, meta = X_all[:cut], y_all[:cut], meta_all[:cut]
    holdout_n = len(y_all) - cut

    print(f"  {len(y_all)} rows total")
    print(f"  RESEARCH  {len(y)} rows  "
          f"{datetime.fromtimestamp(meta[0]['t'], tz=timezone.utc):%Y-%m-%d} -> "
          f"{datetime.fromtimestamp(meta[-1]['t'], tz=timezone.utc):%Y-%m-%d}")
    print(f"  HOLDOUT   {holdout_n} rows — NOT READ by anything below")

    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    times = [m["t"] for m in meta]
    names = list(spec.FEATURE_NAMES)

    # ---------------------------------------------------------------- 1
    head("1. TARGET DIAGNOSTICS")

    base_rate = float(y_arr.mean())
    print(f"  class balance: {base_rate:.4f} win / {1 - base_rate:.4f} loss "
          f"({int(y_arr.sum())} / {len(y_arr) - int(y_arr.sum())})")

    lag = decorrelation_length(y, [m["symbol"] for m in meta])
    n_eff = len(y) / max(lag, 1)
    print(f"  label decorrelation length: {lag} rows")
    print(f"  effective sample size: ~{n_eff:.0f} of {len(y)} nominal rows "
          f"({n_eff / len(y):.1%})")
    print(f"    -> AUC noise floor is ~{0.5 / math.sqrt(max(n_eff, 1)):.4f} wide, "
          f"not the {0.5 / math.sqrt(len(y)):.4f} the row count suggests")
    results["decorrelation_lag"] = lag
    results["n_effective"] = round(n_eff)

    sub("run lengths of identical outcomes")
    runs, current, previous = [], 0, None
    for label in y:
        if label == previous:
            current += 1
        else:
            if current:
                runs.append(current)
            current, previous = 1, label
    runs.append(current)
    print(f"  mean run {statistics.mean(runs):.2f}, max {max(runs)}, "
          f"expected under independence ~{1 / (1 - base_rate * base_rate - (1 - base_rate) ** 2):.2f}")

    sub("conditional win rates (deviation from base rate)")
    for field, label in [("symbol", "SYMBOL"), ("direction", "DIRECTION"),
                         ("regime", "REGIME"), ("session", "SESSION")]:
        buckets = defaultdict(list)
        for value, info in zip(y, meta):
            buckets[info[field]].append(value)
        print(f"  {label}")
        for key, values in sorted(buckets.items(), key=lambda kv: str(kv[0])):
            rate = statistics.mean(values)
            # Noise band from the EFFECTIVE sample in this bucket, not its
            # nominal size — the correction that made the earlier `sig` column
            # overstate significance by roughly 4.5x.
            eff = len(values) / max(lag, 1)
            band = 1.96 * math.sqrt(max(base_rate * (1 - base_rate) / max(eff, 1), 0))
            flag = "  <-- outside noise" if abs(rate - base_rate) > band else ""
            print(f"    {str(key):18s} n={len(values):6d}  win={rate:.4f}  "
                  f"delta={rate - base_rate:+.4f}  band=+/-{band:.4f}{flag}")

    # ---------------------------------------------------------------- 2
    head("2. FEATURE INFORMATION AUDIT")

    print(f"  {'feature':18s} {'unique':>7} {'std':>10} {'autocorr':>9} "
          f"{'AUC':>7} {'|dev|':>7} {'stability':>9}")
    blocks = 4
    block_size = len(y) // blocks
    feature_stats = {}
    for j, name in enumerate(names):
        column = X_arr[:, j]
        uniq = len(np.unique(column))
        std = float(column.std())
        if std > 0 and len(column) > 10:
            autocorr = float(np.corrcoef(column[:-1], column[1:])[0, 1])
        else:
            autocorr = float("nan")
        univariate = auc(y_arr, column)
        deviation = abs(univariate - 0.5) if not math.isnan(univariate) else float("nan")

        # Temporal stability: does the direction of the relationship hold in
        # every quarter of the research window, or does it flip?
        per_block = []
        for b in range(blocks):
            lo, hi = b * block_size, (b + 1) * block_size
            block_auc = auc(y_arr[lo:hi], column[lo:hi])
            if not math.isnan(block_auc):
                per_block.append(block_auc)
        if per_block:
            same_side = sum(1 for a in per_block if (a - 0.5) * (univariate - 0.5) > 0)
            stability = f"{same_side}/{len(per_block)}"
        else:
            stability = "n/a"

        feature_stats[name] = {
            "unique": uniq, "std": round(std, 6),
            "autocorr": None if math.isnan(autocorr) else round(autocorr, 4),
            "auc": None if math.isnan(univariate) else round(univariate, 4),
            "deviation": None if math.isnan(deviation) else round(deviation, 4),
            "block_aucs": [round(a, 4) for a in per_block],
            "stability": stability,
        }
        print(f"  {name:18s} {uniq:7d} {std:10.4f} {autocorr:9.4f} "
              f"{univariate:7.4f} {deviation:7.4f} {stability:>9}")
    results["features"] = feature_stats

    sub("mutual information with the target (bits, higher = more informative)")
    try:
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(X_arr, y_arr, discrete_features=False,
                                 random_state=0)
        for name, value in sorted(zip(names, mi), key=lambda kv: -kv[1]):
            print(f"    {name:18s} {value:.6f}")
        results["mutual_information"] = {n: round(float(v), 6)
                                         for n, v in zip(names, mi)}
    except Exception as exc:
        print(f"    unavailable: {exc}")

    sub("base-rate effects, separated out before anything is called signal")
    raw_best_name, raw_best_dev = None, 0.0
    for j, name in enumerate(names):
        value = auc(y_arr, X_arr[:, j])
        if not math.isnan(value) and abs(value - 0.5) > raw_best_dev:
            raw_best_name, raw_best_dev = name, abs(value - 0.5)
    print(f"  best POOLED univariate: {raw_best_name} |AUC-0.5|={raw_best_dev:.4f}")
    print("  A pooled AUC on a low-cardinality column is mostly a base-rate")
    print("  difference between its levels, not discrimination within them. The")
    print("  headline test below is stratified by (symbol, direction), which")
    print("  removes those offsets — a feature that only shifts the base rate")
    print("  cannot score there, and `direction` itself is excluded by")
    print("  construction because it is constant inside a stratum.")

    strata = [(m["symbol"], m["direction"]) for m in meta]
    results["base_rate_effect"] = {"feature": raw_best_name,
                                   "pooled_deviation": round(raw_best_dev, 4)}

    sub("BEST STRATIFIED univariate signal vs its block-permutation null")
    best_name, observed_best = best_stratified_deviation(y_arr, X_arr, strata, names)
    print(f"  owner of the best stratified signal: {best_name}")
    block = max(2 * args.horizon, lag)
    null = []
    for _ in range(args.permutations):
        shuffled = block_permute(y_arr, block, rng)
        _, dev = best_stratified_deviation(shuffled, X_arr, strata, names)
        null.append(dev)
    pct = permutation_percentile(observed_best, null)
    print(f"  observed best |AUC-0.5| = {observed_best:.4f}")
    print(f"  null (block={block}, {args.permutations} permutations): "
          f"median {statistics.median(null):.4f}, "
          f"p95 {sorted(null)[int(0.95 * len(null))]:.4f}, "
          f"max {max(null):.4f}")
    print(f"  percentile of observed within null: {pct:.1%}")
    print(f"  -> {'BEATS' if pct >= 0.95 else 'DOES NOT BEAT'} the noise floor")
    results["univariate_permutation"] = {
        "best_feature": best_name,
        "observed": round(observed_best, 4),
        "null_median": round(statistics.median(null), 4),
        "null_p95": round(sorted(null)[int(0.95 * len(null))], 4),
        "percentile": round(pct, 4),
        "beats_noise": bool(pct >= 0.95),
    }

    # ---------------------------------------------------------------- 3
    head("3. CONDITIONAL SIGNAL TEST")
    print("  Best univariate |AUC-0.5| within each slice, with the slice's own")
    print("  effective-sample noise band. A slice beating its band is a lead;")
    print("  a slice inside it is noise however large the number looks.\n")

    conditional = {}
    slices = [("ALL", lambda m: True)]
    slices += [(f"dir={d}", (lambda d: lambda m: m["direction"] == d)(d))
               for d in sorted({m["direction"] for m in meta})]
    slices += [(f"sym={s}", (lambda s: lambda m: m["symbol"] == s)(s))
               for s in sorted({m["symbol"] for m in meta})]
    slices += [(f"regime={r}", (lambda r: lambda m: m["regime"] == r)(r))
               for r in sorted({m["regime"] for m in meta})]
    slices += [(f"session={s}", (lambda s: lambda m: m["session"] == s)(s))
               for s in sorted({m["session"] for m in meta})]

    print(f"  {'slice':22s} {'n':>7} {'n_eff':>7} {'win':>7} {'best feat':18s} "
          f"{'|dev|':>7} {'band':>7}")
    for label, predicate in slices:
        idx = [i for i, m in enumerate(meta) if predicate(m)]
        if len(idx) < 300:
            continue
        ys = y_arr[idx]
        if len(set(ys.tolist())) < 2:
            continue
        sub_strata = [strata[i] for i in idx]
        slice_best, best_dev = best_stratified_deviation(
            ys, X_arr[idx], sub_strata, names)
        best_name = slice_best
        eff = len(idx) / max(lag, 1)
        # Noise band for the MAXIMUM over 10 features, not for one: taking a
        # max inflates the expected deviation, so the threshold must too.
        band = 2.6 * 0.5 / math.sqrt(max(eff, 1))
        flag = "  <-- lead" if best_dev > band else ""
        conditional[label] = {"n": len(idx), "n_eff": round(eff),
                              "win_rate": round(float(ys.mean()), 4),
                              "best_feature": best_name,
                              "deviation": round(best_dev, 4),
                              "band": round(band, 4),
                              "above_band": bool(best_dev > band)}
        print(f"  {label:22s} {len(idx):7d} {eff:7.0f} {ys.mean():7.4f} "
              f"{str(best_name):18s} {best_dev:7.4f} {band:7.4f}{flag}")
    results["conditional"] = conditional

    # ---------------------------------------------------------------- 4
    head("4. SIMPLE BASELINES (walk-forward within research only)")

    baselines = {}
    print(f"  {'model':28s} {'mean AUC':>9} {'folds':>7} {'spread':>8} {'per fold'}")

    for label, factory in [("logistic regression", logistic_factory),
                           ("decision tree (depth 3)", tiny_tree_factory),
                           ("small forest (depth 4)", small_forest_factory)]:
        scores = walk_forward_scores(X_arr, y_arr, times, factory, args.folds)
        if not scores:
            print(f"  {label:28s} {'n/a':>9}")
            continue
        spread = max(scores) - min(scores)
        baselines[label] = {"mean": round(statistics.mean(scores), 4),
                            "folds": len(scores),
                            "spread": round(spread, 4),
                            "per_fold": [round(s, 4) for s in scores]}
        print(f"  {label:28s} {statistics.mean(scores):9.4f} {len(scores):7d} "
              f"{spread:8.4f} {[round(s, 3) for s in scores]}")

    # The decisive comparison. `direction` is not a predictor of trade quality;
    # it is the side being evaluated. If BUY and SELL simply have different base
    # rates over the sample — as they do whenever the sample trends — a model
    # scores above 0.5 by learning "prefer BUY", which is a bet on the period,
    # not a filter on the signal. Removing the column answers whether anything
    # else is doing work.
    dir_index = names.index("direction")
    keep = [j for j in range(len(names)) if j != dir_index]
    X_nodir = X_arr[:, keep]

    nodir_scores = walk_forward_scores(X_nodir, y_arr, times, logistic_factory,
                                       args.folds)
    if nodir_scores:
        print(f"  {'logistic WITHOUT direction':28s} {statistics.mean(nodir_scores):9.4f} "
              f"{len(nodir_scores):7d} "
              f"{max(nodir_scores) - min(nodir_scores):8.4f} "
              f"{[round(s, 3) for s in nodir_scores]}")
        baselines["logistic without direction"] = {
            "mean": round(statistics.mean(nodir_scores), 4),
            "folds": len(nodir_scores),
            "spread": round(max(nodir_scores) - min(nodir_scores), 4),
            "per_fold": [round(s, 4) for s in nodir_scores]}

    print(f"\n  constant predictor           {0.5:9.4f}  (by definition)")
    print(f"  majority class accuracy      {max(base_rate, 1 - base_rate):9.4f}  "
          f"(AUC 0.5 — accuracy is not evidence of signal here)")
    results["baselines"] = baselines

    sub("DIRECTION-FREE logistic walk-forward vs its block-permutation null")
    print("  Run without `direction`, because a base-rate difference between")
    print("  BUY and SELL would otherwise beat any null while telling us")
    print("  nothing about distinguishing a good trade from a bad one.\n")
    observed = (statistics.mean(nodir_scores) if nodir_scores else 0.5)
    perms = max(20, args.permutations // 8)
    null_model = []
    for _ in range(perms):
        shuffled = block_permute(y_arr, block, rng)
        scores = walk_forward_scores(X_nodir, shuffled, times, logistic_factory,
                                     args.folds)
        if scores:
            null_model.append(statistics.mean(scores))
    if null_model:
        pct_model = permutation_percentile(observed, null_model)
        print(f"  observed  {observed:.4f}")
        print(f"  null      median {statistics.median(null_model):.4f}, "
              f"p95 {sorted(null_model)[int(0.95 * len(null_model))]:.4f}, "
              f"max {max(null_model):.4f}  ({perms} permutations)")
        print(f"  percentile of observed: {pct_model:.1%}")
        print(f"  -> {'BEATS' if pct_model >= 0.95 else 'DOES NOT BEAT'} the noise floor")
        results["model_permutation"] = {
            "observed": round(observed, 4),
            "null_median": round(statistics.median(null_model), 4),
            "null_p95": round(sorted(null_model)[int(0.95 * len(null_model))], 4),
            "percentile": round(pct_model, 4),
            "beats_noise": bool(pct_model >= 0.95),
        }

    # ---------------------------------------------------------------- 5
    head("5. FEATURE GROUP ABLATION")
    print(f"  {'group':16s} {'features':>8} {'mean AUC':>9} {'spread':>8} {'per fold'}")

    groups_result = {}
    for group, members in FEATURE_GROUPS.items():
        cols = [names.index(m) for m in members if m in names]
        if not cols:
            continue
        scores = walk_forward_scores(X_arr[:, cols], y_arr, times,
                                     logistic_factory, args.folds)
        if not scores:
            continue
        groups_result[group] = {"mean": round(statistics.mean(scores), 4),
                                "spread": round(max(scores) - min(scores), 4),
                                "per_fold": [round(s, 4) for s in scores]}
        print(f"  {group:16s} {len(cols):8d} {statistics.mean(scores):9.4f} "
              f"{max(scores) - min(scores):8.4f} {[round(s, 3) for s in scores]}")

    all_scores = walk_forward_scores(X_arr, y_arr, times, logistic_factory, args.folds)
    if all_scores:
        print(f"  {'ALL COMBINED':16s} {len(names):8d} "
              f"{statistics.mean(all_scores):9.4f} "
              f"{max(all_scores) - min(all_scores):8.4f} "
              f"{[round(s, 3) for s in all_scores]}")
    results["feature_groups"] = groups_result

    # ---------------------------------------------------------------- 6
    head("6. TARGET DESIGN AUDIT")
    print("  Same features, same entries, five different questions about the")
    print("  future. If every framing is equally unpredictable, the target")
    print("  definition is not the binding constraint.\n")

    index_of = {s: {float(c["t"]): i for i, c in enumerate(candles[s]["H4"])}
                for s in symbols}

    alt_rows = defaultdict(lambda: {"idx": [], "value": []})
    for row_i, info in enumerate(meta):
        symbol = info["symbol"]
        decision_open = info["t"] - H4
        i = index_of[symbol].get(decision_open)
        if i is None:
            continue
        h4 = candles[symbol]["H4"]
        visible = ta.closed_slice(h4, "H4", info["t"])
        from analysis.features import live_parity_features as lpf
        indicators = lpf.live_indicators(visible)
        if indicators is None:
            continue
        alts = alternative_targets(h4, i, info["direction"],
                                   float(indicators["atr"]), args.horizon)
        for key, value in alts.items():
            if value is None:
                continue
            alt_rows[key]["idx"].append(row_i)
            alt_rows[key]["value"].append(value)

    target_result = {"A_current_barrier": {}}
    scores = walk_forward_scores(X_arr, y_arr, times, logistic_factory, args.folds)
    target_result["A_current_barrier"] = {
        "n": len(y), "positive_rate": round(base_rate, 4),
        "logistic_auc": round(statistics.mean(scores), 4) if scores else None,
        "best_univariate_dev": round(observed_best, 4),
    }
    print(f"  {'target':28s} {'n':>7} {'pos rate':>9} {'logistic':>9} "
          f"{'strat dev':>14}  driver")
    print(f"  {'A current TP/SL barrier':28s} {len(y):7d} {base_rate:9.4f} "
          f"{(statistics.mean(scores) if scores else float('nan')):9.4f} "
          f"{observed_best:14.4f}  {best_name}")

    for key in ("B_forward_sign", "C_forward_return_atr",
                "D_favorable_excursion_atr", "E_symmetric_barrier"):
        entry = alt_rows.get(key)
        if not entry or len(entry["idx"]) < 500:
            continue
        idx = entry["idx"]
        raw = np.asarray(entry["value"], dtype=float)
        sub_X = X_arr[idx]
        sub_times = [times[i] for i in idx]

        if key == "C_forward_return_atr":
            binary = (raw > 0).astype(float)
        elif key == "D_favorable_excursion_atr":
            binary = (raw >= trainer.ATR_TP_BASE_MULTIPLIER).astype(float)
        else:
            binary = raw.astype(float)

        if len(set(binary.tolist())) < 2:
            continue
        sub_strata = [strata[i] for i in idx]
        driver, dev = best_stratified_deviation(binary, sub_X, sub_strata, names)
        sc = walk_forward_scores(sub_X, binary, sub_times, logistic_factory, args.folds)

        # A target defined in ATR units and "predicted" by an ATR feature is a
        # tautology, not an edge: "will price travel 2.5 ATR" is largely the
        # question "is volatility high", and ATR already answers it. Such a
        # target is flagged rather than counted as promise.
        tautological = driver in {"atr", "volatility_score"} and "atr" in key
        target_result[key] = {
            "n": len(idx), "positive_rate": round(float(binary.mean()), 4),
            "logistic_auc": round(statistics.mean(sc), 4) if sc else None,
            "best_univariate_dev": round(dev, 4),
            "driver": driver,
            "tautological": bool(tautological),
        }
        flag = "  <-- driven by a volatility feature on a volatility-scaled target" \
            if tautological else ""
        print(f"  {key:28s} {len(idx):7d} {binary.mean():9.4f} "
              f"{(statistics.mean(sc) if sc else float('nan')):9.4f} {dev:14.4f}"
              f"  {str(driver)}{flag}")
    results["targets"] = target_result

    # ---------------------------------------------------------------- 7
    head("7. HORIZON SENSITIVITY")
    print("  Features are identical across horizons; only the label changes.")
    print("  Scored DIRECTION-FREE, so a horizon cannot look good merely because")
    print("  the BUY/SELL base rates diverge further at that distance.")
    print("  A rising-then-falling curve would mark a real timescale. A flat")
    print("  line near 0.50 marks structural absence.\n")
    print(f"  {'horizon':>8} {'rows':>7} {'win rate':>9} {'logistic':>9} "
          f"{'spread':>8} {'best univ dev':>14}")

    horizon_result = {}
    for horizon in HORIZONS:
        rows_X, rows_y, rows_t = [], [], []
        for row_i, info in enumerate(meta):
            symbol = info["symbol"]
            i = index_of[symbol].get(info["t"] - H4)
            if i is None:
                continue
            h4 = candles[symbol]["H4"]
            visible = ta.closed_slice(h4, "H4", info["t"])
            from analysis.features import live_parity_features as lpf
            indicators = lpf.live_indicators(visible)
            if indicators is None:
                continue
            outcome = trainer.simulate_trade(h4, i, info["direction"],
                                             float(indicators["atr"]), horizon)
            if outcome is None:
                continue
            rows_X.append(X_arr[row_i])
            rows_y.append(outcome["label"])
            rows_t.append(info["t"])
        if len(rows_y) < 500:
            continue
        arr_X = np.asarray(rows_X)
        arr_y = np.asarray(rows_y, dtype=float)
        sc = walk_forward_scores(arr_X[:, keep], arr_y, rows_t, logistic_factory,
                                 args.folds)
        dev = max(abs(auc(arr_y, arr_X[:, j]) - 0.5) for j in range(len(names))
                  if not math.isnan(auc(arr_y, arr_X[:, j])))
        horizon_result[horizon] = {
            "rows": len(rows_y), "win_rate": round(float(arr_y.mean()), 4),
            "logistic_auc": round(statistics.mean(sc), 4) if sc else None,
            "spread": round(max(sc) - min(sc), 4) if sc and len(sc) > 1 else None,
            "best_univariate_dev": round(dev, 4),
        }
        print(f"  {horizon:8d} {len(rows_y):7d} {arr_y.mean():9.4f} "
              f"{(statistics.mean(sc) if sc else float('nan')):9.4f} "
              f"{(max(sc) - min(sc) if sc and len(sc) > 1 else float('nan')):8.4f} "
              f"{dev:14.4f}")
    results["horizons"] = horizon_result

    # ---------------------------------------------------------------- 8
    head("8. TIMEFRAME FEASIBILITY")
    print("  Only H4 and H1 were exported, so M15/M30 cannot be tested without")
    print("  new data. What CAN be measured is whether the resolution we have")
    print("  matches the thing being predicted.\n")

    holding = [m["bars"] for m in meta]
    quick = sum(1 for b in holding if b <= 2)
    print(f"  holding time: median {statistics.median(holding):.0f} H4 bars, "
          f"mean {statistics.mean(holding):.1f}")
    print(f"  resolved within 2 bars (8h): {quick} of {len(holding)} "
          f"({quick / len(holding):.1%})")
    if quick / len(holding) > 0.4:
        print("  -> a large share of outcomes is decided within 1-2 bars, while the")
        print("     features describe a 100-bar (~17 day) window. That is a")
        print("     resolution mismatch: the predictors are far slower than the")
        print("     event. Finer timeframes are worth testing.")
    else:
        print("  -> outcomes unfold over many bars, so H4/H1 resolution is not")
        print("     obviously mismatched to the target.")
    results["timeframe"] = {
        "median_holding_bars": statistics.median(holding),
        "resolved_within_2_bars": round(quick / len(holding), 4),
    }

    sub("how much of the H4 return is predictable from its own past")
    for symbol in symbols:
        h4 = candles[symbol]["H4"]
        closes = np.asarray([c["close"] for c in h4], dtype=float)
        returns = np.diff(np.log(closes))
        acs = [float(np.corrcoef(returns[:-k], returns[k:])[0, 1])
               for k in (1, 2, 3, 6, 12)]
        print(f"  {symbol:8s} return autocorr lag1/2/3/6/12: "
              f"{' '.join(f'{a:+.4f}' for a in acs)}")

    # ---------------------------------------------------------------- 9
    head("9. CROSS-SECTIONAL TEST (does pooling hide signal?) — direction-free")
    print(f"  {'scope':16s} {'rows':>7} {'mean AUC':>9} {'spread':>8} {'per fold'}")

    cross = {}
    if nodir_scores:
        cross["combined"] = {"rows": len(y),
                             "mean": round(statistics.mean(nodir_scores), 4),
                             "spread": round(max(nodir_scores) - min(nodir_scores), 4)}
        print(f"  {'combined':16s} {len(y):7d} {statistics.mean(nodir_scores):9.4f} "
              f"{max(nodir_scores) - min(nodir_scores):8.4f} "
              f"{[round(s, 3) for s in nodir_scores]}")
    for symbol in symbols:
        idx = [i for i, m in enumerate(meta) if m["symbol"] == symbol]
        if len(idx) < 500:
            continue
        sc = walk_forward_scores(X_nodir[idx], y_arr[idx],
                                 [times[i] for i in idx], logistic_factory, args.folds)
        if not sc:
            continue
        cross[symbol] = {"rows": len(idx), "mean": round(statistics.mean(sc), 4),
                         "spread": round(max(sc) - min(sc), 4),
                         "per_fold": [round(s, 4) for s in sc]}
        print(f"  {symbol:16s} {len(idx):7d} {statistics.mean(sc):9.4f} "
              f"{max(sc) - min(sc):8.4f} {[round(s, 3) for s in sc]}")
    results["cross_sectional"] = cross

    # ---------------------------------------------------------------- 10
    head("10. PERSISTENCE (is any of it stable?)")

    persistence = {}
    for label, predicate in [("BUY", lambda m: m["direction"] == "BUY"),
                             ("SELL", lambda m: m["direction"] == "SELL")]:
        idx = [i for i, m in enumerate(meta) if predicate(m)]
        if len(idx) < 500:
            continue
        sc = walk_forward_scores(X_nodir[idx], y_arr[idx], [times[i] for i in idx],
                                 logistic_factory, args.folds)
        if sc:
            persistence[label] = {"mean": round(statistics.mean(sc), 4),
                                  "folds_above_half": sum(1 for s in sc if s > 0.5),
                                  "n_folds": len(sc),
                                  "per_fold": [round(s, 4) for s in sc]}
            print(f"  {label:10s} mean {statistics.mean(sc):.4f}  "
                  f"folds>0.5 {sum(1 for s in sc if s > 0.5)}/{len(sc)}  "
                  f"{[round(s, 3) for s in sc]}")

    if nodir_scores:
        above = sum(1 for s in nodir_scores if s > 0.5)
        print(f"\n  direction-free folds above 0.5: {above}/{len(nodir_scores)}")
        print(f"  fold spread: {max(nodir_scores) - min(nodir_scores):.4f}")
        print(f"  a signal present in one fold only is noise until shown otherwise")
    results["persistence"] = persistence

    # ---------------------------------------------------------------- 11
    head("11. TRAINING FEASIBILITY SCORE")

    components = []

    def add(name: str, points: int, out_of: int, reason: str) -> None:
        components.append((name, points, out_of, reason))

    univ_beats = results["univariate_permutation"]["beats_noise"]
    add("univariate signal vs noise floor", 20 if univ_beats else 0, 20,
        f"best |AUC-0.5|={observed_best:.4f} sits at the "
        f"{results['univariate_permutation']['percentile']:.0%} percentile of its "
        f"block-permutation null")

    model_beats = results.get("model_permutation", {}).get("beats_noise", False)
    add("model signal vs noise floor (direction-free)", 20 if model_beats else 0, 20,
        f"direction-free logistic {results.get('model_permutation', {}).get('observed', 0):.4f} "
        f"at the {results.get('model_permutation', {}).get('percentile', 0):.0%} percentile")

    # Scored on the direction-free model for the same reason. A score built on
    # "BUY wins more often than SELL over this sample" is a directional bet on
    # the period, not an entry filter, and the rule layer has already chosen the
    # side by the time this gate runs.
    best_base = statistics.mean(nodir_scores) if nodir_scores else 0.5
    base_points = 15 if best_base >= 0.55 else (8 if best_base >= 0.52 else 0)
    add("cheap-model performance (direction-free)", base_points, 15,
        f"direction-free baseline reached {best_base:.4f}; with direction it "
        f"reached {max((v['mean'] for v in baselines.values()), default=0.5):.4f}")

    if nodir_scores:
        above = sum(1 for s in nodir_scores if s > 0.5)
        stab_points = 15 if above == len(nodir_scores) and best_base >= 0.52 else (
            7 if above >= len(nodir_scores) - 1 and best_base >= 0.52 else 0)
        add("temporal stability (direction-free)", stab_points, 15,
            f"{above}/{len(nodir_scores)} direction-free folds above 0.5, "
            f"spread {max(nodir_scores) - min(nodir_scores):.4f}")
    else:
        add("temporal stability (direction-free)", 0, 15, "no folds evaluated")

    # A high score on one instrument means nothing if its folds disagree —
    # that is the shape of noise, not of an instrument-specific edge.
    per_symbol = {k: v for k, v in cross.items() if k != "combined"}
    sym_means = [v["mean"] for v in per_symbol.values()]
    steady = [k for k, v in per_symbol.items()
              if v["mean"] > 0.54 and v.get("spread", 1.0) < 0.10]
    sym_points = 10 if sym_means and min(sym_means) > 0.52 else (5 if steady else 0)
    add("cross-symbol consistency", sym_points, 10,
        (f"per-symbol AUC range {min(sym_means):.4f}-{max(sym_means):.4f}; "
         f"stable-and-above-0.54: {steady or 'none'}")
        if sym_means else "not evaluated")

    dir_means = [v["mean"] for v in persistence.values()]
    dir_points = 5 if dir_means and min(dir_means) > 0.52 else 0
    add("cross-direction consistency", dir_points, 5,
        f"BUY/SELL AUC {['%.4f' % m for m in dir_means]}" if dir_means else "n/a")

    horizon_aucs = [v["logistic_auc"] for v in horizon_result.values()
                    if v["logistic_auc"] is not None]
    hor_points = 10 if horizon_aucs and max(horizon_aucs) >= 0.54 else (
        5 if horizon_aucs and max(horizon_aucs) >= 0.52 else 0)
    add("horizon structure", hor_points, 10,
        f"AUC across horizons {min(horizon_aucs):.4f}-{max(horizon_aucs):.4f}"
        if horizon_aucs else "n/a")

    alt_devs = [v.get("best_univariate_dev", 0) for k, v in target_result.items()
                if k != "A_current_barrier" and not v.get("tautological")]
    tgt_points = 5 if alt_devs and max(alt_devs) > observed_best * 1.5 else 0
    add("alternative target promise", tgt_points, 5,
        (f"best NON-TAUTOLOGICAL alternative-target stratified dev "
         f"{max(alt_devs):.4f} vs current {observed_best:.4f}")
        if alt_devs else "no non-tautological alternative scored")

    total = sum(p for _, p, _, _ in components)
    maximum = sum(o for _, _, o, _ in components)

    for name, points, out_of, reason in components:
        print(f"  {points:3d}/{out_of:<3d}  {name:34s}  {reason}")
    print(f"\n  TRAINING FEASIBILITY SCORE: {total}/{maximum}")
    results["feasibility_score"] = {"score": total, "max": maximum,
                                    "components": [
                                        {"name": n, "points": p, "out_of": o,
                                         "reason": r}
                                        for n, p, o, r in components]}

    # ---------------------------------------------------------------- 12
    head("12. DECISION")

    if total >= 60 and univ_beats and model_beats:
        verdict = "GREEN"
    elif total >= 30 or univ_beats or model_beats:
        verdict = "YELLOW"
    else:
        verdict = "RED"

    print(f"  VERDICT: {verdict}   (score {total}/{maximum})")
    results["verdict"] = verdict

    if verdict == "RED":
        print("""
  No evidence of predictive information survives a noise floor that respects
  the autocorrelation in this data. Retraining the same model on the same
  columns against the same target will not change that, and neither will
  Optuna: hyper-parameters redistribute a score, they do not create
  information.

  What must change before any further training — in order of what the evidence
  above supports:
    1. the target, if section 6 showed an alternative framing carrying more
       univariate signal than the barrier outcome
    2. the timeframe, if section 8 showed outcomes resolving inside 1-2 bars
       while features describe a 17-day window
    3. the feature inputs, if section 2 showed every column near-constant in
       information terms — new inputs, not new combinations of these
    4. the symbol scope, if section 9 showed one instrument behaving
       differently from the pooled model""")
    elif verdict == "YELLOW":
        print("""
  Something is present but it is weak, unstable, or confined to one slice. Do
  not start Optuna: tuning against a signal this fragile fits the folds it was
  measured on.

  The next experiment is the cheapest one that could make the lead solid or
  kill it — narrow to the slice that showed it, and test whether it persists
  out of sample on its own terms.""")
    else:
        print("""
  Evidence supports proceeding, but start with the smallest experiment that
  could fail: a single XGBoost fit with fixed, conservative hyper-parameters
  under the same walk-forward, compared against the logistic baseline above. If
  it does not clearly beat that baseline, the problem is not hyper-parameters
  and Optuna will not help.""")

    print(f"\n  HOLDOUT: {holdout_n} rows remain unread and must stay that way "
          f"until a model is final.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n  machine-readable results -> {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
