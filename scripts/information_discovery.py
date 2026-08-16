"""Post-RED information discovery: does anything NEW carry signal?

The feasibility gate found nothing in the existing ten features
(FEASIBILITY_REPORT.md, direction-free logistic AUC 0.5010, score 10/100,
verdict RED). Retraining on the same inputs will not change that. This script
does not touch those ten features or retest them — it tests genuinely new
information sources (spread, volume, cross-asset context) against the same
statistical apparatus that produced the RED verdict, so a YELLOW or GREEN here
would mean something actually new, not the old result read differently.

Everything is research-only. No model is trained beyond cheap diagnostic
ones (logistic regression, a shallow tree, a small forest), nothing is written
to models/entry/entry_model.json, and Optuna does not appear anywhere in this
file — that stays out until a later, separate step, and only if this script
finds something to tune.

Holdout isolation is not a comment, it is a runnable check: `--verify-holdout-
isolation` reruns the whole research pipeline with the holdout region
corrupted and asserts byte-identical output. If that assertion fails, this
script refuses to report a verdict, because a pipeline that can be moved by
mutating data it swears it never reads cannot be trusted about anything else
it reports either.

Usage (Windows machine, needs the real candle files under data/historical/)::

    python scripts/information_discovery.py --verify-holdout-isolation
    python scripts/information_discovery.py --cross-asset --permutations 200
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

from analysis.features import timeframe_alignment as ta  # noqa: E402
from analysis.features import microstructure_features as micro  # noqa: E402
from analysis.features import cross_asset_features as ca  # noqa: E402
from analysis.models import entry_feature_spec as spec  # noqa: E402

import train_entry_model as trainer  # noqa: E402
import feasibility_gate as fg  # noqa: E402

OUT_DIR = os.path.join("research", "information_discovery")


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def sub(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Repository / source audit (programmatic, against the real files in front of us)
# ---------------------------------------------------------------------------

def audit_sources(candles_by_symbol: dict) -> dict:
    head("1. REPOSITORY / DATA SOURCE AUDIT")
    report: dict = {"microstructure": {}, "cross_asset": {}}

    for symbol, tf_map in candles_by_symbol.items():
        avail = micro.field_availability(tf_map["H4"])
        report["microstructure"][symbol] = avail
        print(f"  {symbol:8s} H4 microstructure: {avail}")

    print("\n  Fields discovered but not currently used anywhere in the repo before")
    print("  this script: MT5's rate struct carries `spread` and `real_volume`")
    print("  alongside OHLC; both mt5_client.get_candles and the pre-existing")
    print("  fetch_training_candles.py discarded them. Capturing them required a")
    print("  one-line addition each, no new dependency, and no new look-ahead risk")
    print("  — they are properties of the closed bar, known at the same instant as")
    print("  its OHLC.")

    print("\n  NOT AVAILABLE historically, checked directly against this repo's data:")
    print("    economic calendar : data/news/calendar.py fetches today/tomorrow only,")
    print("                        needs FINNHUB_API_KEY (unset by default), no archive")
    print("    news               : data/news/fetcher.py is RSS-live-only; the `news`")
    print("                        DB table exists but holds 0 rows")
    print("    sentiment          : derived from news, same gap")
    print("    order book/liquidity: not exposed by the MT5 API calls used here")

    return report


# ---------------------------------------------------------------------------
# Holdout isolation — a runnable proof, not a comment
# ---------------------------------------------------------------------------

def research_cutoff_time(candles_by_symbol, research_frac: float) -> float:
    """A decision timestamp fixed from raw candle span, before any labelling.

    Splitting by a COUNT of resolved rows — `cut = int(len(y_all) *
    research_frac)`, what feasibility_gate.py does — makes the boundary itself
    a function of what happens to be labellable, including in the holdout
    region: corrupt holdout candles, change how many of them resolve, shift
    the total row count, and a handful of borderline rows change which side
    of the split they land on. That is a real, if narrow, dependency on
    holdout content, caught by this module's own isolation self-test.

    Splitting by timestamp instead removes the dependency structurally: the
    cutoff is computed once from the earliest and latest candle across every
    symbol, before a single label is generated, so no amount of holdout
    corruption can move it.
    """
    all_times = [c["t"] for tf_map in candles_by_symbol.values() for c in tf_map["H4"]]
    if not all_times:
        raise ValueError("no H4 candles to determine a research cutoff from")
    start, end = min(all_times), max(all_times)
    return start + research_frac * (end - start)


def build_research_rows(candles_by_symbol, horizon: int, research_frac: float):
    """Everything downstream of this function only ever sees the research slice."""
    cutoff = research_cutoff_time(candles_by_symbol, research_frac)
    X_all, y_all, meta_all = trainer.build_dataset(candles_by_symbol, horizon)

    research_idx = [i for i in range(len(y_all)) if meta_all[i]["t"] <= cutoff]
    research_idx.sort(key=lambda i: meta_all[i]["t"])
    holdout_n = sum(1 for m in meta_all if m["t"] > cutoff)

    X_research = [X_all[i] for i in research_idx]
    y_research = [y_all[i] for i in research_idx]
    meta_research = [meta_all[i] for i in research_idx]
    return X_research, y_research, meta_research, holdout_n


def _corrupt_holdout(candles_by_symbol, cutoff: float, horizon: int):
    """Multiply OHLC (and spread/volume, if present) by a large constant for
    every candle that no research decision could legitimately have read.

    The boundary is the last POSSIBLE decision index under the cutoff — every
    i with `decision_time(h4, i, "H4") <= cutoff` and i inside build_dataset's
    own loop range — not the last index that happened to produce a resolved
    row in the uncorrupted data. Those are different things: a decision whose
    forward barrier is not yet touched in the real data (simulate_trade
    returns None, no row) is still a research-region decision, and corrupting
    candles inside ITS label window can flip it from unresolved to resolved,
    manufacturing a brand-new row that a naive per-row boundary would miss
    entirely. Two earlier versions of this function got exactly that wrong:
    first by using the last research decision's own timestamp (which ignores
    that a barrier label reads forward past it), then by deriving the reach
    from `meta` — the resolved rows — rather than from every decision the
    cutoff actually admits.
    """
    last_h4_index = {}
    for symbol, tf_map in candles_by_symbol.items():
        h4 = tf_map["H4"]
        limit = len(h4) - horizon - 1  # build_dataset's own loop upper bound
        idx = None
        for i in range(trainer.WARMUP_BARS, limit):
            if ta.decision_time(h4, i, "H4") <= cutoff:
                idx = i
            else:
                break
        if idx is not None:
            last_h4_index[symbol] = idx

    last_h1_time = {symbol: cutoff for symbol in candles_by_symbol}

    def tamper(candle):
        out = dict(candle)
        for key in ("open", "high", "low", "close", "spread", "real_volume"):
            if key in out:
                out[key] = out[key] * 97.0 + 13.0
        return out

    corrupted = {}
    for symbol, tf_map in candles_by_symbol.items():
        new_map = {}

        h4 = tf_map["H4"]
        h4_boundary = last_h4_index.get(symbol)
        if h4_boundary is None:
            new_map["H4"] = h4
        else:
            reach = h4_boundary + horizon  # simulate_trade's max reachable index
            new_map["H4"] = [tamper(c) if i > reach else c for i, c in enumerate(h4)]

        h1 = tf_map["H1"]
        h1_cutoff = last_h1_time.get(symbol)
        if h1_cutoff is None:
            new_map["H1"] = h1
        else:
            new_map["H1"] = [tamper(c) if c["t"] > h1_cutoff else c for c in h1]

        corrupted[symbol] = new_map
    return corrupted


def verify_holdout_isolation(candles_by_symbol, horizon: int, research_frac: float) -> bool:
    head("2. HOLDOUT ISOLATION — RUNNABLE PROOF")
    print("  Corrupting every candle in the holdout region (multiply by 97, add 13)")
    print("  and rebuilding the research dataset. If holdout rows are truly never")
    print("  read, the research rows must come out byte-identical.\n")

    Xa, ya, ma, holdout_n = build_research_rows(candles_by_symbol, horizon, research_frac)
    cutoff = research_cutoff_time(candles_by_symbol, research_frac)
    corrupted = _corrupt_holdout(candles_by_symbol, cutoff, horizon)
    Xb, yb, mb, _ = build_research_rows(corrupted, horizon, research_frac)

    same_rows = len(Xa) == len(Xb)
    same_X = same_rows and all(a == b for a, b in zip(Xa, Xb))
    same_y = same_rows and ya == yb
    same_meta = same_rows and all(
        (a["t"], a["symbol"], a["direction"]) == (b["t"], b["symbol"], b["direction"])
        for a, b in zip(ma, mb))

    passed = same_rows and same_X and same_y and same_meta
    print(f"  research rows          : {len(Xa)} vs {len(Xb)}  {'match' if same_rows else 'DIFFER'}")
    print(f"  feature values         : {'identical' if same_X else 'DIFFER'}")
    print(f"  labels                 : {'identical' if same_y else 'DIFFER'}")
    print(f"  row identity (t/sym/dir): {'identical' if same_meta else 'DIFFER'}")
    print(f"\n  HOLDOUT ISOLATION: {'VERIFIED' if passed else 'VIOLATED'}")
    if not passed:
        print("\n  Refusing to report a verdict — the pipeline used data it should not")
        print("  have. Fix the leak before trusting any of its other output.")
    return passed


# ---------------------------------------------------------------------------
# New-information dataset construction
# ---------------------------------------------------------------------------

def build_information_dataset(candles_by_symbol, meta, cross_asset_data):
    """Row-aligned NEW feature vectors for exactly the research rows in `meta`.

    Returns (micro_rows, cross_rows, micro_available, cross_available) where a
    row is None when that source could not be computed for that decision —
    dropped later by intersecting available indices, never filled with zero.
    """
    micro_rows: list = []
    cross_rows: list = []
    micro_names = list(micro.FEATURE_NAMES)

    for row in meta:
        symbol = row["symbol"]
        h4 = candles_by_symbol[symbol]["H4"]
        visible = ta.closed_slice(h4, "H4", row["t"])
        atr = None
        if len(visible) >= trainer.WARMUP_BARS:
            from analysis.features import live_parity_features as lpf
            ind = lpf.live_indicators(visible)
            atr = float(ind["atr"]) if ind else None

        named = (micro.build_microstructure_features(visible, timestamp=row["t"], atr=atr)
                if atr else None)
        micro_rows.append(named)

        cross_named = ca.build_cross_asset_features(
            cross_asset_data, symbol=symbol, decision_timestamp=row["t"])
        cross_rows.append(cross_named)

    micro_available = sum(1 for r in micro_rows if r is not None)
    cross_available = sum(1 for r in cross_rows if r is not None)
    return micro_rows, cross_rows, micro_available, cross_available


# ---------------------------------------------------------------------------
# Information audit for one new feature matrix
# ---------------------------------------------------------------------------

def information_audit(X, y, meta, names, rng, block, permutations):
    y_arr = np.asarray(y, dtype=float)
    strata = [(m["symbol"], m["direction"]) for m in meta]

    print(f"  {'feature':26s} {'unique':>7} {'std':>10} {'autocorr':>9} "
          f"{'strat AUC':>9} {'MI':>8}")
    audit = {}
    for j, name in enumerate(names):
        col = X[:, j]
        uniq = len(np.unique(col))
        std = float(col.std())
        autocorr = (float(np.corrcoef(col[:-1], col[1:])[0, 1])
                   if std > 0 and len(col) > 10 else float("nan"))
        s_auc = fg.stratified_auc(y_arr, col, strata)
        audit[name] = {"unique": uniq, "std": round(std, 6),
                       "autocorr": None if math.isnan(autocorr) else round(autocorr, 4),
                       "stratified_auc": None if math.isnan(s_auc) else round(s_auc, 4)}
        print(f"  {name:26s} {uniq:7d} {std:10.4f} "
              f"{(autocorr if not math.isnan(autocorr) else float('nan')):9.4f} "
              f"{(s_auc if not math.isnan(s_auc) else float('nan')):9.4f}")

    try:
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(X, y_arr, discrete_features=False, random_state=0)
        for name, value in zip(names, mi):
            audit[name]["mutual_information"] = round(float(value), 6)
        for name, value in sorted(zip(names, mi), key=lambda kv: -kv[1]):
            print(f"    MI  {name:26s} {value:.6f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  mutual information unavailable: {exc}")

    owner, observed = fg.best_stratified_deviation(y_arr, X, strata, names)
    null = []
    for _ in range(permutations):
        shuffled = fg.block_permute(y_arr, block, rng)
        null.append(fg.best_stratified_deviation(shuffled, X, strata, names)[1])
    pct = fg.permutation_percentile(observed, null)
    beats = pct >= 0.95
    print(f"\n  best stratified deviation: {owner} = {observed:.4f} "
          f"at {pct:.1%} of its block-permutation null "
          f"(median {statistics.median(null):.4f}, p95 {sorted(null)[int(0.95*len(null))]:.4f})")
    print(f"  -> {'BEATS' if beats else 'NO EVIDENCE:'} the noise floor"
          f"{'' if beats else ' (observed <= p95 of null)'}")

    return {"per_feature": audit, "best_owner": owner, "best_deviation": round(observed, 4),
            "null_median": round(statistics.median(null), 4),
            "null_p95": round(sorted(null)[int(0.95 * len(null))], 4),
            "percentile": round(pct, 4), "beats_noise": bool(beats)}


# ---------------------------------------------------------------------------
# OLD vs NEW vs OLD+NEW, direction-free, walk-forward
# ---------------------------------------------------------------------------

def compare_old_new(X_old, X_new, y, meta, times, args):
    strata_index = spec.FEATURE_NAMES.index("direction")
    old_nodir = np.delete(X_old, strata_index, axis=1)

    combined = np.concatenate([old_nodir, X_new], axis=1)

    rows = [
        ("OLD FEATURES ONLY (direction-free)", old_nodir),
        ("NEW INFORMATION ONLY", X_new),
        ("OLD + NEW (direction-free)", combined),
    ]

    result = {}
    print(f"\n  {'view':40s} {'nf':>3} {'mean AUC':>9} {'folds':>6} {'spread':>8}")
    for label, matrix in rows:
        scores = fg.walk_forward_scores(matrix, y, times, fg.logistic_factory, args.folds)
        if not scores:
            print(f"  {label:40s} {matrix.shape[1]:3d}       n/a")
            result[label] = None
            continue
        mean = statistics.mean(scores)
        spread = max(scores) - min(scores) if len(scores) > 1 else 0.0
        result[label] = {"mean": round(mean, 4), "folds": len(scores),
                         "spread": round(spread, 4),
                         "per_fold": [round(s, 4) for s in scores]}
        print(f"  {label:40s} {matrix.shape[1]:3d} {mean:9.4f} {len(scores):6d} {spread:8.4f}")

    return result, combined


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", default=os.path.join("data", "historical"))
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--research-frac", type=float, default=0.70)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--cross-asset", action="store_true",
                        help="also attempt DXY/yields/silver/oil via yfinance")
    parser.add_argument("--cross-asset-start", default="2023-01-01")
    parser.add_argument("--cross-asset-end", default=None)
    parser.add_argument("--verify-holdout-isolation", action="store_true",
                        help="only run the holdout-isolation self-test, then exit")
    parser.add_argument("--json", default=os.path.join(OUT_DIR, "information_audit.json"))
    args = parser.parse_args()

    rng = np.random.default_rng(20260816)

    from config import SYMBOLS
    symbols = args.symbols or list(SYMBOLS)

    provenance = trainer.check_provenance(args.candles)
    if not provenance["real"]:
        print(f"REFUSING: {provenance['reason']}")
        return 1

    candles_by_symbol = {s: {tf: trainer.load_candles(s, tf, args.candles)
                             for tf in ("H4", "H1")} for s in symbols}

    if not verify_holdout_isolation(candles_by_symbol, args.horizon, args.research_frac):
        return 1
    if args.verify_holdout_isolation:
        print("\nHoldout isolation verified. Exiting (--verify-holdout-isolation).")
        return 0

    audit_sources(candles_by_symbol)

    X_old, y, meta, holdout_n = build_research_rows(
        candles_by_symbol, args.horizon, args.research_frac)
    X_old = np.asarray(X_old, dtype=float)
    times = [m["t"] for m in meta]
    print(f"\n  research rows: {len(y)}   holdout rows (UNREAD): {holdout_n}")

    cross_asset_data, cross_report = {}, ca.FetchReport()
    if args.cross_asset:
        head("3. CROSS-ASSET FETCH")
        end = args.cross_asset_end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cross_asset_data, cross_report = ca.fetch_all(args.cross_asset_start, end)
        for key, status in cross_report.status.items():
            print(f"  {key:10s} {status}")
    else:
        print("\n  (--cross-asset not passed; skipping DXY/yields/silver/oil)")

    micro_rows, cross_rows, micro_n, cross_n = build_information_dataset(
        candles_by_symbol, meta, cross_asset_data)

    head("4. NEW FEATURE AVAILABILITY ON RESEARCH ROWS")
    print(f"  microstructure available: {micro_n}/{len(meta)} rows")
    print(f"  cross-asset available   : {cross_n}/{len(meta)} rows")

    if micro_n < 500:
        print("\n  Too few rows with microstructure data to evaluate — the candle files")
        print("  under data/historical/ likely predate the spread/real_volume capture")
        print("  added to fetch_training_candles.py. Re-run that script to regenerate")
        print("  data/historical/*.json, then re-run this script.")
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"status": "INSUFFICIENT_DATA",
                      "reason": "spread/real_volume not present in candle files",
                      "micro_available": micro_n, "cross_available": cross_n},
                     fh, indent=2)
        print(f"\n  wrote {args.json}")
        print("\nDECISION GATE: RED (no new information could be evaluated)")
        return 0

    idx = [i for i, r in enumerate(micro_rows) if r is not None]
    X_new = np.asarray([[micro_rows[i][n] for n in micro.FEATURE_NAMES] for i in idx],
                       dtype=float)
    X_old_sub = X_old[idx]
    y_sub = [y[i] for i in idx]
    meta_sub = [meta[i] for i in idx]
    times_sub = [times[i] for i in idx]
    names_new = list(micro.FEATURE_NAMES)

    if cross_n >= 500:
        cross_idx = [i for i in idx if cross_rows[i] is not None]
        if len(cross_idx) >= 500:
            all_cross_keys = sorted({k for i in cross_idx for k in cross_rows[i]})
            cross_matrix = np.asarray(
                [[cross_rows[i].get(k, 0.0) for k in all_cross_keys] for i in cross_idx],
                dtype=float)
            micro_matrix = np.asarray(
                [[micro_rows[i][n] for n in micro.FEATURE_NAMES] for i in cross_idx],
                dtype=float)
            X_new = np.concatenate([micro_matrix, cross_matrix], axis=1)
            names_new = names_new + all_cross_keys
            idx = cross_idx
            X_old_sub = X_old[idx]
            y_sub = [y[i] for i in idx]
            meta_sub = [meta[i] for i in idx]
            times_sub = [times[i] for i in idx]

    head("5. INFORMATION AUDIT — NEW FEATURES")
    block = max(2 * args.horizon, 32)
    audit = information_audit(X_new, y_sub, meta_sub, names_new, rng, block, args.permutations)

    head("6. OLD vs NEW vs OLD+NEW (direction-free, walk-forward)")
    comparison, combined = compare_old_new(X_old_sub, X_new, y_sub, meta_sub, times_sub, args)

    head("7. SYMBOL-SPECIFIC")
    per_symbol = {}
    for symbol in symbols:
        sidx = [i for i, m in enumerate(meta_sub) if m["symbol"] == symbol]
        if len(sidx) < 300:
            continue
        scores = fg.walk_forward_scores(
            X_new[sidx], [y_sub[i] for i in sidx], [times_sub[i] for i in sidx],
            fg.logistic_factory, args.folds)
        if scores:
            per_symbol[symbol] = {"mean": round(statistics.mean(scores), 4),
                                  "n": len(sidx)}
            print(f"  {symbol:10s} n={len(sidx):6d} new-info-only AUC={statistics.mean(scores):.4f}")

    head("8. DECISION GATE")
    new_only = comparison.get("NEW INFORMATION ONLY")
    combined_view = comparison.get("OLD + NEW (direction-free)")
    old_only = comparison.get("OLD FEATURES ONLY (direction-free)")

    stable = (new_only and new_only["spread"] < 0.10 and
             sum(1 for s in new_only["per_fold"] if s > 0.5) >= len(new_only["per_fold"]) - 1)
    improves = (combined_view and old_only and
               combined_view["mean"] > old_only["mean"] + 0.01)

    if audit["beats_noise"] and stable and improves:
        verdict = "GREEN"
    elif audit["beats_noise"] or (new_only and new_only["mean"] > 0.52):
        verdict = "YELLOW"
    else:
        verdict = "RED"

    print(f"  new-information beats permutation null : {audit['beats_noise']}")
    print(f"  new-information-only AUC               : {new_only['mean'] if new_only else 'n/a'}")
    print(f"  stable across folds                    : {stable}")
    print(f"  OLD+NEW improves over OLD alone         : {improves}")
    print(f"\n  VERDICT: {verdict}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_rows": len(y), "holdout_rows_unread": holdout_n,
        "microstructure_available_rows": micro_n, "cross_asset_available_rows": cross_n,
        "audit": audit, "comparison": comparison, "per_symbol": per_symbol,
        "verdict": verdict,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  wrote {args.json}")

    if verdict == "RED":
        print("\n  No new evidence. models/entry/entry_model.json unchanged. Optuna: NOT ALLOWED.")
    elif verdict == "YELLOW":
        print("\n  A lead exists but is weak/unstable. Research continues. Optuna: NOT ALLOWED.")
    else:
        print("\n  Evidence supports a small, fixed-hyperparameter XGBoost experiment next,")
        print("  compared against this logistic baseline — still not Optuna.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
