"""Train the entry model on the ten features the live path actually sends.

Pipeline: raw candles -> live-formula indicators -> the shared 10-feature spec
-> TP/SL labels simulated on future bars -> validation -> walk-forward
-> final model.

Three properties this script exists to guarantee, each of which was violated by
the model it replaces:

1. **Feature parity.** Vectors are built by
   `analysis.models.entry_feature_spec.build_feature_vector`, the same function
   live inference calls. Indicators are recomputed by
   `analysis.features.live_parity_features`, which transcribes the live
   arithmetic (simple-average RSI, SMA-based MACD) rather than the standard
   formulas. The deployed 65-feature model was trained on a different feature
   set entirely and served a 10-slot vector, so every probability it produced
   was unrelated to the trade.

2. **Both directions.** Each qualifying bar is labelled twice, once as a BUY and
   once as a SELL, with TP/SL mirrored. The previous entry_v2 dataset had no
   `direction` column and defaulted every row to BUY, which is why the deployed
   model returned identical p_win for BUY and SELL.

3. **No look-ahead.** Features at bar *i* use candles up to and including *i*
   only. Labels look forward from *i+1*. H1 values are joined by "most recent H1
   bar that closed at or before this H4 bar's close" — never a later one.
   Validation is walk-forward in time, never a random split.

Usage::

    python scripts/fetch_training_candles.py     # on Windows, once
    python scripts/train_entry_model.py --dry-run
    python scripts/train_entry_model.py          # writes the model
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.features import live_parity_features as lpf  # noqa: E402
from analysis.models import entry_feature_spec as spec  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("train_entry_model")

CANDLE_DIR = os.path.join("data", "historical")
MODEL_PATH = os.path.join("models", "entry", "entry_model.json")
BACKUP_DIR = os.path.join("models_backup")
REPORT_PATH = os.path.join("models", "entry", "training_report.json")

# Stop/target distances must match what the bot actually places, or the labels
# describe trades it would never take. Sourced from tm_config, not hardcoded.
from trade_management.tm_config import (  # noqa: E402
    ATR_SL_BASE_MULTIPLIER,
    ATR_TP_BASE_MULTIPLIER,
)

# Labelling horizon in H4 bars. Justified empirically — see choose_horizon().
DEFAULT_HORIZON = 24

# Warm-up: live computes indicators over a 100-candle window and needs >=50
# bars for ma_trend. Bars before this produce a fallback row in production and
# must not become training data.
WARMUP_BARS = lpf.LIVE_INDICATOR_WINDOW


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_candles(symbol: str, timeframe: str, directory: str) -> List[Dict[str, float]]:
    path = os.path.join(directory, f"{symbol}_{timeframe}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/fetch_training_candles.py on the "
            f"Windows machine first — it needs a live MT5 terminal."
        )
    with open(path, encoding="utf-8") as fh:
        candles = json.load(fh)
    candles.sort(key=lambda c: c["t"])
    # Defensive: duplicate timestamps would double-count a bar.
    deduped, seen = [], set()
    for c in candles:
        if c["t"] in seen:
            continue
        seen.add(c["t"])
        deduped.append(c)
    return deduped


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

def simulate_trade(
    candles: List[Dict[str, float]],
    entry_idx: int,
    direction: str,
    atr: float,
    horizon: int,
) -> Optional[Dict[str, Any]]:
    """Walk forward from entry_idx+1 and see whether TP or SL is touched first.

    Returns None when neither is reached inside the horizon. Those rows are
    dropped rather than labelled by proximity: the previous pipeline's
    `fallback_neither` rule called a trade a win if the final close sat nearer
    TP than SL, which is a different question from the one the model is asked.
    """
    entry = float(candles[entry_idx]["close"])
    sl_dist = atr * ATR_SL_BASE_MULTIPLIER
    tp_dist = atr * ATR_TP_BASE_MULTIPLIER
    if sl_dist <= 0 or tp_dist <= 0:
        return None

    if direction == "BUY":
        tp, sl = entry + tp_dist, entry - sl_dist
    else:
        tp, sl = entry - tp_dist, entry + sl_dist

    end = min(entry_idx + horizon, len(candles) - 1)
    for j in range(entry_idx + 1, end + 1):
        hi = float(candles[j]["high"])
        lo = float(candles[j]["low"])

        if direction == "BUY":
            hit_tp, hit_sl = hi >= tp, lo <= sl
        else:
            hit_tp, hit_sl = lo <= tp, hi >= sl

        # Both touched inside one bar: without tick data the order is unknowable.
        # Count it as a loss — the pessimistic reading, so the model is never
        # trained to expect a win it may not have received.
        if hit_tp and hit_sl:
            return {"label": 0.0, "reason": "both_same_bar", "bars": j - entry_idx}
        if hit_tp:
            return {"label": 1.0, "reason": "tp_first", "bars": j - entry_idx}
        if hit_sl:
            return {"label": 0.0, "reason": "sl_first", "bars": j - entry_idx}

    return None


def choose_horizon(candles_by_symbol: Dict[str, List], probe_horizon: int = 48) -> Tuple[int, Dict]:
    """Pick the labelling horizon from how long trades actually take to resolve.

    Runs the simulation once at a deliberately generous horizon, then reports
    the resolution curve so the choice is evidence-based rather than inherited.
    """
    resolutions: List[int] = []
    unresolved = 0
    total = 0

    for symbol, tf_map in candles_by_symbol.items():
        h4 = tf_map["H4"]
        step = max(1, len(h4) // 400)  # sample, this is only for calibration
        for i in range(WARMUP_BARS, len(h4) - probe_horizon - 1, step):
            ind = lpf.live_indicators(h4[: i + 1])
            if ind is None:
                continue
            for direction in ("BUY", "SELL"):
                total += 1
                out = simulate_trade(h4, i, direction, float(ind["atr"]), probe_horizon)
                if out is None:
                    unresolved += 1
                else:
                    resolutions.append(out["bars"])

    if not resolutions:
        return DEFAULT_HORIZON, {"reason": "no samples; kept default"}

    resolutions.sort()

    def pct_within(h: int) -> float:
        return bisect.bisect_right(resolutions, h) / max(total, 1)

    curve = {h: round(pct_within(h), 4) for h in (4, 8, 12, 16, 20, 24, 32, 40, 48)}

    # Choose the smallest horizon that resolves >=85% of attempts, capped at the
    # probe. Beyond that the curve flattens and a longer horizon mostly holds
    # trades open past the point the live time-stop would have closed them.
    chosen = DEFAULT_HORIZON
    for h in sorted(curve):
        if curve[h] >= 0.85:
            chosen = h
            break

    stats = {
        "probe_horizon": probe_horizon,
        "attempts": total,
        "resolved": len(resolutions),
        "unresolved_pct": round(unresolved / max(total, 1), 4),
        "median_bars_to_resolution": statistics.median(resolutions),
        "p90_bars_to_resolution": resolutions[int(0.9 * len(resolutions)) - 1],
        "resolution_curve": curve,
        "chosen_horizon": chosen,
        "rule": "smallest horizon resolving >=85% of attempts",
    }
    return chosen, stats


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _h1_index(h1: List[Dict[str, float]]) -> List[float]:
    return [c["t"] for c in h1]


def build_dataset(
    candles_by_symbol: Dict[str, Dict[str, List]],
    horizon: int,
) -> Tuple[List[List[float]], List[float], List[Dict[str, Any]]]:
    """Build X, y and per-row metadata.

    One H4 bar yields up to two rows (BUY and SELL). Features come from bars
    <= i; labels from bars > i. H1 is joined by last-closed-at-or-before, so no
    H1 bar from the future leaks in.
    """
    X: List[List[float]] = []
    y: List[float] = []
    meta: List[Dict[str, Any]] = []

    for symbol, tf_map in candles_by_symbol.items():
        h4 = tf_map["H4"]
        h1 = tf_map["H1"]
        h1_times = _h1_index(h1)

        for i in range(WARMUP_BARS, len(h4) - horizon - 1):
            bar_close_t = h4[i]["t"]

            h4_ind = lpf.live_indicators(h4[: i + 1])
            if h4_ind is None:
                continue

            # Most recent H1 bar closing at or before this H4 bar.
            pos = bisect.bisect_right(h1_times, bar_close_t)
            if pos < WARMUP_BARS:
                continue
            h1_ind = lpf.live_indicators(h1[:pos])
            if h1_ind is None:
                continue

            trend_score, trend_dir = lpf.trend_score_from_indicators(h4_ind)
            momentum_score, mom_dir = lpf.momentum_score_from_indicators(h1_ind)
            # volatility_score comes from H1, matching the live snapshot's
            # TF_DECISION lookup.
            vol = lpf.volatility_score_from_indicators(h1_ind)
            regime = lpf.regime_from_scores(trend_dir, vol)
            # The live analyser derives strength from H4/H1/M15 alignment. M15
            # is not fetched for training, so H1 stands in for it — recorded
            # here rather than hidden, and the encoding is the shared one.
            strength = lpf.mtf_strength_from_directions(trend_dir, mom_dir, mom_dir)
            session = spec.session_from_timestamp(bar_close_t)
            atr = float(h4_ind["atr"])

            for direction in ("BUY", "SELL"):
                outcome = simulate_trade(h4, i, direction, atr, horizon)
                if outcome is None:
                    continue

                X.append(spec.build_feature_vector(
                    rsi=h1_ind["rsi"],
                    atr=atr,
                    macd=h1_ind["macd"],
                    trend_strength=strength,
                    trend_score=trend_score,
                    momentum_score=momentum_score,
                    volatility_score=vol,
                    market_regime=regime,
                    session=session,
                    direction=direction,
                ))
                y.append(outcome["label"])
                meta.append({
                    "symbol": symbol, "t": bar_close_t, "direction": direction,
                    "reason": outcome["reason"], "bars": outcome["bars"],
                    "regime": regime, "session": session,
                })

    return X, y, meta


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def check_provenance(directory: str) -> Dict[str, Any]:
    """Refuse to train a production model on anything but real broker candles.

    A model is only as trustworthy as the data under it, and synthetic candles
    exercise the code path without saying anything about market behaviour. The
    fetch script writes a manifest; its absence means the candles came from
    somewhere else.
    """
    manifest_path = os.path.join(directory, "manifest.json")
    if not os.path.exists(manifest_path):
        return {
            "real": False,
            "reason": (
                f"no manifest.json in {directory}. Only "
                "scripts/fetch_training_candles.py writes one, and it only runs "
                "against a live MT5 terminal. Candles of unknown origin will not "
                "be used to train a production model."
            ),
        }
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    return {"real": True, "manifest": manifest}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataset(X, y, meta) -> Dict[str, Any]:
    """Everything that should be inspected before a single tree is grown."""
    report: Dict[str, Any] = {}
    n = len(X)
    report["rows"] = n
    if n == 0:
        report["fatal"] = "empty dataset"
        return report

    report["features"] = spec.FEATURE_COUNT
    report["wrong_width_rows"] = sum(1 for row in X if len(row) != spec.FEATURE_COUNT)

    per_symbol: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "win": 0, "buy": 0, "sell": 0})
    for row_y, m in zip(y, meta):
        s = per_symbol[m["symbol"]]
        s["n"] += 1
        s["win"] += int(row_y == 1.0)
        s[m["direction"].lower()] += 1
    report["per_symbol"] = {
        k: {**v, "win_rate": round(v["win"] / v["n"], 4)} for k, v in per_symbol.items()
    }

    wins = sum(1 for v in y if v == 1.0)
    report["overall"] = {
        "win": wins, "loss": n - wins, "win_rate": round(wins / n, 4),
        "balance_ok": 0.25 <= wins / n <= 0.75,
    }

    report["label_reasons"] = dict(Counter(m["reason"] for m in meta))
    report["direction_split"] = dict(Counter(m["direction"] for m in meta))

    # Missing / non-finite
    bad = sum(
        1 for row in X for v in row
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    )
    report["non_finite_values"] = bad

    # Duplicates: identical feature vector AND identical label.
    seen = set()
    dupes = 0
    for row, row_y in zip(X, y):
        key = (tuple(row), row_y)
        if key in seen:
            dupes += 1
        seen.add(key)
    report["duplicate_rows"] = dupes
    report["duplicate_pct"] = round(dupes / n, 4)

    # Per-feature distribution, and which features are constant.
    dist = {}
    constants = []
    for idx, name in enumerate(spec.FEATURE_NAMES):
        col = [row[idx] for row in X]
        uniq = len(set(col))
        dist[name] = {
            "min": round(min(col), 6), "max": round(max(col), 6),
            "mean": round(statistics.mean(col), 6),
            "unique": uniq,
        }
        if uniq == 1:
            constants.append(name)
    report["feature_distribution"] = dist
    report["constant_features"] = constants
    report["constant_features_expected"] = sorted(spec.LIVE_CONSTANT_FEATURES)

    # Leakage probes: a feature that predicts the label almost perfectly is a
    # red flag, and the label must not be inferable from the direction alone.
    suspicious = []
    for idx, name in enumerate(spec.FEATURE_NAMES):
        col = [row[idx] for row in X]
        if len(set(col)) == 1:
            continue
        won = [c for c, v in zip(col, y) if v == 1.0]
        lost = [c for c, v in zip(col, y) if v == 0.0]
        if not won or not lost:
            continue
        pooled = statistics.pstdev(col) or 1e-12
        sep = abs(statistics.mean(won) - statistics.mean(lost)) / pooled
        if sep > 2.0:
            suspicious.append({"feature": name, "separation_sigma": round(sep, 2)})
    report["leakage_suspects"] = suspicious

    # Direct leakage probe: no feature may be an near-perfect classifier on its
    # own. If one is, an outcome-derived value has reached the feature vector.
    single_feature_auc = {}
    for idx, name in enumerate(spec.FEATURE_NAMES):
        col = [row[idx] for row in X]
        if len(set(col)) == 1:
            continue
        pairs = sorted(zip(col, y))
        pos = sum(1 for _, v in pairs if v == 1.0)
        neg = len(pairs) - pos
        if not pos or not neg:
            continue
        rank_sum = sum(i + 1 for i, (_, v) in enumerate(pairs) if v == 1.0)
        auc = (rank_sum - pos * (pos + 1) / 2) / (pos * neg)
        single_feature_auc[name] = round(max(auc, 1 - auc), 4)
    report["single_feature_auc"] = single_feature_auc
    report["leaky_features"] = [
        n for n, a in single_feature_auc.items() if a >= 0.90
    ]

    by_dir = defaultdict(lambda: [0, 0])
    for row_y, m in zip(y, meta):
        by_dir[m["direction"]][0] += 1
        by_dir[m["direction"]][1] += int(row_y == 1.0)
    report["win_rate_by_direction"] = {
        k: round(v[1] / v[0], 4) for k, v in by_dir.items()
    }

    # Outcome distribution over time — a model trained on one regime and served
    # in another is a silent failure mode.
    by_period = defaultdict(lambda: [0, 0])
    for row_y, m in zip(y, meta):
        q = datetime.fromtimestamp(m["t"], tz=timezone.utc).strftime("%Y-Q%m")
        by_period[q][0] += 1
        by_period[q][1] += int(row_y == 1.0)
    report["win_rate_by_period"] = {
        k: {"n": v[0], "win_rate": round(v[1] / v[0], 4)}
        for k, v in sorted(by_period.items())
    }

    return report


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def _metrics(y_true, p) -> Dict[str, Any]:
    """Full classification metrics. Accuracy alone is not a success criterion:
    with a ~34% win rate, always predicting "loss" scores 66%."""
    import numpy as np

    yt = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float).ravel()
    pred = (p >= 0.5).astype(float)

    tp = float(((pred == 1) & (yt == 1)).sum())
    tn = float(((pred == 0) & (yt == 0)).sum())
    fp = float(((pred == 1) & (yt == 0)).sum())
    fn = float(((pred == 0) & (yt == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    pos, neg = p[yt == 1.0], p[yt == 0.0]
    roc_auc = None
    if len(pos) and len(neg):
        allp = np.concatenate([pos, neg])
        ranks = allp.argsort().argsort().astype(float) + 1
        roc_auc = float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                        / (len(pos) * len(neg)))

    # PR-AUC by step-wise interpolation over descending score order.
    pr_auc = None
    if len(pos):
        order = np.argsort(-p)
        ys = yt[order]
        tps = np.cumsum(ys)
        fps = np.cumsum(1 - ys)
        prec = tps / np.maximum(tps + fps, 1e-12)
        rec = tps / max(ys.sum(), 1e-12)
        pr_auc = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))

    # Calibration: mean predicted vs actual, in probability deciles.
    calib = []
    for lo in np.arange(0.0, 1.0, 0.1):
        m = (p >= lo) & (p < lo + 0.1)
        if m.sum() >= 10:
            calib.append({"bin": round(float(lo), 1), "n": int(m.sum()),
                          "mean_pred": round(float(p[m].mean()), 4),
                          "actual": round(float(yt[m].mean()), 4)})

    base_rate = float(yt.mean())
    return {
        "n": int(len(yt)),
        "accuracy": round(float((pred == yt).mean()), 4),
        "majority_baseline": round(max(base_rate, 1 - base_rate), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "pr_auc_baseline": round(base_rate, 4),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "base_win_rate": round(base_rate, 4),
        "calibration": calib,
    }


def _baselines(y_train, y_test, X_test, meta_test) -> Dict[str, Any]:
    """Simple comparators the model must beat to be worth deploying."""
    import numpy as np

    yt = np.asarray(y_test, dtype=float)
    prior = float(np.asarray(y_train, dtype=float).mean())

    out = {
        "always_loss": {"accuracy": round(float((yt == 0).mean()), 4)},
        "always_win": {"accuracy": round(float((yt == 1).mean()), 4)},
        "train_prior_constant": _metrics(yt, np.full(len(yt), prior)),
    }

    # A trend-following heuristic using only trend_score, no learning.
    idx = spec.FEATURE_NAMES.index("trend_score")
    ts = np.asarray([row[idx] for row in X_test], dtype=float)
    spread = float(ts.max() - ts.min())  # np.ndarray.ptp() was removed in NumPy 2
    out["trend_score_heuristic"] = _metrics(yt, (ts - ts.min()) / max(spread, 1e-12))
    return out


def walk_forward(X, y, meta, folds: int = 5) -> Dict[str, Any]:
    """Expanding-window validation in chronological order.

    A random split would let the model see bar i+1 while being scored on bar i.
    Adjacent bars share almost all of their indicator window, so a random split
    reports a score the live system can never reproduce.
    """
    import numpy as np
    import xgboost as xgb

    order = sorted(range(len(X)), key=lambda i: meta[i]["t"])
    Xo = [X[i] for i in order]
    yo = [y[i] for i in order]
    mo = [meta[i] for i in order]

    n = len(Xo)
    fold_size = n // (folds + 1)
    if fold_size < 50:
        return {"skipped": f"not enough rows for {folds} folds (n={n})"}

    results = []
    baselines_all = []
    for k in range(1, folds + 1):
        tr_end = fold_size * k
        te_end = min(fold_size * (k + 1), n)
        X_tr, y_tr = Xo[:tr_end], yo[:tr_end]
        X_te, y_te, m_te = Xo[tr_end:te_end], yo[tr_end:te_end], mo[tr_end:te_end]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            continue

        dtr = xgb.DMatrix(np.asarray(X_tr, dtype=float), label=np.asarray(y_tr, dtype=float),
                          feature_names=list(spec.FEATURE_NAMES))
        dte = xgb.DMatrix(np.asarray(X_te, dtype=float), feature_names=list(spec.FEATURE_NAMES))

        booster = xgb.train(_params(), dtr, num_boost_round=200, verbose_eval=False)
        p = np.asarray(booster.predict(dte)).ravel()

        fold = {"fold": k, "train": len(X_tr), "test": len(X_te)}
        fold["overall"] = _metrics(y_te, p)

        # Slices: per symbol, per direction, per regime.
        for slice_name, key in (("per_symbol", "symbol"),
                                ("per_direction", "direction"),
                                ("per_regime", "regime")):
            buckets: Dict[str, List[int]] = defaultdict(list)
            for i, m in enumerate(m_te):
                buckets[str(m.get(key))].append(i)
            fold[slice_name] = {
                kk: _metrics([y_te[i] for i in idxs], p[idxs])
                for kk, idxs in buckets.items()
                if len(idxs) >= 30 and len(set(y_te[i] for i in idxs)) > 1
            }

        # Behaviour at the live entry threshold.
        mask = p >= 0.60
        fold["at_threshold_0.60"] = (
            {"n": int(mask.sum()),
             "win_rate": round(float(np.asarray(y_te)[mask].mean()), 4),
             "base_win_rate": round(float(np.asarray(y_te).mean()), 4)}
            if mask.sum() >= 20 else None
        )

        baselines_all.append(_baselines(y_tr, y_te, X_te, m_te))
        results.append(fold)

    if not results:
        return {"skipped": "no usable folds"}

    aucs = [r["overall"]["roc_auc"] for r in results if r["overall"]["roc_auc"] is not None]
    praucs = [r["overall"]["pr_auc"] for r in results if r["overall"]["pr_auc"] is not None]
    return {
        "folds": results,
        "baselines_per_fold": baselines_all,
        "mean_roc_auc": round(statistics.mean(aucs), 4) if aucs else None,
        "mean_pr_auc": round(statistics.mean(praucs), 4) if praucs else None,
        "mean_accuracy": round(statistics.mean(r["overall"]["accuracy"] for r in results), 4),
        "mean_majority_baseline": round(
            statistics.mean(r["overall"]["majority_baseline"] for r in results), 4),
        "mean_f1": round(statistics.mean(r["overall"]["f1"] for r in results), 4),
    }


def _params() -> Dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 5,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# Verification before install
# ---------------------------------------------------------------------------

def verify_model(path: str) -> Dict[str, Any]:
    """Load from the live path and run a live-shaped prediction through it."""
    import xgboost as xgb

    checks: Dict[str, Any] = {}

    booster = xgb.Booster()
    booster.load_model(path)
    checks["num_features"] = booster.num_features()
    checks["num_features_ok"] = booster.num_features() == spec.FEATURE_COUNT

    names = booster.feature_names
    checks["feature_names"] = names
    checks["feature_names_ok"] = (
        names is not None and tuple(names) == spec.FEATURE_NAMES
    )
    return checks


def live_inference_check() -> Dict[str, Any]:
    """Exercise the real production entry point, not a reimplementation."""
    import analysis.models.xgboost_v2_inference as inf

    inf._model = None  # drop the cached booster so the new file is loaded

    out = {}
    for symbol, kwargs in {
        "EURUSD": dict(rsi=58.0, atr=0.0018, macd=0.0004),
        "GBPUSD": dict(rsi=44.0, atr=0.0023, macd=-0.0006),
        "XAUUSD": dict(rsi=62.0, atr=41.6, macd=3.2),
    }.items():
        per_symbol = {}
        for direction in ("BUY", "SELL"):
            r = inf.predict_with_v2(
                trend_strength=0.0, trend_score=70.0, momentum_score=65.0,
                volatility_score=55.0, market_regime="TRENDING",
                direction=direction, **kwargs,
            )
            per_symbol[direction] = {
                "status": r["status"], "available": r["available"],
                "p_win": None if r["p_win"] is None else round(r["p_win"], 6),
            }
        per_symbol["direction_changes_p_win"] = (
            per_symbol["BUY"]["p_win"] != per_symbol["SELL"]["p_win"]
        )
        out[symbol] = per_symbol
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", default=CANDLE_DIR)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--horizon", type=int, default=None,
                        help="override the empirically chosen horizon")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true",
                        help="run everything but do not touch the live model file")
    parser.add_argument("--min-auc", type=float, default=0.52,
                        help="refuse to install a model below this walk-forward ROC-AUC")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="run on candles without a fetch manifest. Exercises the "
                             "code path only; the model is never installed.")
    args = parser.parse_args()

    from config import SYMBOLS
    symbols = args.symbols or list(SYMBOLS)

    provenance = check_provenance(args.candles)
    if not provenance["real"]:
        if not args.allow_synthetic:
            print("REFUSING TO TRAIN: " + provenance["reason"])
            print("\nPass --allow-synthetic ONLY to exercise the code path; a "
                  "model trained that way must never be installed.")
            return 1
        print("WARNING: unverified candle source; --allow-synthetic given.")
        print("         The model will NOT be installed.")

    print("Loading candles...")
    candles_by_symbol: Dict[str, Dict[str, List]] = {}
    for s in symbols:
        candles_by_symbol[s] = {
            "H4": load_candles(s, "H4", args.candles),
            "H1": load_candles(s, "H1", args.candles),
        }
        print(f"  {s}: H4={len(candles_by_symbol[s]['H4'])} H1={len(candles_by_symbol[s]['H1'])}")

    if args.horizon:
        horizon, horizon_stats = args.horizon, {"override": True}
    else:
        print("\nCalibrating labelling horizon...")
        horizon, horizon_stats = choose_horizon(candles_by_symbol)
        print(f"  chosen horizon = {horizon} H4 bars")
        print(f"  resolution curve: {horizon_stats.get('resolution_curve')}")

    print("\nBuilding dataset...")
    X, y, meta = build_dataset(candles_by_symbol, horizon)
    print(f"  {len(X)} rows")

    print("\nValidating dataset...")
    validation = validate_dataset(X, y, meta)
    print(json.dumps({k: validation[k] for k in
                      ("rows", "overall", "per_symbol", "label_reasons",
                       "direction_split", "constant_features", "duplicate_pct",
                       "non_finite_values", "leakage_suspects",
                       "win_rate_by_direction") if k in validation},
                     indent=2, ensure_ascii=False))

    blockers = []
    if validation.get("fatal"):
        blockers.append(validation["fatal"])
    if validation.get("non_finite_values"):
        blockers.append(f"{validation['non_finite_values']} non-finite values")
    if validation.get("wrong_width_rows"):
        blockers.append(f"{validation['wrong_width_rows']} rows of wrong width")
    if not validation.get("overall", {}).get("balance_ok", False):
        blockers.append(f"class balance outside 25-75%: {validation.get('overall')}")
    unexpected_constants = set(validation.get("constant_features", [])) - set(
        spec.LIVE_CONSTANT_FEATURES)
    if unexpected_constants:
        blockers.append(f"unexpected constant features: {sorted(unexpected_constants)}")
    if validation.get("leakage_suspects"):
        blockers.append(f"possible leakage: {validation['leakage_suspects']}")

    if blockers:
        print("\nDATASET REJECTED:")
        for b in blockers:
            print("  - " + str(b))
        return 1

    print("\nWalk-forward validation...")
    wf = walk_forward(X, y, meta, folds=args.folds)
    print(json.dumps(wf, indent=2))

    if wf.get("skipped"):
        print(f"\nABORT: walk-forward skipped ({wf['skipped']})")
        return 1
    if wf.get("mean_roc_auc") is not None and wf["mean_roc_auc"] < args.min_auc:
        print(f"\nABORT: mean walk-forward ROC-AUC {wf['mean_roc_auc']} < {args.min_auc}. "
              "The model has no demonstrated edge; refusing to install it.")
        return 1

    print("\nTraining final model on all data...")
    import numpy as np
    import xgboost as xgb

    order = sorted(range(len(X)), key=lambda i: meta[i]["t"])
    dtrain = xgb.DMatrix(
        np.asarray([X[i] for i in order], dtype=float),
        label=np.asarray([y[i] for i in order], dtype=float),
        feature_names=list(spec.FEATURE_NAMES),
    )
    booster = xgb.train(_params(), dtrain, num_boost_round=200, verbose_eval=False)

    # The staged filename must keep the .json extension: XGBoost picks its
    # serialisation format from the extension, so saving to "*.candidate"
    # silently wrote binary UBJSON, which then failed to parse once the file
    # was renamed to .json. verify_model() did not catch it — loading the
    # candidate by its original name auto-detected the binary format — which is
    # precisely why the check below runs against the live path instead.
    staged = MODEL_PATH.replace(".json", ".candidate.json")
    os.makedirs(os.path.dirname(staged), exist_ok=True)
    booster.save_model(staged)

    print("\nVerifying the candidate...")
    checks = verify_model(staged)
    print(json.dumps(checks, indent=2))
    if not (checks["num_features_ok"] and checks["feature_names_ok"]):
        print("\nABORT: candidate failed the contract check; live model untouched.")
        os.remove(staged)
        return 1

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "horizon_bars": horizon,
        "horizon_stats": horizon_stats,
        "sl_multiplier": ATR_SL_BASE_MULTIPLIER,
        "tp_multiplier": ATR_TP_BASE_MULTIPLIER,
        "features": list(spec.FEATURE_NAMES),
        "validation": validation,
        "walk_forward": wf,
        "model_checks": checks,
    }

    if args.allow_synthetic and not provenance["real"]:
        os.remove(staged)
        print("\nSYNTHETIC RUN — pipeline verified, live model untouched.")
        print(json.dumps({"walk_forward_summary": {
            k: wf.get(k) for k in ("mean_roc_auc", "mean_pr_auc", "mean_accuracy",
                                   "mean_majority_baseline", "mean_f1")},
            "checks": checks}, indent=2))
        return 0

    if args.dry_run:
        os.remove(staged)
        print("\nDRY RUN — live model untouched.")
        print(json.dumps({"walk_forward": wf, "checks": checks}, indent=2))
        return 0

    if os.path.exists(MODEL_PATH):
        os.makedirs(BACKUP_DIR, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(BACKUP_DIR, f"entry_model_before_{stamp}.json")
        shutil.copy2(MODEL_PATH, backup)
        print(f"\nBacked up previous model -> {backup}")
        report["previous_model_backup"] = backup

    shutil.move(staged, MODEL_PATH)
    print(f"Installed -> {MODEL_PATH}")

    print("\nLive inference check...")
    live = live_inference_check()
    report["live_inference"] = live
    print(json.dumps(live, indent=2))

    gate_ok = all(
        v[d]["status"] == "OK"
        for v in live.values() for d in ("BUY", "SELL")
    )
    direction_ok = all(v["direction_changes_p_win"] for v in live.values())

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nReport -> {REPORT_PATH}")

    if not gate_ok:
        print("\nWARNING: ML gate did not return OK for every case. Investigate "
              "before trading.")
        return 1
    if not direction_ok:
        print("\nWARNING: p_win is identical for BUY and SELL on some symbol — "
              "the model is not direction-aware.")
        return 1

    print("\nAll checks passed. ML_GATE_INVALID should be gone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
