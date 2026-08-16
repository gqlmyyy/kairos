from __future__ import annotations

"""scripts/verify_entry_model_v2.py

Strict verification for Entry v2 model robustness.

Runs (sequential, no pauses):
1) Data source audit rerun on data/entry_v2/labeled_dataset.csv
2) Leakage check: feature list from analysis/entry_v2/feature_schema.py only.
   Verify that label-related/non-feature columns are NOT in feature list.
   Also print full feature columns used.
3) 5-fold CV with conservative XGBoost hyperparams + early stopping.
4) Baseline: majority-class accuracy.
5) Random 30-vector sanity: variance/const check.
6) Final unified report.

Decision:
- If leakage_detected == False and CV Std < ~0.02 and mean accuracy > baseline + clear margin => keep existing model.
- Else retrain using best params with full dataset and overwrite models/entry/entry_model.json.

This script is designed to be deterministic.
"""

import csv
import json
import math
import os
import random
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np  # type: ignore

# ensure repo root on path
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis.entry_v2.dataset_audit import audit_dataset, AuditConfig
from analysis.entry_v2.feature_schema import FEATURE_COLUMNS


DATASET_CSV = os.path.join(REPO_ROOT, "data", "entry_v2", "labeled_dataset.csv")
AUDIT_OUT_DIR = os.path.join(REPO_ROOT, "data", "entry_v2", "audit_reports")

# QUARANTINED alongside analysis/entry_v2 — this script trains on the same
# invalidated dataset and used to overwrite the production model in place, with
# no metadata, no backup and no gate. It now writes a research artifact.
ENTRY_MODEL_PATH = os.path.join(
    REPO_ROOT, "models", "entry", "research", "legacy_verify_v2", "entry_model.json")
ENTRY_MODEL_FEATURE_STATS_PATH = os.path.join(REPO_ROOT, "models", "entry", "training_feature_stats.json")

RANDOM_SEED = 42
N_FOLDS = 5
MAX_CONSTANT_FEATURE_RATIO = 0.20
MIN_ROWS_TO_TRAIN = 50

# conservative search space
MAX_DEPTH_RANGE = [2, 3, 4, 5]
N_ESTIMATORS_RANGE = [50, 100, 150, 200, 250]
EARLY_STOPPING_ROUNDS = 15
REG_ALPHA_RANGE = [0.1, 0.5, 1.0]
REG_LAMBDA_RANGE = [0.1, 1.0, 5.0]

N_RANDOM_VECTORS = 30


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _safe_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        if isinstance(x, str):
            s = x.strip()
            if s == "" or s.lower() in {"none", "null", "nan"}:
                return default
            return float(s)
        return float(x)
    except Exception:
        return default


def _load_dataset_rows() -> List[Dict[str, Any]]:
    with open(DATASET_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _prepare_X_y(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    # label is 0/1 float; cast to int
    if "label" not in rows[0]:
        raise RuntimeError("labeled_dataset.csv missing 'label' column")

    X = []
    y = []

    for r in rows:
        vec = [_safe_float(r.get(col), 0.0) for col in FEATURE_COLUMNS]
        X.append(vec)
        y.append(1.0 if _safe_float(r.get("label"), 0.0) >= 0.5 else 0.0)

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)

    stats = {
        "rows": int(len(rows)),
        "n_features": int(X_arr.shape[1]) if X_arr.ndim == 2 else 0,
        "label_win": int((y_arr == 1.0).sum()),
        "label_loss": int((y_arr == 0.0).sum()),
    }
    return X_arr, y_arr, stats


def _majority_baseline(y: np.ndarray) -> Tuple[int, float]:
    n1 = int((y == 1.0).sum())
    n0 = int((y == 0.0).sum())
    if n1 >= n0:
        return 1, n1 / max(n1 + n0, 1)
    return 0, n0 / max(n1 + n0, 1)


def _logloss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _train_xgb_booster(X: np.ndarray, y: np.ndarray, params: Dict[str, Any], n_estimators: int, eval_X: np.ndarray, eval_y: np.ndarray) -> Any:
    import xgboost as xgb  # type: ignore

    dtr = xgb.DMatrix(X, label=y)
    dva = xgb.DMatrix(eval_X, label=eval_y)

    booster = xgb.train(
        params,
        dtr,
        num_boost_round=int(n_estimators),
        evals=[(dva, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    return booster


def _predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    import xgboost as xgb  # type: ignore

    d = xgboost.DMatrix(X) if False else None
    dtest = xgb.DMatrix(X)
    p = model.predict(dtest)
    return np.asarray(p, dtype=np.float32).reshape(-1)


def _cv_search_and_train(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    import xgboost as xgb  # type: ignore

    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(idx)

    folds = np.array_split(idx, N_FOLDS)

    best: Dict[str, Any] = {
        "mean_accuracy": -1.0,
        "mean_logloss": float("inf"),
        "params": None,
        "folds": None,
    }

    for max_depth in MAX_DEPTH_RANGE:
        for n_estimators in N_ESTIMATORS_RANGE:
            for reg_alpha in REG_ALPHA_RANGE:
                for reg_lambda in REG_LAMBDA_RANGE:
                    fold_accs: List[float] = []
                    fold_lls: List[float] = []

                    for f in range(N_FOLDS):
                        val_idx = folds[f]
                        train_idx = np.hstack([folds[j] for j in range(N_FOLDS) if j != f])

                        Xtr, ytr = X[train_idx], y[train_idx]
                        Xva, yva = X[val_idx], y[val_idx]

                        params = {
                            "objective": "binary:logistic",
                            "eval_metric": "logloss",
                            "max_depth": int(max_depth),
                            "eta": 0.05,
                            "subsample": 0.9,
                            "colsample_bytree": 0.9,
                            "min_child_weight": 5,
                            "seed": RANDOM_SEED,
                            "reg_alpha": float(reg_alpha),
                            "reg_lambda": float(reg_lambda),
                        }

                        booster = _train_xgb_booster(
                            X=Xtr,
                            y=ytr,
                            params=params,
                            n_estimators=n_estimators,
                            eval_X=Xva,
                            eval_y=yva,
                        )

                        prob = booster.predict(xgb.DMatrix(Xva))
                        prob = np.asarray(prob, dtype=np.float32).reshape(-1)
                        pred = (prob >= 0.5).astype(np.float32)
                        acc = float((pred == yva).mean())
                        ll = _logloss(yva, prob)

                        fold_accs.append(acc)
                        fold_lls.append(ll)

                    mean_acc = float(np.mean(fold_accs))
                    std_acc = float(np.std(fold_accs))
                    mean_ll = float(np.mean(fold_lls))

                    # Selection: maximize mean acc; tie-break lower mean logloss; also prefer lower std
                    improved = False
                    if mean_acc > best["mean_accuracy"]:
                        improved = True
                    elif mean_acc == best["mean_accuracy"]:
                        if mean_ll < best["mean_logloss"]:
                            improved = True

                    if improved:
                        best.update(
                            {
                                "mean_accuracy": mean_acc,
                                "mean_logloss": mean_ll,
                                "std_accuracy": std_acc,
                                "params": {
                                    "max_depth": max_depth,
                                    "n_estimators": n_estimators,
                                    "reg_alpha": reg_alpha,
                                    "reg_lambda": reg_lambda,
                                    "eta": 0.05,
                                },
                                "folds": {
                                    "fold_accuracy": fold_accs,
                                    "fold_logloss": fold_lls,
                                },
                            }
                        )

    return best


def _train_final_model(X: np.ndarray, y: np.ndarray, best_params: Dict[str, Any]) -> Any:
    import xgboost as xgb  # type: ignore

    d = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(best_params["max_depth"]),
        "eta": float(best_params.get("eta", 0.05)),
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 5,
        "seed": RANDOM_SEED,
        "reg_alpha": float(best_params["reg_alpha"]),
        "reg_lambda": float(best_params["reg_lambda"]),
    }

    booster = xgb.train(params, d, num_boost_round=int(best_params["n_estimators"]))
    return booster


def _save_booster_to_entry_json(booster: Any, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    from analysis.models.production_model_guard import assert_not_production
    assert_not_production(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    booster.save_model(tmp)
    # replace atomically-ish
    os.replace(tmp, path)


def _random_vector_sanity(X: np.ndarray, best_params: Dict[str, Any]) -> Dict[str, Any]:
    # Use feature-wise range sampling from dataset
    import xgboost as xgb  # type: ignore

    rng = np.random.default_rng(RANDOM_SEED)
    feat_min = X.min(axis=0)
    feat_max = X.max(axis=0)

    # Create random vectors in feature ranges
    vecs = rng.random((N_RANDOM_VECTORS, X.shape[1]), dtype=np.float32)
    Xrand = feat_min + vecs * (feat_max - feat_min + 1e-12)

    # load trained model from disk (caller ensures it exists)
    if not os.path.exists(ENTRY_MODEL_PATH):
        raise RuntimeError("entry_model.json not found for random sanity test")

    booster = xgb.Booster()
    booster.load_model(ENTRY_MODEL_PATH)

    p = booster.predict(xgb.DMatrix(Xrand))
    p = np.asarray(p, dtype=np.float32).reshape(-1)

    variance = float(np.var(p))
    return {
        "n_vectors": int(len(p)),
        "variance": variance,
        "p_min": float(p.min()),
        "p_max": float(p.max()),
        "p_mean": float(p.mean()),
        "const_false": variance > 0.0,
        "sample_probs_first5": [float(x) for x in p[:5].tolist()],
    }


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    report: Dict[str, Any] = {
        "run_ts": datetime.now().isoformat(),
        "data": {
            "dataset_csv": DATASET_CSV,
            "exists": os.path.exists(DATASET_CSV),
            "n_features": len(FEATURE_COLUMNS),
        },
        "leakage_check": {
            "features_used": list(FEATURE_COLUMNS),
            "label_columns_present_in_dataset": [],
            "suspected_future_leak_columns": [],
        },
        "audit": None,
        "cv": None,
        "baseline": None,
        "random_30_vector": None,
        "decision": None,
    }

    if not os.path.exists(DATASET_CSV):
        raise RuntimeError(f"Dataset CSV not found: {DATASET_CSV}")

    # STEP 1: rerun audit
    _ensure_dir(AUDIT_OUT_DIR)
    audit = audit_dataset(
        dataset_csv_path=DATASET_CSV,
        output_dir=AUDIT_OUT_DIR,
        label_column="label",
        config=AuditConfig(class_imbalance_max_ratio=0.9),
    )
    report["audit"] = audit

    # load dataset rows
    rows = _load_dataset_rows()
    if len(rows) == 0:
        raise RuntimeError("Dataset loaded 0 rows")

    # STEP 2: Leakage explicit check (schema-based)
    # verify any label-related columns are not in FEATURE_COLUMNS
    non_feature_cols = {
        "t",
        "symbol",
        "label",
        "label_reason",
        "holding_bars",
        "tp_hit",
        "sl_hit",
        "exit_timestamp",
    }
    leaked = []
    for col in FEATURE_COLUMNS:
        if col in non_feature_cols:
            leaked.append(col)
    report["leakage_check"]["suspected_future_leak_columns"] = leaked
    report["leakage_check"]["leakage_detected"] = bool(leaked)

    # STEP 3/4: CV and baseline
    X, y, stats = _prepare_X_y(rows)

    if stats["rows"] < MIN_ROWS_TO_TRAIN:
        report["decision"] = {"action": "stop", "reason": "not_enough_rows"}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    baseline_majority, baseline_acc = _majority_baseline(y)
    report["baseline"] = {
        "majority_class": baseline_majority,
        "baseline_accuracy": float(baseline_acc),
        "label_win": stats["label_win"],
        "label_loss": stats["label_loss"],
    }

    cv_best = _cv_search_and_train(X, y)

    report["cv"] = {
        "mean_accuracy": cv_best.get("mean_accuracy"),
        "std_accuracy": cv_best.get("std_accuracy"),
        "mean_logloss": cv_best.get("mean_logloss"),
        "best_params": cv_best.get("params"),
        "folds": cv_best.get("folds"),
    }

    # STEP 6: random vector sanity requires model present
    # STEP 7 decision
    std_acc = float(cv_best.get("std_accuracy", 1.0))
    mean_acc = float(cv_best.get("mean_accuracy", 0.0))
    baseline = float(report["baseline"]["baseline_accuracy"])

    leakage_detected = bool(report["leakage_check"]["leakage_detected"])

    action = "keep"
    reason = []

    if leakage_detected:
        action = "retrain"
        reason.append("leakage_detected_by_schema")
    if std_acc >= 0.02:
        action = "retrain"
        reason.append(f"cv_std_too_high:{std_acc:.6f}")
    if mean_acc <= baseline + 0.05:
        action = "retrain"
        reason.append(f"model_not_better_than_baseline:mean_acc={mean_acc:.4f},baseline={baseline:.4f}")

    # Retrain if needed
    if action == "retrain":
        booster = _train_final_model(X, y, best_params=cv_best["params"])
        _save_booster_to_entry_json(booster, ENTRY_MODEL_PATH)

    # ensure model exists even if keep
    if not os.path.exists(ENTRY_MODEL_PATH):
        # keep mode might still lack file
        booster = _train_final_model(X, y, best_params=cv_best["params"])
        _save_booster_to_entry_json(booster, ENTRY_MODEL_PATH)

    sanity = _random_vector_sanity(X, best_params=cv_best["params"])
    report["random_30_vector"] = sanity

    report["decision"] = {
        "action": action,
        "retrain_reason": reason,
        "keep_model_entry_path": ENTRY_MODEL_PATH,
    }

    print("===== FINAL ENTRY V2 VERIFICATION REPORT =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

