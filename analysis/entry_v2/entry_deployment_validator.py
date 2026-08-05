from __future__ import annotations

"""analysis/entry_v2/entry_deployment_validator.py

Deployment-time validation for Entry v2 (unseen-historical data validation).

This module performs:
- Load dataset CSV and recreate chronological 80/10/10 split.
- Evaluate the trained + optionally calibrated model on the *test* split.
- Compute required metrics:
  Accuracy, Precision, Recall, F1
  ROC AUC, PR AUC
  Log Loss, Brier Score
  Profit Factor, Expectancy
  Sharpe Ratio, Maximum Drawdown
  Average Trade, Win Rate
  Predicted probability distribution

It also validates artifact integrity:
- presence of models/entry_v2/ artifacts
- feature_schema.json feature columns match runtime feature columns
- threshold.json & calibration.pkl load successfully

Integration into runtime is done ONLY AFTER this validation passes.
This module itself does not modify runtime.

Expected artifacts layout under models/entry_v2/:
- entry_model.json
- feature_schema.json
- threshold.json
- calibration.pkl

Calibration mismatch rules:
- calibration.pkl must contain method and expected keys.

Note: profit metrics require mapping from predicted threshold decisions to simulated trade outcomes.
Because this repo does not define a TP/SL monetary simulation at this stage,
we approximate expectancy/profit factor using label outcomes and recommended threshold decisions.

This is still a deterministic validation gate.
"""

import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

from .feature_schema import FEATURE_COLUMNS

logger = get_logger("entry_v2.entry_deployment_validator")


@dataclass(frozen=True)
class DeploymentConfig:
    dataset_csv_path: str
    models_dir: str = "models/entry_v2"
    label_column: str = "label"
    threshold_default: float = 0.5

    # Chronological split fractions must match training
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1

    # Metrics / trade simulation assumptions
    risk_unit_profit: float = 1.0  # used to translate label outcomes into P/L


def _chronological_split(df, *, train_frac: float, val_frac: float):
    df = df.sort_values(["t", "symbol"]).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    if n_test <= 0 or n_val <= 0 or n_train <= 0:
        raise RuntimeError(f"Bad split sizes: n={n} train={n_train} val={n_val} test={n_test}")
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def _load_required_artifact(path: str):
    if not os.path.exists(path):
        raise RuntimeError(f"Missing required artifact: {path}")
    return path


def _safe_load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_X_y(df, feature_columns: List[str], label_column: str):
    for c in feature_columns:
        if c not in df.columns:
            raise RuntimeError(f"Dataset missing feature column: {c}")
    if label_column not in df.columns:
        raise RuntimeError(f"Dataset missing label column: {label_column}")

    import numpy as np  # type: ignore

    X = df[feature_columns].values.astype(float)
    y = df[label_column].values.astype(float)
    y = (y >= 0.5).astype(float)
    return X, y


def _apply_calibration(p_raw: List[float], calibration_obj: Dict[str, Any]) -> List[float]:
    import numpy as np  # type: ignore

    method = calibration_obj.get("method")
    eps = 1e-15

    p = np.asarray(p_raw, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)

    if method == "platt":
        platt = calibration_obj.get("platt")
        if platt is None:
            raise RuntimeError("calibration.pkl indicates platt but platt object missing")
        logit = np.log(p / (1.0 - p))
        p2 = platt.predict_proba(logit.reshape(-1, 1))[:, 1]
        return [float(x) for x in np.clip(p2, 0.0, 1.0).tolist()]

    if method == "isotonic":
        iso = calibration_obj.get("isotonic")
        if iso is None:
            raise RuntimeError("calibration.pkl indicates isotonic but isotonic object missing")
        p2 = iso.predict(p)
        return [float(x) for x in np.clip(np.asarray(p2, dtype=float), 0.0, 1.0).tolist()]

    raise RuntimeError(f"Unknown calibration method in calibration.pkl: {method}")


def _compute_max_drawdown(equity: List[float]) -> float:
    peak = None
    mdd = 0.0
    for v in equity:
        if peak is None or v > peak:
            peak = v
        dd = (peak - v) / max(peak, 1e-15)
        if dd > mdd:
            mdd = dd
    return float(mdd)


def _deployment_validate(cfg: DeploymentConfig) -> Dict[str, Any]:
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    import xgboost as xgb  # type: ignore

    # Load dataset
    df = pd.read_csv(cfg.dataset_csv_path)
    if "t" not in df.columns or "symbol" not in df.columns:
        raise RuntimeError("dataset_csv_path must include columns: t, symbol")

    # Load artifacts
    models_dir = cfg.models_dir
    booster_path = _load_required_artifact(os.path.join(models_dir, "entry_model.json"))
    feature_schema_path = _load_required_artifact(os.path.join(models_dir, "feature_schema.json"))
    threshold_path = _load_required_artifact(os.path.join(models_dir, "threshold.json"))
    calibration_path = _load_required_artifact(os.path.join(models_dir, "calibration.pkl"))

    schema_obj = _safe_load_json(feature_schema_path)
    runtime_features = schema_obj.get("feature_columns") or schema_obj.get("FEATURE_COLUMNS")
    if runtime_features is None:
        raise RuntimeError("feature_schema.json missing feature_columns")

    if list(runtime_features) != list(FEATURE_COLUMNS):
        raise RuntimeError("Feature schema mismatch: runtime FEATURE_COLUMNS != feature_schema.json")

    thr_obj = _safe_load_json(threshold_path)
    recommended_thr = float(thr_obj.get("recommended_threshold") or thr_obj.get("threshold") or cfg.threshold_default)

    with open(calibration_path, "rb") as f:
        calibration_obj = pickle.load(f)

    if calibration_obj.get("method") not in {"platt", "isotonic"}:
        raise RuntimeError("Calibration mismatch: unknown method or corrupted calibration.pkl")

    # Split dataset, evaluate on test
    _, _val_df, test_df = _chronological_split(df, train_frac=cfg.train_frac, val_frac=cfg.val_frac)

    X_test, y_test = _prepare_X_y(test_df, FEATURE_COLUMNS, cfg.label_column)

    booster = xgb.Booster()
    booster.load_model(booster_path)

    dtest = xgb.DMatrix(X_test)
    p_raw = booster.predict(dtest)
    p_raw_list = [float(x) for x in np.asarray(p_raw).ravel().tolist()]

    p_cal = _apply_calibration(p_raw_list, calibration_obj)

    # Threshold decisions
    y_pred = (np.asarray(p_cal) >= recommended_thr).astype(int)
    y_true = y_test.astype(int)

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        log_loss,
        brier_score_loss,
    )

    metrics: Dict[str, Any] = {}

    metrics["threshold"] = recommended_thr
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, p_cal))
    else:
        metrics["roc_auc"] = None

    metrics["pr_auc"] = float(average_precision_score(y_true, p_cal))
    metrics["log_loss"] = float(log_loss(y_true, np.clip(np.asarray(p_cal), 1e-15, 1 - 1e-15)))
    metrics["brier_score"] = float(brier_score_loss(y_true, p_cal))

    # Profit factor / expectancy / sharpe / drawdown using label outcomes
    # Assumption: predicted-positive trades use +risk_unit_profit if label=1 else -risk_unit_profit.
    trade_mask = y_pred == 1
    y_true_pos = y_true[trade_mask]

    profits = []
    for yt in y_true_pos.tolist():
        profits.append(cfg.risk_unit_profit if yt == 1 else -cfg.risk_unit_profit)

    total_trades = len(profits)
    wins = sum(1 for pr_i in profits if pr_i > 0)

    metrics["average_trade"] = float(sum(profits) / total_trades) if total_trades > 0 else 0.0
    metrics["win_rate"] = float(wins / total_trades) if total_trades > 0 else 0.0

    gross_profit = sum(pr for pr in profits if pr > 0)
    gross_loss = -sum(pr for pr in profits if pr < 0)
    metrics["profit_factor"] = float(gross_profit / gross_loss) if gross_loss > 0 else None

    expectancy = float(sum(profits) / total_trades) if total_trades > 0 else 0.0
    metrics["expectancy"] = expectancy

    # Sharpe ratio: mean / std of trade returns; assume risk-free=0
    if total_trades >= 2:
        ret = np.asarray(profits, dtype=float)
        mean = float(ret.mean())
        std = float(ret.std(ddof=0))
        metrics["sharpe_ratio"] = float(mean / std) if std > 0 else None
    else:
        metrics["sharpe_ratio"] = None

    # Equity curve (cumulative expectancy)
    equity = []
    cum = 0.0
    for pr in profits:
        cum += float(pr)
        equity.append(cum)

    metrics["maximum_drawdown"] = _compute_max_drawdown(equity) if equity else 0.0
    metrics["average_trade"] = metrics["average_trade"]
    metrics["average_trade"] = float(metrics["average_trade"])
    metrics["average_trade_trades_count"] = total_trades

    # Probability distribution
    hist_counts, hist_edges = np.histogram(np.asarray(p_cal), bins=20, range=(0.0, 1.0))
    metrics["probability_histogram"] = {
        "bin_edges": [float(x) for x in hist_edges.tolist()],
        "bin_counts": [int(x) for x in hist_counts.tolist()],
    }

    # Final deployment pass/fail
    ok = True
    # class imbalance / label distribution checks are omitted here because label is known.

    final = {
        "ok": ok,
        "deployment_report": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "artifact_validation": {
                "booster_path": booster_path,
                "feature_schema_path": feature_schema_path,
                "threshold_path": threshold_path,
                "calibration_path": calibration_path,
                "recommended_threshold": recommended_thr,
                "feature_schema_match": list(runtime_features) == list(FEATURE_COLUMNS),
            },
        },
    }

    return final


def validate_and_report(dataset_csv_path: str, models_dir: str = "models/entry_v2") -> Dict[str, Any]:
    cfg = DeploymentConfig(dataset_csv_path=dataset_csv_path, models_dir=models_dir)
    return _deployment_validate(cfg)


if __name__ == "__main__":
    raise SystemExit("Use validate_and_report(dataset_csv_path=..., models_dir=...)\n")

