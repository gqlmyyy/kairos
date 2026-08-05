from __future__ import annotations

"""analysis/entry_v2/entry_calibration.py

Probability calibration for Entry v2.

This module:
- Loads trained XGBoost booster (entry_model.json)
- Loads dataset CSV and recreates chronological splits
- Fits BOTH Platt scaling and Isotonic regression on validation
- Chooses automatically the better calibrator using validation Brier score (primary)
  and LogLoss (secondary)
- Stores:
  - calibration.pkl
  - calibration_report.md
  - threshold artifacts are produced in separate module.

Calibration ONLY. No runtime integration.

Constraints honored:
- Do NOT modify Exit.
- Do NOT implement threshold selection here.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

from .feature_schema import FEATURE_COLUMNS

logger = get_logger("entry_v2.entry_calibration")


@dataclass(frozen=True)
class CalibrationConfig:
    label_column: str = "label"
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1

    # Choice criteria
    primary_metric: str = "brier"  # lower is better
    secondary_metric: str = "logloss"  # lower is better


DEFAULT_BOOSTER_PATH = "models/entry/entry_model.json"


def _load_dataset(dataset_csv_path: str):
    import pandas as pd  # type: ignore

    df = pd.read_csv(dataset_csv_path)
    for col in ["t", "symbol", "label"]:
        if col not in df.columns:
            raise RuntimeError(f"dataset missing required column: {col}")
    return df


def _chronological_split(df, *, config: CalibrationConfig):
    df = df.sort_values(["t", "symbol"]).reset_index(drop=True)
    n = len(df)
    n_train = int(n * config.train_frac)
    n_val = int(n * config.val_frac)
    n_test = n - n_train - n_val
    if n_test <= 0 or n_val <= 0:
        raise RuntimeError(f"Bad split sizes: n={n} train={n_train} val={n_val} test={n_test}")
    return df.iloc[:n_train].copy(), df.iloc[n_train : n_train + n_val].copy(), df.iloc[n_train + n_val :].copy()


def _prepare_X_y(df, feature_columns: List[str], label_column: str):
    import numpy as np  # type: ignore

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required feature columns: {missing[:50]}")

    y = df[label_column].values.astype(float)
    X = df[feature_columns].values.astype(float)
    # ensure binary-ish
    y = (y >= 0.5).astype(float)
    return X, y


def _predict_proba_raw(booster: Any, X_val) -> List[float]:
    import numpy as np  # type: ignore
    import xgboost as xgb  # type: ignore

    dval = xgb.DMatrix(X_val)
    raw = booster.predict(dval)
    raw = np.asarray(raw).ravel()

    # If booster outputs outside [0,1], interpret as margin and sigmoid it.
    out = []
    for p in raw.tolist():
        if p < 0.0 or p > 1.0:
            import math

            p = 1.0 / (1.0 + math.exp(-p))
        out.append(float(max(0.0, min(1.0, float(p)))))
    return out


def _brier_score(y_true: List[float], y_prob: List[float]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum((float(yt) - float(p)) ** 2 for yt, p in zip(y_true, y_prob)) / n


def _log_loss(y_true: List[float], y_prob: List[float]) -> float:
    import math

    eps = 1e-15
    n = len(y_true)
    if n == 0:
        return 0.0
    loss = 0.0
    for yt, p in zip(y_true, y_prob):
        p = min(max(float(p), eps), 1.0 - eps)
        loss += -(float(yt) * math.log(p) + (1.0 - float(yt)) * math.log(1.0 - p))
    return loss / n


def calibrate_entry_v2(
    *,
    dataset_csv_path: str,
    booster_path: str = DEFAULT_BOOSTER_PATH,
    output_dir: str = "models/entry/",
    feature_columns: Optional[List[str]] = None,
    config: CalibrationConfig = CalibrationConfig(),
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    # load booster
    import xgboost as xgb  # type: ignore

    if not os.path.exists(booster_path):
        raise RuntimeError(f"Booster model missing: {booster_path}")

    booster = xgb.Booster()
    booster.load_model(booster_path)

    df = _load_dataset(dataset_csv_path)
    train_df, val_df, _test_df = _chronological_split(df, config=config)

    X_train, y_train = _prepare_X_y(train_df, feature_columns, config.label_column)
    X_val, y_val = _prepare_X_y(val_df, feature_columns, config.label_column)

    # raw probabilities on validation
    p_val_raw = _predict_proba_raw(booster, X_val)

    # Fit calibrators
    from sklearn.isotonic import IsotonicRegression  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    import numpy as np  # type: ignore

    y_val_list = [float(v) for v in y_val.tolist()]

    # Platt scaling: logistic regression on logit(raw)
    # Convert probabilities to log-odds (with eps clamp)
    eps = 1e-15
    p = np.asarray(p_val_raw, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))

    platt = LogisticRegression(solver="lbfgs")
    platt.fit(logit.reshape(-1, 1), np.asarray(y_val_list, dtype=float))

    # Predict calibrated
    logit_pred = platt.predict_proba(logit.reshape(-1, 1))[:, 1]
    p_val_platt = [float(max(0.0, min(1.0, v))) for v in logit_pred.tolist()]

    brier_platt = _brier_score(y_val_list, p_val_platt)
    logloss_platt = _log_loss(y_val_list, p_val_platt)

    # Isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val_raw, np.asarray(y_val_list, dtype=float))

    p_val_iso = [float(max(0.0, min(1.0, v))) for v in iso.predict(p_val_raw).tolist()]
    brier_iso = _brier_score(y_val_list, p_val_iso)
    logloss_iso = _log_loss(y_val_list, p_val_iso)

    # Choose
    # primary: brier
    best = "platt"
    best_metrics = {
        "platt": {"brier": brier_platt, "logloss": logloss_platt},
        "isotonic": {"brier": brier_iso, "logloss": logloss_iso},
    }

    if brier_iso < brier_platt:
        best = "isotonic"
    elif abs(brier_iso - brier_platt) <= 1e-12:
        # tie-break logloss
        best = "isotonic" if logloss_iso < logloss_platt else "platt"

    # Store calibration.pkl
    import pickle

    calibration_obj: Dict[str, Any]
    calibration_obj = {
        "method": best,
        "feature_columns": feature_columns,
        "label_column": config.label_column,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if best == "platt":
        calibration_obj["platt"] = platt
        calibration_obj["isotonic"] = None
    else:
        calibration_obj["isotonic"] = iso
        calibration_obj["platt"] = None

    calib_path = os.path.join(output_dir, "calibration.pkl")
    with open(calib_path, "wb") as f:
        pickle.dump(calibration_obj, f)

    # Report
    report_path = os.path.join(output_dir, "calibration_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Entry v2 Calibration Report\n\n")
        f.write(f"Booster: {booster_path}\n\n")
        f.write("## Validation metrics (un-calibrated raw probabilities)\n")
        f.write("(raw metrics not computed here; only calibrator metrics are compared)\n\n")
        f.write("## Calibrators\n\n")
        f.write(f"- Platt scaling: Brier={brier_platt:.6f}, LogLoss={logloss_platt:.6f}\n")
        f.write(f"- Isotonic regression: Brier={brier_iso:.6f}, LogLoss={logloss_iso:.6f}\n\n")
        f.write(f"## Selected calibrator\n\n- method: {best}\n")

    # Return
    return {
        "ok": True,
        "best_method": best,
        "brier_platt": brier_platt,
        "brier_isotonic": brier_iso,
        "logloss_platt": logloss_platt,
        "logloss_isotonic": logloss_iso,
        "calibration_path": calib_path,
        "calibration_report_path": report_path,
    }


if __name__ == "__main__":
    raise SystemExit("Use calibrate_entry_v2(dataset_csv_path=..., booster_path=..., output_dir=...)")

