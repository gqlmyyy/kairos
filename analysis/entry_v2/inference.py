from __future__ import annotations

"""Entry v2 runtime inference.

This module is fully independent from legacy Entry (v1) and does not touch Exit.

Contract used by main.py:
  - predict_with_entry_v2(...) -> {"p_win": float, "available": bool, "raw_score": float?}
  - It must apply probability calibration and load the recommended threshold artifact
    for thresholding (thresholding can still be done in main.py if needed).

Artifacts layout (models/entry_v2/):
  - versions/<model_version>/entry_model.json
  - versions/<model_version>/calibration.json
  - versions/<model_version>/threshold.json

To keep integration minimal, this module provides:
  - get_entry_threshold()
  - predict_with_entry_v2(...)

If artifacts are missing, it returns conservative defaults.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from utils.logger import get_logger

logger = get_logger("entry_v2.inference")


@dataclass
class _Artifacts:
    model_path: str
    calibration_path: str
    threshold_path: str
    feature_order_path: Optional[str] = None


_DEFAULT_MODEL_VERSION = os.getenv("ENTRY_V2_MODEL_VERSION", "latest")


def _artifacts_base() -> str:
    return os.getenv("ENTRY_V2_ARTIFACTS_BASE", "models/entry_v2")


def _resolve_artifacts() -> _Artifacts:
    base = _artifacts_base()
    model_version = _DEFAULT_MODEL_VERSION

    # Support two layouts:
    # 1) base/versions/<version>/...
    # 2) base/<version>/...
    v1 = os.path.join(base, "versions", model_version)
    v2 = os.path.join(base, model_version)

    chosen = v1 if os.path.isdir(v1) else v2

    model_path = os.path.join(chosen, "entry_model.json")
    calibration_path = os.path.join(chosen, "calibration.json")
    threshold_path = os.path.join(chosen, "threshold.json")
    feature_order_path = os.path.join(chosen, "feature_order.json")

    return _Artifacts(
        model_path=model_path,
        calibration_path=calibration_path,
        threshold_path=threshold_path,
        feature_order_path=feature_order_path if os.path.exists(feature_order_path) else None,
    )


_cached: Dict[str, Any] = {
    "loaded": False,
    "booster": None,
    "calibration": None,
    "threshold": None,
    "feature_order": None,
    "artifacts": None,
}


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load json %s: %s", path, e)
        return None


def _load_model(model_path: str):
    try:
        import xgboost as xgb  # type: ignore

        if not os.path.exists(model_path):
            return None
        booster = xgb.Booster()
        booster.load_model(model_path)
        return booster
    except Exception as e:
        logger.warning("Failed to load xgboost booster from %s: %s", model_path, e)
        return None


def _apply_calibration(p_raw: float, calibration: Optional[Dict[str, Any]]) -> float:
    if calibration is None:
        return p_raw

    method = calibration.get("method")
    try:
        if method == "platt":
            # p = sigmoid(a * logit(p_raw) + b)
            a = float(calibration.get("a"))
            b = float(calibration.get("b"))
            import math

            eps = 1e-9
            pr = min(max(float(p_raw), eps), 1.0 - eps)
            logit = math.log(pr / (1.0 - pr))
            z = a * logit + b
            calibrated = 1.0 / (1.0 + math.exp(-z))
            return float(calibrated)

        if method == "isotonic":
            # Simple isotonic lookup (store thresholds and values)
            xs = calibration.get("x") or []
            ys = calibration.get("y") or []
            if not xs or not ys or len(xs) != len(ys):
                return p_raw
            # piecewise linear interpolation
            import numpy as np  # type: ignore

            x_arr = np.asarray(xs, dtype=float)
            y_arr = np.asarray(ys, dtype=float)
            calibrated = float(np.interp([p_raw], x_arr, y_arr)[0])
            return calibrated

    except Exception as e:
        logger.warning("Calibration apply failed: %s", e)

    return p_raw


def _load_everything(force_reload: bool = False):
    if _cached["loaded"] and not force_reload:
        return

    artifacts = _resolve_artifacts()
    booster = _load_model(artifacts.model_path)
    calibration = _load_json(artifacts.calibration_path)
    threshold = _load_json(artifacts.threshold_path)

    feature_order = None
    if artifacts.feature_order_path and os.path.exists(artifacts.feature_order_path):
        fo = _load_json(artifacts.feature_order_path)
        if isinstance(fo, dict) and "feature_order" in fo:
            feature_order = fo.get("feature_order")

    _cached.update(
        {
            "loaded": True,
            "booster": booster,
            "calibration": calibration,
            "threshold": threshold,
            "feature_order": feature_order,
            "artifacts": artifacts,
        }
    )


def get_entry_threshold() -> Optional[float]:
    _load_everything()
    th = _cached.get("threshold")
    if not th:
        return None
    try:
        t = th.get("recommended_threshold")
        if t is None:
            return None
        return float(t)
    except Exception:
        return None


def _default_threshold() -> float:
    # Conservative default; but should be replaced by artifact in real deployments.
    return 0.5


def predict_with_entry_v2(
    *,
    rsi: float,
    atr: float,
    macd: float,
    trend_strength: float,
    trend_score: float,
    momentum_score: float,
    volatility_score: float,
    market_regime: str,
    direction: str,
) -> Dict[str, Any]:
    """Predict calibrated win probability for Entry v2.

    Note: feature mapping below is a placeholder until feature_schema v2 is implemented.
    The training pipeline in this PR will create feature_order.json and ensure the
    runtime uses it.
    """

    _load_everything()
    booster = _cached.get("booster")
    if booster is None:
        return {"p_win": 0.0, "available": False}

    # Temporary feature vector consistent with legacy main inputs.
    # v2 feature_schema will replace this mapping with the exact feature order.
    try:
        import xgboost as xgb  # type: ignore
        import numpy as np  # type: ignore

        features = [
            float(rsi or 0),
            float(atr or 0),
            float(macd or 0),
            float(trend_strength or 0),
            float(trend_score or 50),
            float(momentum_score or 50),
            float(volatility_score or 50),
            float(0.0),  # regime placeholder
            float(0.0),  # session placeholder
            float(1.0) if str(direction).lower().startswith("buy") else float(0.0),
        ]

        dmat = xgb.DMatrix(np.asarray([features], dtype=float))
        raw_pred = float(booster.predict(dmat)[0])

        p_cal = _apply_calibration(raw_pred, _cached.get("calibration"))
        p_cal = float(max(0.0, min(1.0, p_cal)))

        return {"p_win": p_cal, "available": True, "raw_score": raw_pred}

    except Exception as e:
        logger.warning("entry_v2 predict failed: %s", e)
        return {"p_win": 0.0, "available": False}

