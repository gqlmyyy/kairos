# Trading Bot V3 - analysis/models/xgboost_inference.py

from __future__ import annotations

from typing import Any, Dict, Optional

from utils.logger import get_logger

from analysis.features.ml_dataset_builder import build_ml_row
from analysis.models.model_manager import load_latest_model, get_cached_model_version

logger = get_logger("xgboost_inference")


def load_model(path: str = "models/entry/entry_model.json"):
    """Load an XGBoost model.

    Tries:
      - xgboost.Booster.load_model(path)

    Fallback:
      - if file missing or xgboost import fails, returns None.

    Notes:
      - This project currently expects XGBoost native Booster.
    """
    try:
        import os
        import xgboost as xgb  # type: ignore

        if not path:
            logger.warning("Model path is empty")
            return None

        if not os.path.exists(path):
            logger.warning("Model file not found: %s", path)
            return None

        booster = xgb.Booster()
        booster.load_model(path)
        return booster
    except Exception as e:
        logger.warning("Failed to load xgboost model: %s", e)
        return None


def predict_trade(execution_row: Dict[str, Any], model=None, model_version: Optional[str] = None) -> Dict[str, float]:

    """Predict win probability for a single execution row.

    Returns:
      {
        'p_win': float,
        'p_loss': float,
        'confidence': float,
        'raw_score': float,
      }

    Fallback logic:
      - If model missing/unavailable, returns conservative defaults.
    """
    if model is None:
        # Conservative fallback: below typical threshold.
        return {"p_win": 0.0, "p_loss": 1.0, "confidence": 0.0, "raw_score": 0.0}

    built = build_ml_row(execution_row)
    if built is None:
        return {"p_win": 0.0, "p_loss": 1.0, "confidence": 0.0, "raw_score": 0.0}

    X, _y = built

    # XGBoost Booster predict API:
    # - For binary classification, booster.predict returns probabilities (depends on model objective).
    # - To keep generic, we interpret output:
    #   - if 2D (n,2): take p_win = out[:,1]
    #   - if 1D (n,): treat as probability/logit; convert using sigmoid if needed heuristically.
    try:
        import numpy as np  # type: ignore

        import xgboost as xgb  # type: ignore

        dmat = xgb.DMatrix(np.asarray([X], dtype=float))
        raw = model.predict(dmat)


        # raw could be shape (1,) or (1,2)
        raw_arr = np.asarray(raw)
        raw_score = float(raw_arr.ravel()[0]) if raw_arr.size > 0 else 0.0

        if raw_arr.ndim == 2 and raw_arr.shape[1] >= 2:
            p_win = float(raw_arr[0, 1])
            p_loss = float(raw_arr[0, 0])
            confidence = abs(p_win - p_loss)
            return {
                "p_win": p_win,
                "p_loss": p_loss,
                "confidence": confidence,
                "raw_score": raw_score,
            }

        # 1D case
        p = float(raw_score)

        # Heuristic: if p outside [0,1], assume it's logit and apply sigmoid.
        if p < 0.0 or p > 1.0:
            import math

            p = 1.0 / (1.0 + math.exp(-p))

        p_win = p
        p_loss = 1.0 - p
        confidence = abs(p_win - p_loss)

        return {
            "p_win": float(p_win),
            "p_loss": float(p_loss),
            "confidence": float(confidence),
            "raw_score": float(raw_score),
        }
    except Exception as e:
        logger.warning("predict_trade failed: %s", e)
        return {"p_win": 0.0, "p_loss": 1.0, "confidence": 0.0, "raw_score": 0.0}


def _maybe_hot_reload(model, model_version: Optional[str]):
    try:
        cached = get_cached_model_version()
        if model_version is None:
            model_version = cached
        if cached is not None and model_version is not None and cached != model_version:
            m, _v = load_latest_model()
            return m, cached
    except Exception:
        pass
    return model, model_version


def should_trade(p_win: float, threshold: float = 0.30) -> bool:
    try:
        return float(p_win) >= float(threshold)
    except Exception:
        return False

