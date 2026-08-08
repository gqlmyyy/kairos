"""XGBoost Exit Model -Expert decision maker for trade closure

This module provides an expert XGBoost model that predicts the probability of a trade
reversing and ending in a loss. It's consulted ONLY when other features raise "red flags".

The model is trained on historical closed trades with features including:
- MFE (Max Favorable Excursion)
- MAE (Max Adverse Excursion)
- ATR (Average True Range)
- Trade Health Score
- Market Regime
- RSI
- Time Open (hours)
- Spread
- News Impact
- Profit Decay %

The label is: 1 if trade closed at a loss after reaching peak profit, 0 otherwise.
"""

from __future__ import annotations

import os
import pickle
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger
from config import DB_FILE

from analysis.models.feature_schema import FEATURE_ORDER, build_feature_vector



logger = get_logger("xgboost_exit_model")

# Model file path - file is at analysis/models/xgboost_exit_model.py
# Need 3 levels: models -> analysis -> project_root -> models
_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) ,
    "models"
)
MODEL_PATH = os.path.join(_MODELS_DIR, "exit", "exit_model.json")
FALLBACK_THRESHOLD_PROBABILITY = 0.90  # If profit_decay > 70% and health < 40

# Cached model instance - loaded once
_cached_model = None

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None
    logger.warning("XGBoost not available; using fallback rules only")


def _get_db_connection():
    """Get database connection."""
    import sqlite3
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def _collect_training_data(execution_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Collect features and labels from execution data for training.

    Args:
        execution_data: List of execution records from database.

    Returns:
        (X, y) tuple of features and labels.
    """
    features_list = []
    labels = []

    for row in execution_data:
        try:
            # Extract values used by schema
            expected_atr = float(row.get("expected_atr", 0) or 0)
            actual_atr = float(row.get("actual_atr", 0) or 0)
            entry_atr = actual_atr if actual_atr > 0 else expected_atr

            actual_pnl = float(row.get("actual_pnl", 0) or 0)

            # Calculate MFE/MAE if available (from indicators JSON)
            mfe = 0.0
            mae = 0.0
            try:
                indicators = row.get("actual_indicators_json") or row.get("expected_indicators_json")
                if indicators:
                    ind = json.loads(indicators) if isinstance(indicators, str) else indicators
                    mfe = float(ind.get("mfe", 0) or 0)
                    mae = float(ind.get("mae", 0) or 0)
            except Exception:
                pass

            spread = float(row.get("spread_at_entry", 0) or row.get("actual_spread", 0) or 0)

            # time_open_hours
            time_open_hours = 0.0
            try:
                dataset_created = row.get("dataset_created_at") or row.get("created_at")
                dataset_updated = row.get("dataset_updated_at") or row.get("closed_at")
                if dataset_created and dataset_updated:
                    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                        try:
                            from datetime import datetime
                            dt1 = datetime.strptime(str(dataset_created), fmt)
                            dt2 = datetime.strptime(str(dataset_updated), fmt)
                            time_open_hours = max(0.0, (dt2 - dt1).total_seconds() / 3600.0)
                            break
                        except Exception:
                            pass
            except Exception:
                pass

            entry_rsi = float(row.get("actual_rsi", 0) or row.get("expected_rsi", 0) or 0)

            # market_regime (schema encodes)
            market_regime = str(row.get("actual_market_regime", "") or row.get("expected_market_regime", ""))

            # volume
            volume = 0.0
            try:
                order_id = str(row.get("order_id", ""))
                if order_id:
                    conn = _get_db_connection()
                    c = conn.execute("SELECT size FROM trades WHERE order_id=?", (order_id,))
                    tr = c.fetchone()
                    if tr:
                        volume = float(tr["size"] or 0)
                    conn.close()
            except Exception:
                pass

            # schema fields (missing adx/session/trends in history are filled as 0.0 for training rows)
            entry_adx = float(row.get("expected_adx", None) or row.get("actual_adx", None) or 0.0)
            session = row.get("expected_session", None) or row.get("actual_session", None) or row.get("session", None)
            trend_h1 = float(row.get("expected_trend_h1", None) or row.get("actual_trend_h1", None) or row.get("trend_h1", None) or 0.0)
            trend_h4 = float(row.get("expected_trend_h4", None) or row.get("actual_trend_h4", None) or row.get("trend_h4", None) or 0.0)

            trade_duration = time_open_hours

            vec_list = build_feature_vector(
                {
                    "mfe": mfe,
                    "mae": mae,
                    "entry_atr": entry_atr,
                    "entry_rsi": entry_rsi,
                    "entry_adx": entry_adx,
                    "market_regime": market_regime,
                    "trade_duration": trade_duration,
                    "spread": spread,
                    "volume": volume,
                    "session": session,
                    "trend_h1": trend_h1,
                    "trend_h4": trend_h4,
                }
            )

            features_list.append(vec_list)

            # label
            label = 1 if actual_pnl < 0 else 0
            labels.append(label)

        except Exception as e:
            logger.debug(f"Skipping row due to error: {e}")
            continue

    if not features_list:
        return np.array([]), np.array([])


    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    return X, y


def train_exit_model(execution_data: List[Dict]) -> bool:

    """Train XGBoost exit model on historical execution data.

    Args:
        execution_data: List of execution records from database.
        Each record should be a dict with fields like:
        - symbol, direction, expected_atr, actual_atr, actual_pnl, actual_entry, actual_exit
        - mfe, mae (from indicators), actual_rsi, spread_at_entry, etc.

    Returns:
        True if training succeeded, False otherwise.
    """
    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost not available; cannot train model")
        return False

    if not execution_data:
        logger.error("No execution data provided for training")
        return False

    logger.info(f"Training Exit Model on {len(execution_data)} records...")

    try:
        X, y = _collect_training_data(execution_data)

        if X.shape[0] < 10:
            logger.error(f"Not enough samples for training: {X.shape[0]}")
            return False

        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Positive labels: {y.sum()}/{len(y)}")

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)

        # XGBoost parameters for binary classification
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "seed": 42,
            "verbosity": 1,
        }

        # Train
        bst = xgb.train(params, dtrain, num_boost_round=100)

        # Save model
        os.makedirs(_MODELS_DIR, exist_ok=True)
        bst.save_model(MODEL_PATH)

        logger.info(f"Exit Model saved to {MODEL_PATH}")
        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False


def _load_model():
    """Load trained XGBoost model (cached).

    Returns:
        XGBoost model object or None if not available.
    """
    global _cached_model

    # Return cached model if already loaded
    if _cached_model is not None:
        return _cached_model

    if not XGBOOST_AVAILABLE:
        return None

    if not os.path.exists(MODEL_PATH):
        logger.info("Exit Model not found, will use fallback")
        return None

    try:
        bst = xgb.Booster()
        bst.load_model(MODEL_PATH)
        _cached_model = bst  # Cache it
        logger.info(f"Loaded Exit Model from {MODEL_PATH}")
        return bst
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def _compute_fallback_probability(
    profit_decay_pct: float,
    trade_health: float,
    mfe: float,
    mae: float,
) -> float:
    """Compute fallback probability when model is not available.

    Fallback rule: if profit_decay > 70% and trade_health < 40, return 90%.

    Args:
        profit_decay_pct: Current profit as % of peak (0-100)
        trade_health: Trade health score (0-100)
        mfe: Max Favorable Excursion
        mae: Max Adverse Excursion

    Returns:
        Probability of reversal (0-1)
    """
    # High decay + low health = high probability of reversal
    if profit_decay_pct < 30 and trade_health < 40:
        return 0.90
    if profit_decay_pct < 50 and trade_health < 30:
        return 0.90

    # MAE > MFE indicates bad trade
    if mfe > 0 and mae > mfe * 2:
        return 0.80

    # Moderate cases
    if profit_decay_pct < 70 or trade_health < 50:
        return 0.60

    return 0.30


def predict_exit_probability(
    features: Dict[str, Any],
) -> float:
    """Predict probability that trade will reverse and end in loss.

    Args:
        features: Dictionary containing:
            - symbol: str
            - direction: str ("buy" or "sell")
            - atr: float (ATR value)
            - rsi: float (RSI)
            - mfe: float (Max Favorable Excursion)
            - mae: float (Max Adverse Excursion)
            - trade_health: float (0-100)
            - profit_decay_pct: float (currentprofit/peak*100)
            - time_open_hours: float
            - spread: float
            - news_impact: float
            - market_regime: str
            - volume: float

    Returns:
        Probability (0-1) that trade will reverse. Values:
        - > 0.9: strong sell signal
        - 0.7-0.9: moderate (move SL closer)
        - < 0.7: hold
    """
    # Try to load model
    bst = _load_model()

    if bst is None:
        # Use fallback
        profit_decay_pct = features.get("profit_decay_pct", 100.0)
        trade_health = features.get("trade_health", 50.0)
        mfe = features.get("mfe", 0.0)
        mae = features.get("mae", 0.0)

        prob = _compute_fallback_probability(
            profit_decay_pct=profit_decay_pct,
            trade_health=trade_health,
            mfe=mfe,
            mae=mae,
        )

        logger.info(f"[EXIT_MODEL] Using fallback: probability={prob:.1%}")
        return prob

    try:
        # Map adapter fields -> schema names expected by the model.
        # Keep schema single-source-of-truth in feature_schema.py
        vec_list = build_feature_vector(
            {
                **features,
                # normalize input keys used across the project
                "entry_atr": features.get("atr"),
                "entry_rsi": features.get("rsi"),
                "entry_adx": features.get("adx"),
                "trade_duration": features.get("time_open_hours"),
            }
        )


        # Convert to 2D matrix

        feature_vec = np.asarray([vec_list], dtype=np.float32)
        feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=1e6, neginf=-1e6)

        # Assertion protection against silent mismatch
        assert int(feature_vec.shape[1]) == int(bst.num_features()), (
            f"Exit model feature mismatch: vector_len={feature_vec.shape[1]} bst.num_features={bst.num_features()} "
            f"FEATURE_ORDER={FEATURE_ORDER}"
        )

        dtest = xgb.DMatrix(feature_vec, feature_names=FEATURE_ORDER)
        prob = float(bst.predict(dtest)[0])


        prob = float(max(0.0, min(1.0, prob)))

        logger.info(f"[EXIT_MODEL] XGBoost prediction: probability={prob:.1%}")
        return prob

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        # Fallback
        profit_decay_pct = features.get("profit_decay_pct", 100.0)
        trade_health = features.get("trade_health", 50.0)
        mfe = features.get("mfe", 0.0)
        mae = features.get("mae", 0.0)

        return _compute_fallback_probability(
            profit_decay_pct=profit_decay_pct,
            trade_health=trade_health,
            mfe=mfe,
            mae=mae,
        )


def is_model_trained() -> bool:
    """Check if Exit Model is trained and available."""
    return os.path.exists(MODEL_PATH) and XGBOOST_AVAILABLE


def get_model_info() -> Dict[str, Any]:
    """Get model information."""
    if not os.path.exists(MODEL_PATH):
        return {"trained": False, "path": MODEL_PATH}

    try:
        stat = os.stat(MODEL_PATH)
        return {
            "trained": True,
            "path": MODEL_PATH,
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
        }
    except Exception:
        return {"trained": False, "path": MODEL_PATH}