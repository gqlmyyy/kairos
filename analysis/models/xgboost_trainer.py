# Trading Bot V3 - analysis/models/xgboost_trainer.py

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

from analysis.features.ml_dataset_builder import build_dataset_from_db, build_ml_row



def debug_dataset():
    """Debug execution_dataset -> ML-row conversion.

    Returns a dict:
    {
      "total_rows": ...,
      "trainable_rows": ...,
      "rejected_rows": ...,
      "reasons": {order_id: {"reason":..., "missing_fields":[...]}}
    }

    No training logic is modified.
    """

    from data.storage.database import get_conn

    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM execution_dataset")
    rows = c.fetchall()
    cols = [d[0] for d in c.description]

    total = len(rows)
    rejected = 0
    trainable = 0
    reasons: Dict[str, Any] = {}

    def row_to_dict(r):
        return {cols[i]: r[i] for i in range(len(cols))}

    # Mirror ml_dataset_builder expectations: critical features and categorical encoding keys.
    # We detect rejection cause by running build_ml_row() and then recomputing missing fields.
    critical = [
        "rsi",
        "atr",
        "macd",
        "trend_strength",
        "momentum_score",
        "volatility_score",
        "spread",
        "ai_score",
        "sentiment_score",
        "news_impact_score",
        "market_regime",
        "session",
    ]

    for r in rows:
        rd = row_to_dict(r)
        order_id = rd.get("order_id")

        built = build_ml_row(rd)
        if built is None:
            rejected += 1
            # Compute missing critical fields similarly to ml_dataset_builder STRICT_MODE
            missing = []
            for k in critical:
                actual_key = f"actual_{k}"
                if actual_key in rd and rd.get(actual_key) is not None:
                    continue
                expected_key = f"expected_{k}"
                if expected_key not in rd or rd.get(expected_key) is None:
                    missing.append(k)

            reasons[str(order_id)] = {
                "reason": "build_ml_row returned None (STRICT_MODE missing critical features)",
                "missing_fields": missing,
            }
        else:
            trainable += 1

    conn.close()

    return {
        "total_rows": total,
        "trainable_rows": trainable,
        "rejected_rows": rejected,
        "reasons": reasons,
    }


logger = get_logger("xgboost_trainer")



# This trainer is LEGACY. It builds a 12-feature vector from the
# `execution_dataset` table and labels it `actual_pnl > 0` — a different schema
# and a different target from the 10 features the live path sends. It used to
# default to models/entry/entry_model.json and overwrite it in place.
#
# It now writes to a research directory. Nothing here reaches production
# without production_model_guard.install(), which would reject this schema
# anyway. Keeping the trainer runnable preserves the ability to study the
# execution dataset; keeping it pointed at production preserved nothing.
DEFAULT_MODEL_PATH = os.path.join(
    "models", "entry", "research", "legacy_execution_dataset", "entry_model.json")
DEFAULT_FEATURE_IMPORTANCE_PATH = os.path.join(
    "models", "entry", "research", "legacy_execution_dataset", "feature_importance.json")


def _ensure_models_dir():
    os.makedirs("models", exist_ok=True)


def _infer_task_from_y(y: List[float]) -> str:
    """Infer classification/regression.

    If y looks like binary {0,1} (or close), use classification.
    Otherwise regression.
    """
    uniq = set()
    for v in y:
        try:
            fv = float(v)
        except Exception:
            continue
        if fv in (0.0, 1.0):
            uniq.add(float(fv))

    if uniq.issubset({0.0, 1.0}) and len(uniq) >= 1:
        return "classification"
    return "regression"


def evaluate_model(model: Any, task: str, X_test, y_test) -> Dict[str, Any]:
    """Basic evaluation.

    For classification: computes accuracy, precision/recall and profit-based metric via
    win-rate simulation using predicted p_win.

    For regression: computes simple MSE/MAE.
    """
    try:
        import numpy as np  # type: ignore
    except Exception:
        logger.warning("numpy missing - evaluation limited")
        np = None  # type: ignore

    metrics: Dict[str, Any] = {}

    if task == "classification":
        # Use probabilities if model supports it.
        # For xgboost.Booster: predict(DMatrix) might output probabilities.
        # For safety, we coerce into p_win.
        if np is None:
            return {"ok": False, "reason": "numpy not available"}

        dtest = None
        try:
            import xgboost as xgb  # type: ignore

            dtest = xgb.DMatrix(np.asarray(X_test, dtype=float), label=np.asarray(y_test, dtype=float))
            raw = model.predict(dtest)
        except Exception as e:
            return {"ok": False, "reason": f"prediction failed: {e}"}

        pred = np.asarray(raw).ravel()

        # If pred is 0..1 treat as p_win; else threshold directly.
        p_win = pred
        if pred.min() < 0.0 or pred.max() > 1.0:
            # heuristic sigmoid
            p_win = 1.0 / (1.0 + np.exp(-pred))

        y_true = np.asarray(y_test, dtype=float)
        y_pred = (p_win >= 0.5).astype(float)

        tp = float(((y_pred == 1.0) & (y_true == 1.0)).sum())
        tn = float(((y_pred == 0.0) & (y_true == 0.0)).sum())
        fp = float(((y_pred == 1.0) & (y_true == 0.0)).sum())
        fn = float(((y_pred == 0.0) & (y_true == 1.0)).sum())

        acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)

        # Profit-based win-rate simulation:
        # If y_test represents pnl (not binary), this is not meaningful. But task==classification implies y∈{0,1}.
        win_rate = float((y_pred[y_true == 1.0].size))  # unused
        # Better: win-rate among predicted-positive trades
        pred_pos = p_win >= 0.5
        if pred_pos.any():
            win_rate = float(y_true[pred_pos].mean())
        else:
            win_rate = 0.0

        metrics.update(
            {
                "task": task,
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "win_rate_pred_positive": float(win_rate),
            }
        )
        return metrics

    # regression
    if np is None:
        return {"ok": False, "reason": "numpy not available"}

    try:
        import xgboost as xgb  # type: ignore

        dtest = xgb.DMatrix(np.asarray(X_test, dtype=float))
        pred = np.asarray(model.predict(dtest)).ravel()
        y_true = np.asarray(y_test, dtype=float)

        mse = float(((pred - y_true) ** 2).mean())
        mae = float(np.abs(pred - y_true).mean())
        metrics.update({"task": task, "mse": mse, "mae": mae})
        return metrics
    except Exception as e:
        return {"ok": False, "reason": f"regression evaluation failed: {e}"}


def get_feature_importance(model: Any) -> Dict[str, float]:
    """Extract feature importance from xgboost model."""
    try:
        # Booster.get_score returns importance by feature index/name.
        score = model.get_score(importance_type="gain")
        # score keys are like 'f0', 'f1', ...
        out: Dict[str, float] = {}
        for k, v in score.items():
            out[k] = float(v)

        return out
    except Exception as e:
        logger.warning("Failed to get feature importance: %s", e)
        return {}


FORCE_TRAIN = True


def train_model_from_db(
    strict_mode: bool = True,
    model_out: str = DEFAULT_MODEL_PATH,
    feature_importance_out: str = DEFAULT_FEATURE_IMPORTANCE_PATH,
    test_size: float = 0.25,
    min_rows: int = 50,
    retrain_row_threshold: int = 50,
) -> Dict[str, Any]:
    """Train XGBoost model from execution_dataset.

    Uses build_dataset_from_db(strict_mode=True) => X/y.
    Target y:
      - if y is binary {0,1}: classification
      - else regression

    Model versioning:
      - saves to models/xgb_model_v{timestamp}.json
      - updates models/entry/entry_model.json as latest pointer
    """

    _ensure_models_dir()

    logger.info("XGBoost training started (FORCE_TRAIN=%s, min_rows=%s)", str(FORCE_TRAIN), str(min_rows))

    X, y = build_dataset_from_db(strict_mode=strict_mode)
    total = len(X)

    effective_min_rows = int(min_rows) if min_rows is not None else 0
    can_force = bool(FORCE_TRAIN)

    if total < effective_min_rows:
        logger.warning(
            "Not enough rows to train: total=%d < min_rows=%d (FORCE_TRAIN=%s)",
            total,
            effective_min_rows,
            str(can_force),
        )
        if not can_force:
            return {"ok": False, "reason": "not_enough_data", "total": total}
        # else: continue training even with too few rows

    if total < 10 and not can_force:
        return {"ok": False, "reason": "dataset_too_small", "total": total}


    # naive split (shuffle)
    import random

    idx = list(range(total))
    random.shuffle(idx)

    test_n = max(1, int(total * test_size))
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    task = _infer_task_from_y(y)

    try:
        import numpy as np  # type: ignore
        import xgboost as xgb  # type: ignore
    except Exception as e:
        logger.error("xgboost/numpy required for training: %s", e)
        return {"ok": False, "reason": f"missing_deps: {e}"}

    X_train_arr = np.asarray(X_train, dtype=float)
    y_train_arr = np.asarray(y_train, dtype=float)
    if X_train_arr.ndim == 1:
        X_train_arr = X_train_arr.reshape(1, -1)

    # Guard: ensure feature dimension is 12 (expected by ml_dataset_builder)
    if X_train_arr.ndim != 2 or X_train_arr.shape[1] == 0:
        logger.error("Invalid X_train_arr shape: %s", getattr(X_train_arr, 'shape', None))
        return {"ok": False, "reason": f"invalid_X_train_shape: {getattr(X_train_arr, 'shape', None)}"}

    dtrain = xgb.DMatrix(X_train_arr, label=y_train_arr)

    dtest = xgb.DMatrix(np.asarray(X_test, dtype=float), label=np.asarray(y_test, dtype=float))

    # If dataset split results in empty test set, avoid DMatrix shape errors by reusing train.
    # Ensure feature matrix has correct 2D shape and non-zero columns.
    if X_train_arr.ndim != 2 or X_train_arr.shape[1] == 0:
        return {"ok": False, "reason": f"invalid_X_train_shape: {X_train_arr.shape}"}

    # Ensure test matrix shape is valid too.
    X_test_arr = np.asarray(X_test, dtype=float)
    if X_test_arr.ndim == 1:
        X_test_arr = X_test_arr.reshape(1, -1)

    if X_test_arr.shape[0] == 0 or X_test_arr.shape[1] == 0:
        dtest = dtrain
    else:
        dtest = xgb.DMatrix(X_test_arr, label=np.asarray(y_test, dtype=float))


    params: Dict[str, Any] = {

        "max_depth": 4,

        "eta": 0.05,

        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "seed": 42,
    }

    if task == "classification":
        # Ensure binary labels
        y_train_bin = np.asarray(y_train, dtype=float)
        y_test_bin = np.asarray(y_test, dtype=float)
        # If not strictly binary, map pnl->win.
        if not set(np.unique(y_train_bin)).issubset({0.0, 1.0}):
            y_train_bin = (y_train_bin > 0).astype(float)
            y_test_bin = (y_test_bin > 0).astype(float)
            dtrain = xgb.DMatrix(np.asarray(X_train, dtype=float), label=y_train_bin)
            dtest = xgb.DMatrix(np.asarray(X_test, dtype=float), label=y_test_bin)

        params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
    else:
        params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})

    timestamp = int(time.time())
    version_path = f"models/xgb_model_v{timestamp}.json"

    num_boost_round = 200
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dtest, "test")], verbose_eval=False)

    # Save versioned model
    model.save_model(version_path)

    saved_ok = os.path.exists(version_path)
    if saved_ok:
        logger.info("Saved model version: %s", version_path)
    else:
        logger.warning("Model version file missing after save_model: %s", version_path)

    # Refuse to become a writer to the production artifact again, whatever
    # `model_out` was set to by a caller.
    from analysis.models.production_model_guard import assert_not_production
    assert_not_production(model_out)

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    with open(version_path, "rb") as fsrc:
        with open(model_out, "wb") as fdst:
            fdst.write(fsrc.read())

    # Evaluate
    metrics = evaluate_model(model, task=task, X_test=X_test, y_test=y_test)

    # Feature importance
    fi = get_feature_importance(model)
    try:
        with open(feature_importance_out, "w", encoding="utf-8") as f:
            json.dump(fi, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("Failed writing feature_importance: %s", e)

    # Also store training distribution stats (mean/std per feature) for drift detection
    try:
        stats_out = feature_importance_out.replace("feature_importance.json", "training_feature_stats.json")
        # compute mean/std for X_train
        import numpy as np  # type: ignore
        arr = np.asarray(X_train, dtype=float)
        feat_stats: Dict[str, Any] = {"features": {}}
        for j in range(arr.shape[1]):
            col = arr[:, j]
            feat_stats["features"][str(j)] = {
                "mean": float(col.mean()),
                "std": float(col.std(ddof=0)),
            }
        with open(stats_out, "w", encoding="utf-8") as f:
            json.dump(feat_stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("Failed writing training_feature_stats: %s", e)

    logger.info("XGBoost training completed: total=%d train=%d test=%d task=%s metrics=%s", total, len(X_train), len(X_test), task, metrics)

    # Ensure models/entry/entry_model.json exists after training
    try:
        latest_exists = os.path.exists(model_out)
        if latest_exists:
            logger.info("Saved latest model pointer: %s", model_out)
        else:
            logger.warning("Latest model pointer not found after training: %s", model_out)
    except Exception:
        pass


    return {
        "ok": True,
        "total": total,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "task": task,
        "metrics": metrics,
        "model_path": model_out,
        "version_path": version_path,
        "feature_importance_path": feature_importance_out,
    }


# ------------------------------
# Auto retrain trigger helpers
# ------------------------------

def should_retrain(
    new_rows_count: int,
    last_train_ts: Optional[float],
    max_elapsed_hours: float = 6.0,
    row_threshold: int = 50,
) -> bool:
    """Whether enough genuinely new data has arrived to justify retraining.

    `new_rows_count` must be a delta since the last training run. The caller
    used to pass the *total* row count, so the first branch was true on every
    call from the moment the table held 50 rows; and an unknown
    `last_train_ts` returned True as well. Between them the function was a
    constant `True` wearing the shape of a decision.

    Unknown training time is now treated as "cannot establish that retraining
    is due", which is the fail-closed reading: retraining replaces a model, so
    absence of evidence must not authorise it.
    """
    if new_rows_count < 0:
        raise ValueError(f"new_rows_count must be a non-negative delta, got {new_rows_count}")

    if new_rows_count >= row_threshold:
        return True
    if last_train_ts is None:
        return False
    elapsed_hours = (time.time() - float(last_train_ts)) / 3600.0
    return elapsed_hours >= max_elapsed_hours and new_rows_count > 0

