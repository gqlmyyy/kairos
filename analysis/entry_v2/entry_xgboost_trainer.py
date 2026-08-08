from __future__ import annotations

"""analysis/entry_v2/entry_xgboost_trainer.py

Production-grade XGBoost training for Entry v2.

Strict rules:
- Entry v2 ONLY.
- No Exit model modifications.
- No calibration (done in later step).
- Chronological split: 80/10/10 with no shuffle.
- Optuna hyperparameter optimization with early stopping.

Inputs:
- feature dataset CSV exported by feature generation stage.
- must include label column `label`.

Outputs to output_dir:
- entry_model.json (xgboost native Booster JSON)
- metadata.json
- best_params.json
- training_log.json

No calibration.

Notes on XGBoost:
- Uses xgboost.Booster / xgboost.train.
- Objective: binary:logistic
- eval_metric: logloss

"""

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

from .feature_schema import FEATURE_COLUMNS, validate_feature_columns

logger = get_logger("entry_v2.entry_xgboost_trainer")


@dataclass(frozen=True)
class TrainConfig:
    label_column: str = "label"
    # Split fractions
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1

    # Optuna
    n_trials: int = 25
    random_seed: int = 42

    # Early stopping
    early_stopping_rounds: int = 25
    max_boost_round: int = 5000

    # Data filtering
    min_rows: int = 200


DEFAULT_OUTPUT_MODEL_PATH = "models/entry/entry_model.json"


def _ensure_dirs(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def _load_dataset(dataset_csv_path: str):
    import pandas as pd  # type: ignore

    df = pd.read_csv(dataset_csv_path)
    if "t" not in df.columns:
        raise RuntimeError("dataset must contain column 't' for chronological split")
    if "symbol" not in df.columns:
        raise RuntimeError("dataset must contain column 'symbol'")
    return df


def _chronological_split(df, *, config: TrainConfig):
    # Ensure chronological order by t (and tie-break by symbol)
    df = df.sort_values(["t", "symbol"]).reset_index(drop=True)

    n = len(df)
    n_train = int(n * config.train_frac)
    n_val = int(n * config.val_frac)
    n_test = n - n_train - n_val

    if n_test <= 0 or n_val <= 0 or n_train <= 0:
        raise RuntimeError(f"Bad split sizes: n={n} train={n_train} val={n_val} test={n_test}")

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()

    return train_df, val_df, test_df


def _prepare_xy(df, feature_columns: List[str], label_column: str):
    import numpy as np  # type: ignore

    missing_feats = [c for c in feature_columns if c not in df.columns]
    if missing_feats:
        raise RuntimeError(f"Missing required feature columns: {missing_feats[:20]} (total {len(missing_feats)})")

    # Drop rows where label is missing/NaN
    df = df.copy()
    y = df[label_column].values
    mask = ~pd_is_nan(y)
    df = df.loc[mask]

    X = df[feature_columns].values
    y = df[label_column].values.astype(float)

    X = X.astype(float)

    # enforce binary
    y = (y >= 0.5).astype(float)

    return X, y, df


def pd_is_nan(arr):
    # minimal helper without importing pandas in module scope
    try:
        import numpy as np  # type: ignore

        return np.isnan(arr.astype(float))
    except Exception:
        return [False] * len(arr)


def _train_once(
    X_train,
    y_train,
    X_val,
    y_val,
    params: Dict[str, Any],
    config: TrainConfig,
):
    import numpy as np  # type: ignore
    import xgboost as xgb  # type: ignore

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    evals = [(dval, "val")]

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=config.max_boost_round,
        evals=evals,
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=False,
    )

    # best_score is lower for logloss
    best_score = float(booster.best_score)
    best_iteration = int(booster.best_iteration)

    return booster, best_score, best_iteration


def _objective_factory(
    X_train,
    y_train,
    X_val,
    y_val,
    feature_columns: List[str],
    config: TrainConfig,
):
    import optuna  # type: ignore

    def objective(trial: "optuna.trial.Trial") -> float:
        # Tune parameters only; keep stable ones fixed
        params: Dict[str, Any] = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "seed": config.random_seed,
            "verbosity": 0,
            # Tuned
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 50.0, log=True),
            "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 10.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
        }

        # Note: xgboost.train uses num_boost_round separately. We map n_estimators to num_boost_round.
        num_boost_round = int(params.pop("n_estimators"))

        booster, best_score, _best_iter = _train_once(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params=params,
            config=TrainConfig(
                label_column=config.label_column,
                train_frac=config.train_frac,
                val_frac=config.val_frac,
                test_frac=config.test_frac,
                n_trials=config.n_trials,
                random_seed=config.random_seed,
                early_stopping_rounds=config.early_stopping_rounds,
                max_boost_round=max(num_boost_round, 50),
                min_rows=config.min_rows,
            ),
        )

        trial.set_user_attr("best_score", best_score)
        return best_score

    return objective


def train_entry_xgboost(
    *,
    dataset_csv_path: str,
    output_dir: str,
    feature_columns: Optional[List[str]] = None,
    config: TrainConfig = TrainConfig(),
    model_out_path: str = DEFAULT_OUTPUT_MODEL_PATH,
) -> Dict[str, Any]:
    """Train entry v2 model and save artifacts."""

    _ensure_dirs(output_dir)
    validate_feature_columns(feature_columns)

    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    df = _load_dataset(dataset_csv_path)

    if len(df) < config.min_rows:
        raise RuntimeError(f"Dataset too small: rows={len(df)} < min_rows={config.min_rows}")

    train_df, val_df, test_df = _chronological_split(df, config=config)

    # Prepare X/y
    import numpy as np  # type: ignore

    X_train, y_train, _ = _prepare_xy(train_df, feature_columns, config.label_column)
    X_val, y_val, _ = _prepare_xy(val_df, feature_columns, config.label_column)
    X_test, y_test, _ = _prepare_xy(test_df, feature_columns, config.label_column)

    import optuna  # type: ignore

    study = optuna.create_study(direction="minimize")
    objective = _objective_factory(X_train, y_train, X_val, y_val, feature_columns, config)

    study.optimize(objective, n_trials=config.n_trials)

    best_params = study.best_params

    # Re-train on train+val using best params with early stopping on val
    best_params_train: Dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "seed": config.random_seed,
        "verbosity": 0,

        # tuned params
        "max_depth": int(best_params["max_depth"]),
        "learning_rate": float(best_params["learning_rate"]),
        "subsample": float(best_params["subsample"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
        "gamma": float(best_params["gamma"]),
        "min_child_weight": float(best_params["min_child_weight"]),
        "reg_alpha": float(best_params["reg_alpha"]),
        "reg_lambda": float(best_params["reg_lambda"]),
        "max_delta_step": float(best_params["max_delta_step"]),
        "scale_pos_weight": float(best_params["scale_pos_weight"]),
    }

    num_boost_round = int(best_params["n_estimators"])

    import xgboost as xgb  # type: ignore

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    booster, best_score, best_iter = _train_once(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        params=best_params_train,
        config=TrainConfig(
            label_column=config.label_column,
            train_frac=config.train_frac,
            val_frac=config.val_frac,
            test_frac=config.test_frac,
            n_trials=config.n_trials,
            random_seed=config.random_seed,
            early_stopping_rounds=config.early_stopping_rounds,
            max_boost_round=max(num_boost_round, 50),
            min_rows=config.min_rows,
        ),
    )

    # Evaluate on test
    dtest = xgb.DMatrix(X_test, label=y_test)
    test_pred = booster.predict(dtest)
    import numpy as np  # type: ignore

    test_pred = np.asarray(test_pred).ravel()
    test_logloss = float(np.mean(- (y_test * np.log(test_pred + 1e-15) + (1.0 - y_test) * np.log(1.0 - test_pred + 1e-15))))

    # Save artifacts
    ts = int(time.time())
    entry_model_json_path = os.path.join(output_dir, "entry_model.json")
    booster.save_model(entry_model_json_path)

    best_params_path = os.path.join(output_dir, "best_params.json")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    metadata = {
        "builder": "analysis/entry_v2/entry_xgboost_trainer.py",
        "timestamp_utc": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "dataset_csv_path": dataset_csv_path,
        "output_dir": output_dir,
        "model_artifact": entry_model_json_path,
        "model_out_path": model_out_path,
        "split": {
            "train_frac": config.train_frac,
            "val_frac": config.val_frac,
            "test_frac": config.test_frac,
        },
        "feature_columns": feature_columns,
        "best_params": best_params,
        "val_best_logloss": best_score,
        "best_iteration": best_iter,
        "test_logloss": test_logloss,
        "optuna": {
            "n_trials": config.n_trials,
            "optuna_best_value": float(study.best_value),
        },
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    training_log = {
        "optuna_best_value": float(study.best_value),
        "n_trials": config.n_trials,
        "study_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": dict(t.user_attrs),
            }
            for t in study.trials
        ],
        "final": {
            "best_score": best_score,
            "best_iter": best_iter,
            "test_logloss": test_logloss,
        },
    }

    training_log_path = os.path.join(output_dir, "training_log.json")
    with open(training_log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)

    # Copy/overwrite production pointer if desired
    # NOTE: This is Entry-only. Exit unaffected.
    try:
        os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
        with open(entry_model_json_path, "rb") as src:
            with open(model_out_path, "wb") as dst:
                dst.write(src.read())
    except Exception as e:
        logger.warning("Failed to copy entry model to production pointer: %s", e)

    return {
        "ok": True,
        "entry_model_json_path": entry_model_json_path,
        "metadata_path": metadata_path,
        "best_params_path": best_params_path,
        "training_log_path": training_log_path,
        "test_logloss": test_logloss,
        "val_best_logloss": best_score,
        "best_iteration": best_iter,
        "model_out_path": model_out_path,
    }


if __name__ == "__main__":
    raise SystemExit("Use train_entry_xgboost(dataset_csv_path=..., output_dir=...)")

