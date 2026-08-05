from __future__ import annotations

"""analysis/entry_v2/entry_model_reporting.py

Entry v2 complete model reports (reports ONLY).

Generates:
- training_report.md
- optuna_report.md
- feature_importance.csv
- ROC / PR curves
- confusion matrix
- calibration curve
- probability distribution
- feature gain/weight/cover (via Booster.get_score)
- permutation importance (slow)
- SHAP values + summary + dependence plots (requires shap)
- metadata.json in reports dir

Stops after reports; no runtime integration.

Expected inputs:
- dataset_csv_path including label
- trained booster json path
- best_params / optuna log paths optional
- feature_columns optional (defaults to FEATURE_COLUMNS)

Caveats:
- This module may be heavy due to SHAP/permutation.
- If dependencies are missing, it will still generate non-SHAP reports.
"""

import json
import os
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

from .feature_schema import FEATURE_COLUMNS, validate_feature_columns

logger = get_logger("entry_v2.entry_model_reporting")


@dataclass(frozen=True)
class ReportConfig:
    label_column: str = "label"
    output_subdir_shap: str = "shap"
    output_subdir_reports: str = "reports"
    max_shap_samples: int = 2000


def _ensure_dirs(base_out: str) -> Dict[str, str]:
    os.makedirs(base_out, exist_ok=True)
    shap_dir = os.path.join(base_out, ReportConfig().output_subdir_shap)
    reports_dir = os.path.join(base_out, ReportConfig().output_subdir_reports)
    os.makedirs(shap_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    return {"base_out": base_out, "shap_dir": shap_dir, "reports_dir": reports_dir}


def _chronological_split(df, *, train_frac: float = 0.8, val_frac: float = 0.1):
    df = df.sort_values(["t", "symbol"]).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def generate_reports(
    *,
    dataset_csv_path: str,
    booster_path: str,
    output_dir: str,
    feature_columns: Optional[List[str]] = None,
    label_column: str = "label",
    best_params_path: Optional[str] = None,
    training_log_path: Optional[str] = None,
    calibration_model_path: Optional[str] = None,
) -> Dict[str, Any]:

    cfg = ReportConfig(label_column=label_column)

    dirs = _ensure_dirs(output_dir)

    feature_columns = feature_columns or FEATURE_COLUMNS
    validate_feature_columns(feature_columns)

    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    import xgboost as xgb  # type: ignore

    if not os.path.exists(booster_path):
        raise RuntimeError(f"Booster not found: {booster_path}")

    df = pd.read_csv(dataset_csv_path)
    for req in ["t", "symbol", label_column]:
        if req not in df.columns:
            raise RuntimeError(f"Dataset missing required column: {req}")

    for f in feature_columns:
        if f not in df.columns:
            raise RuntimeError(f"Dataset missing feature column: {f}")

    train_df, val_df, test_df = _chronological_split(df)

    # Use validation for PR/ROC etc by default
    eval_df = val_df if len(val_df) > 0 else test_df

    X_eval = eval_df[feature_columns].values.astype(float)
    y_eval = (eval_df[label_column].values.astype(float) >= 0.5).astype(float)

    dmat = xgb.DMatrix(X_eval)
    booster = xgb.Booster()
    booster.load_model(booster_path)
    p_raw = booster.predict(dmat)
    p = np.asarray(p_raw).ravel().astype(float)
    p = np.clip(p, 0.0, 1.0)

    # Optional: apply calibration.pkl if provided
    calibration_info = None
    if calibration_model_path is not None and os.path.exists(calibration_model_path):
        import pickle

        with open(calibration_model_path, "rb") as f:
            calibration_info = pickle.load(f)

        method = calibration_info.get("method")
        # calibration_info stores platt/isotonic objects
        if method == "platt" and calibration_info.get("platt") is not None:
            platt = calibration_info["platt"]
            eps = 1e-15
            logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
            p = platt.predict_proba(logit.reshape(-1, 1))[:, 1]
        elif method == "isotonic" and calibration_info.get("isotonic") is not None:
            iso = calibration_info["isotonic"]
            p = iso.predict(p)
        p = np.clip(np.asarray(p).ravel().astype(float), 0.0, 1.0)

    # Metrics
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        log_loss,
        brier_score_loss,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
    )

    roc_auc = float(roc_auc_score(y_eval, p)) if len(np.unique(y_eval)) > 1 else None
    pr_auc = float(average_precision_score(y_eval, p))
    ll = float(log_loss(y_eval, np.clip(p, 1e-15, 1 - 1e-15)))
    brier = float(brier_score_loss(y_eval, p))

    # Default threshold 0.5 for confusion matrix visuals (threshold optimization happens later)
    thr = 0.5
    y_pred = (p >= thr).astype(int)
    cm = confusion_matrix(y_eval, y_pred)

    # Calibration curve
    from sklearn.calibration import calibration_curve  # type: ignore

    frac_pos, mean_pred = calibration_curve(y_eval, p, n_bins=10, strategy="uniform")

    # Probability distribution
    hist_bins = 20
    hist_counts, hist_edges = np.histogram(p, bins=hist_bins, range=(0.0, 1.0))

    # Save plots via matplotlib if available
    plot_paths: Dict[str, str] = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        # ROC
        if roc_auc is not None:
            fpr, tpr, _ = roc_curve(y_eval, p)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.4f}")
            plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend()
            roc_path = os.path.join(dirs["reports_dir"], "roc_curve.png")
            plt.tight_layout()
            plt.savefig(roc_path)
            plot_paths["roc_curve"] = roc_path
            plt.close()

        # PR
        prec, rec, _ = precision_recall_curve(y_eval, p)
        plt.figure(figsize=(6, 4))
        plt.plot(rec, prec, label=f"PR AUC={pr_auc:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        pr_path = os.path.join(dirs["reports_dir"], "pr_curve.png")
        plt.tight_layout()
        plt.savefig(pr_path)
        plot_paths["pr_curve"] = pr_path
        plt.close()

        # Confusion matrix
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix (thr={thr})")
        plt.colorbar()
        tick_marks = range(cm.shape[0])
        plt.xticks(tick_marks, ["0", "1"])
        plt.yticks(tick_marks, ["0", "1"])
        cm_path = os.path.join(dirs["reports_dir"], "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_path)
        plot_paths["confusion_matrix"] = cm_path
        plt.close()

        # Calibration curve
        plt.figure(figsize=(6, 4))
        plt.plot(mean_pred, frac_pos, marker="o")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Curve")
        cal_path = os.path.join(dirs["reports_dir"], "calibration_curve.png")
        plt.tight_layout()
        plt.savefig(cal_path)
        plot_paths["calibration_curve"] = cal_path
        plt.close()

        # Probability distribution
        plt.figure(figsize=(6, 4))
        plt.hist(p, bins=hist_bins, range=(0.0, 1.0), alpha=0.7)
        plt.xlabel("Predicted probability")
        plt.ylabel("Count")
        pd_path = os.path.join(dirs["reports_dir"], "probability_distribution.png")
        plt.tight_layout()
        plt.savefig(pd_path)
        plot_paths["probability_distribution"] = pd_path
        plt.close()

    except Exception as e:
        logger.warning("Plot generation skipped: %s", e)

    # Feature importance
    # gain/weight/cover
    def _score(importance_type: str) -> Dict[str, float]:
        score = booster.get_score(importance_type=importance_type)
        return {k: float(v) for k, v in score.items()}

    gain = _score("gain")
    weight = _score("weight")
    cover = _score("cover")

    # Convert f{idx} keys to provided feature names by order if possible
    # xgboost Booster feature names typically are f0, f1...
    # We map indices to FEATURE_COLUMNS ordering.
    idx_to_name = {f"f{i}": name for i, name in enumerate(feature_columns)}

    fi_rows: List[Dict[str, Any]] = []
    all_keys = set(gain.keys()) | set(weight.keys()) | set(cover.keys())
    for k in sorted(all_keys):
        name = idx_to_name.get(k, k)
        fi_rows.append(
            {
                "feature": name,
                "xgb_key": k,
                "gain": gain.get(k, 0.0),
                "weight": weight.get(k, 0.0),
                "cover": cover.get(k, 0.0),
            }
        )

    import pandas as pd  # type: ignore

    fi_df = pd.DataFrame(fi_rows)
    fi_csv_path = os.path.join(dirs["reports_dir"], "feature_importance.csv")
    fi_df.to_csv(fi_csv_path, index=False)

    # Permutation importance (optional)
    perm_path = None
    try:
        from sklearn.inspection import permutation_importance  # type: ignore
        from sklearn.base import BaseEstimator  # type: ignore

        # Wrap booster in a predict_proba-like estimator
        class _BoosterWrapper:
            def __init__(self, booster):
                self.booster = booster

            def predict_proba(self, X):
                import numpy as np  # type: ignore
                d = xgb.DMatrix(np.asarray(X, dtype=float))
                p1 = self.booster.predict(d)
                p1 = np.asarray(p1).ravel()
                p0 = 1.0 - p1
                return np.vstack([p0, p1]).T

        wrapper = _BoosterWrapper(booster)

        # Use small sample for speed
        X_imp = X_eval
        if len(X_imp) > 5000:
            X_imp = X_imp[:5000]
            y_imp = y_eval[:5000]
        else:
            y_imp = y_eval

        perm = permutation_importance(wrapper, X_imp, y_imp, scoring="neg_log_loss", n_repeats=5, random_state=42)
        perm_df = pd.DataFrame({"feature": feature_columns, "perm_importance": perm.importances_mean})
        perm_df = perm_df.sort_values("perm_importance", ascending=False)
        perm_path = os.path.join(dirs["reports_dir"], "permutation_importance.csv")
        perm_df.to_csv(perm_path, index=False)
    except Exception as e:
        logger.warning("Permutation importance skipped: %s", e)

    # SHAP (optional)
    shap_summary_path = None
    try:
        import shap  # type: ignore
        import numpy as np  # type: ignore

        # sample
        if len(X_eval) > cfg.max_shap_samples:
            X_shap = X_eval[: cfg.max_shap_samples]
        else:
            X_shap = X_eval

        # TreeExplainer works with xgboost models via booster
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_shap)

        shap_summary_path = os.path.join(dirs["shap_dir"], "shap_summary.png")

        # summary plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        shap.summary_plot(shap_values, features=X_shap, feature_names=feature_columns, show=False)
        plt.tight_layout()
        plt.savefig(shap_summary_path)
        plt.close()

        # dependence plots for top features by mean abs shap
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:5]
        dep_dir = os.path.join(dirs["shap_dir"], "dependence")
        os.makedirs(dep_dir, exist_ok=True)
        for j, idx in enumerate(top_idx):
            feat = feature_columns[idx]
            dep_path = os.path.join(dep_dir, f"dependence_{j}_{feat}.png")
            shap.dependence_plot(idx, shap_values, X_shap, feature_names=feature_columns, show=False, interaction_index=None)
            plt.tight_layout()
            plt.savefig(dep_path)
            plt.close()

    except Exception as e:
        logger.warning("SHAP generation skipped: %s", e)

    # Reports markdown
    ts = datetime.now(timezone.utc).isoformat()
    training_report_md = os.path.join(dirs["reports_dir"], "training_report.md")
    optuna_report_md = os.path.join(dirs["reports_dir"], "optuna_report.md")
    metadata_path = os.path.join(dirs["reports_dir"], "metadata.json")

    best_params = None
    training_log = None
    if best_params_path and os.path.exists(best_params_path):
        with open(best_params_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)
    if training_log_path and os.path.exists(training_log_path):
        with open(training_log_path, "r", encoding="utf-8") as f:
            training_log = json.load(f)

    with open(training_report_md, "w", encoding="utf-8") as f:
        f.write("# Entry v2 Training Report\n\n")
        f.write(f"Generated UTC: {ts}\n\n")
        f.write("## Eval (validation split)\n\n")
        f.write(f"- ROC AUC: {roc_auc}\n")
        f.write(f"- PR AUC: {pr_auc}\n")
        f.write(f"- LogLoss: {ll}\n")
        f.write(f"- Brier: {brier}\n")
        f.write("\n## Confusion Matrix\n\n")
        f.write(f"- threshold(thr)=0.5\n")
        f.write(f"- cm=[[{cm[0,0]},{cm[0,1]}],[{cm[1,0]},{cm[1,1]}]]\n")
        f.write("\n## Feature importance\n\n")
        f.write(f"- feature_importance.csv: {fi_csv_path}\n")
        if perm_path:
            f.write(f"- permutation_importance.csv: {perm_path}\n")
        if shap_summary_path:
            f.write(f"- shap_summary: {shap_summary_path}\n")

    with open(optuna_report_md, "w", encoding="utf-8") as f:
        f.write("# Entry v2 Optuna Report\n\n")
        f.write(f"Generated UTC: {ts}\n\n")
        if best_params is not None:
            f.write("## Best Params\n\n")
            f.write(json.dumps(best_params, indent=2))
            f.write("\n\n")
        if training_log is not None:
            f.write("## Training Log\n\n")
            f.write(json.dumps(training_log, indent=2))
            f.write("\n\n")

    metadata = {
        "generated_utc": ts,
        "dataset_csv_path": dataset_csv_path,
        "booster_path": booster_path,
        "eval_split": "val",
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "logloss": ll,
        "brier": brier,
        "confusion_matrix": cm.tolist(),
        "feature_importance_csv": fi_csv_path,
        "plots": plot_paths,
        "permutation_importance_csv": perm_path,
        "shap_summary": shap_summary_path,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "ok": True,
        "outputs": {
            "training_report_md": training_report_md,
            "optuna_report_md": optuna_report_md,
            "feature_importance_csv": fi_csv_path,
            "metadata_json": metadata_path,
        },
    }


if __name__ == "__main__":
    raise SystemExit("Use generate_reports(dataset_csv_path=..., booster_path=..., output_dir=...)\n")

