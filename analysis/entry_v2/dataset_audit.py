from __future__ import annotations

"""analysis/entry_v2/dataset_audit.py

Entry v2 dataset audit.

GOAL
- Run FAST/DEEP audits for the generated feature dataset.
- Always write the required deliverables:
  - feature_statistics.csv
  - dataset_report.md
  - constant_feature_report.csv
  - constant_feature_report.md
  - dataset_validation_report.md

IMPORTANT
- No heavy correlation / pairwise feature loops in FAST.
- DEEP (ENTRY_V2_DEEP_AUDIT=1) may compute correlation matrix.

This module is intentionally self-contained and does not modify Exit code.
"""

import csv
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore

from utils.logger import get_logger

logger = get_logger("entry_v2.dataset_audit")


@dataclass(frozen=True)
class AuditConfig:
    near_constant_std_eps: float = 1e-9
    near_zero_corr_threshold: float = 0.999  # leakage heuristic
    class_imbalance_max_ratio: float = 0.9


def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if _is_nan(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _compute_mean_median(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    s = sum(values)
    mean = s / len(values)
    # median
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return mean, float(vals[mid])
    return mean, float((vals[mid - 1] + vals[mid]) / 2.0)


def _compute_variance_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    return var, std


def _compute_correlation(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for a, b in zip(xs, ys):
        dx = a - mean_x
        dy = b - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    den = math.sqrt(den_x * den_y)
    if den == 0:
        return None
    return num / den


def _is_near_constant_from_std(std: float, eps: float) -> bool:
    # "near constant" definition uses variance/std epsilon threshold.
    return std <= eps


def audit_dataset(
    *,
    dataset_csv_path: str,
    output_dir: str,
    label_column: str = "label",
    feature_columns: Optional[List[str]] = None,
    config: AuditConfig = AuditConfig(),
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.perf_counter()
    df = pd.read_csv(dataset_csv_path)
    t1 = time.perf_counter()

    logger.info(
        "[entry_v2.dataset_audit] Load dataset: %.3fs rows=%s cols=%s",
        t1 - t0,
        getattr(df, "shape", (None, None))[0],
        getattr(df, "shape", (None, None))[1],
    )

    # Always audit ONLY the engineered numeric model features defined by the schema.
    # Metadata/identifier columns (symbol, t, label_reason, timestamps, etc.) must be ignored.
    from analysis.entry_v2.feature_schema import FEATURE_COLUMNS as _SCHEMA_FEATURES
    feature_columns = list(_SCHEMA_FEATURES)


    # Label distribution
    has_label = label_column in df.columns
    label_values = df[label_column].tolist() if has_label else []
    win_count = 0
    loss_count = 0
    for v in label_values:
        fv = _safe_float(v)
        if fv is None:
            continue
        if fv >= 0.5:
            win_count += 1
        else:
            loss_count += 1

    majority_ratio = (max(win_count, loss_count) / max(win_count + loss_count, 1)) if has_label else None

    # Duplicates
    duplicate_rows_count = int(df.duplicated().sum()) if df is not None else 0
    duplicate_symbol_t_count = 0
    if "symbol" in df.columns and "t" in df.columns:
        duplicate_symbol_t_count = int(df.duplicated(subset=["symbol", "t"]).sum())

    # Missing values (only feature_columns + label)
    missing_cells_total = 0
    missing_cells_per_col: Dict[str, int] = {}
    for c in (list(feature_columns) + ([label_column] if has_label else [])):
        if c not in df.columns:
            continue
        miss = int(df[c].isna().sum())
        missing_cells_per_col[c] = miss
        missing_cells_total += miss

    deep_audit = os.environ.get("ENTRY_V2_DEEP_AUDIT", "0") == "1"

    # Feature stats + constant/near-constant reports
    stats_rows: List[Dict[str, Any]] = []
    constant_features: List[str] = []
    near_constant_features: List[str] = []
    constant_report_rows: List[Dict[str, Any]] = []

    # Prepare label arrays for leakage heuristic correlations (FAST allowed; O(K*N) per feature is acceptable)
    aligned_y: List[float] = []
    if has_label:
        aligned_y = [_safe_float(v) for v in label_values]

    # Per-feature computation (FAST/DEEP share; only correlation matrix is DEEP-only)
    stage_t = time.perf_counter()

    for col in feature_columns:
        # collect non-null values
        col_series = df[col] if col in df.columns else []
        col_vals_raw = [v for v in col_series.tolist()]
        col_vals = [fv for fv in (_safe_float(v) for v in col_vals_raw) if fv is not None]

        missing_pct = (int(df[col].isna().sum()) / max(len(df), 1)) * 100.0 if col in df.columns else 100.0
        zero_count = sum(1 for v in col_vals if v == 0.0)
        unique_count = len(set(col_vals))
        zero_pct = (zero_count / max(len(col_vals), 1)) * 100.0 if col_vals else 100.0

        if not col_vals:
            var = 0.0
            std = 0.0
            mean = 0.0
            median = 0.0
            mn = 0.0
            mx = 0.0
        else:
            mn = float(min(col_vals))
            mx = float(max(col_vals))
            mean, median = _compute_mean_median(col_vals)
            var, std = _compute_variance_std(col_vals)

        constant = var == 0.0
        near_constant = (not constant) and _is_near_constant_from_std(std, config.near_constant_std_eps)

        if constant:
            constant_features.append(col)
            cls = "Exact Constant"
        elif near_constant:
            near_constant_features.append(col)
            cls = "Near Constant"
        else:
            cls = ""

        # record constant_report_rows if constant OR near-constant
        if constant or near_constant:
            constant_report_rows.append(
                {
                    "feature_name": col,
                    "variance": var,
                    "std": std,
                    "unique_count": unique_count,
                    "zero_percentage": zero_pct,
                    "mean": mean,
                    "median": median,
                    "min": mn,
                    "max": mx,
                    "classification": cls,
                    "source_module": "analysis/entry_v2/feature_engineering.py",
                    "generation_function": "Feature column generation",
                    "reason_why_constant": "Detected by audit: variance/std ~ 0.0. Trace to feature_engineering source and upstream indicator/merge logic.",
                    "recommended_fix": "Trace feature generation in feature_engineering.py; verify indicator series length/alignment, lag/delta/interaction calculations, and conditional updates for this feature.",
                }
            )

        # Leakage heuristic: correlation with label (fast, per-feature)
        corr_with_label: Any = ""
        if has_label and len(col_vals) >= 3:
            # compute corr using aligned non-null pairs
            xs: List[float] = []
            ys: List[float] = []
            for xv, yv in zip(col_series.tolist(), label_values):
                xf = _safe_float(xv)
                yf = _safe_float(yv)
                if xf is None or yf is None:
                    continue
                xs.append(float(xf))
                ys.append(float(yf))
            corr = _compute_correlation(xs, ys)
            if corr is not None:
                corr_with_label = float(corr)

        stats_rows.append(
            {
                "feature": col,
                "missing_pct": missing_pct,
                "variance": var,
                "std": std,
                "zero_pct": zero_pct,
                "unique_count": unique_count,
                "corr_with_label": corr_with_label,
                "constant": constant,
                "near_constant": near_constant,
            }
        )

    stats_t = time.perf_counter() - stage_t

    # correlation matrix: DEEP-only
    corr_matrix_path = os.path.join(output_dir, "correlation_matrix.csv")
    if deep_audit:
        feats_for_corr = sorted(stats_rows, key=lambda r: float(r.get("variance") or 0.0), reverse=True)[:30]
        feat_names = [r["feature"] for r in feats_for_corr]
        with open(corr_matrix_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["feature"] + feat_names)
            for c1 in feat_names:
                row = [c1]
                s1 = df[c1].tolist()
                v1 = [_safe_float(v) for v in s1]
                for c2 in feat_names:
                    s2 = df[c2].tolist()
                    v2 = [_safe_float(v) for v in s2]
                    xs: List[float] = []
                    ys: List[float] = []
                    for a, b in zip(v1, v2):
                        if a is None or b is None:
                            continue
                        xs.append(float(a))
                        ys.append(float(b))
                    corr = _compute_correlation(xs, ys)
                    row.append(corr if corr is not None else "")
                w.writerow(row)
    else:
        with open(corr_matrix_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["feature"])  # FAST placeholder

    # feature_statistics.csv
    feat_stats_path = os.path.join(output_dir, "feature_statistics.csv")
    with open(feat_stats_path, "w", newline="", encoding="utf-8") as f:
        if not stats_rows:
            f.write("feature\n")
        else:
            fieldnames = list(stats_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in stats_rows:
                writer.writerow(r)

    # Leakage detection summary (FAST uses corr_with_label; DEEP may also include more checks)
    leakage_detected = False
    leakage_reasons: List[str] = []
    if has_label:
        for r in stats_rows:
            c = r.get("corr_with_label")
            if c == "" or c is None:
                continue
            try:
                cv = float(c)
            except Exception:
                continue
            if abs(cv) >= config.near_zero_corr_threshold:
                leakage_detected = True
                leakage_reasons.append(f"near-perfect corr with label: feature={r['feature']} corr={cv}")

    imbalance_exceeds = bool(has_label and majority_ratio is not None and majority_ratio >= config.class_imbalance_max_ratio)

    # dataset_report.md
    dataset_report_path = os.path.join(output_dir, "dataset_report.md")
    ts = datetime.now(timezone.utc).isoformat()
    with open(dataset_report_path, "w", encoding="utf-8") as f:
        f.write("# Entry v2 Dataset Audit\n\n")
        f.write(f"Generated: {ts}\n\n")
        f.write(f"Dataset: {dataset_csv_path}\n\n")
        f.write(f"Feature count analyzed: {len(feature_columns)}\n\n")

        f.write("## Class balance\n\n")
        if has_label:
            f.write(f"- count: {win_count + loss_count}\n")
            f.write(f"- win_count: {win_count}\n")
            f.write(f"- loss_count: {loss_count}\n")
            f.write(f"- majority_ratio: {majority_ratio}\n")
            f.write(f"- imbalance_exceeds_threshold: {imbalance_exceeds}\n\n")
        else:
            f.write("- label column not found; class balance skipped\n\n")

        f.write("## Constant/near-constant features\n\n")
        f.write(f"- constant_features_count: {len(constant_features)}\n")
        f.write(f"- near_constant_features_count: {len(near_constant_features)}\n\n")
        if constant_features:
            f.write("### Constant features (exact zero variance)\n")
            for c in constant_features[:50]:
                f.write(f"- {c}\n")
            if len(constant_features) > 50:
                f.write(f"... and {len(constant_features)-50} more\n")
            f.write("\n")

        f.write("## Leakage detection\n\n")
        f.write(f"- leakage_detected: {leakage_detected}\n")
        if leakage_detected:
            for r in leakage_reasons[:50]:
                f.write(f"- {r}\n")
            if len(leakage_reasons) > 50:
                f.write(f"... and {len(leakage_reasons)-50} more\n")
        else:
            f.write("- reason: none triggered by heuristics\n")
        f.write("\n")

        f.write("## Output files\n\n")
        f.write(f"- feature_statistics.csv: {feat_stats_path}\n")
        f.write(f"- correlation_matrix.csv: {corr_matrix_path}\n")
        f.write(f"- dataset_report.md: {dataset_report_path}\n")

    # constant_feature_report.csv
    constant_csv_path = os.path.join(output_dir, "constant_feature_report.csv")
    constant_md_path = os.path.join(output_dir, "constant_feature_report.md")
    if constant_report_rows:
        fieldnames = list(constant_report_rows[0].keys())
        with open(constant_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in constant_report_rows:
                w.writerow(r)
    else:
        with open(constant_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["feature_name", "classification"])

    # constant_feature_report.md
    with open(constant_md_path, "w", encoding="utf-8") as f:
        f.write("# Constant / Near-Constant Feature Report (Entry v2)\n\n")
        f.write(f"Total constant features: {len(constant_features)}\n\n")
        f.write(f"Total near-constant features: {len(near_constant_features)}\n\n")
        if not constant_report_rows:
            f.write("No constant features detected.\n")
        else:
            f.write("## Details\n\n")
            for r in constant_report_rows:
                f.write(f"### {r['feature_name']} ({r['classification']})\n\n")
                f.write(f"- variance: {r['variance']}\n")
                f.write(f"- std: {r['std']}\n")
                f.write(f"- unique_count: {r['unique_count']}\n")
                f.write(f"- zero_percentage: {r['zero_percentage']}\n")
                f.write(f"- mean: {r['mean']}\n")
                f.write(f"- median: {r['median']}\n")
                f.write(f"- min: {r['min']}\n")
                f.write(f"- max: {r['max']}\n\n")
                f.write(f"- source module: {r['source_module']}\n")
                f.write(f"- generation function: {r['generation_function']}\n")
                f.write(f"- reason why constant: {r['reason_why_constant']}\n")
                f.write(f"- recommended fix: {r['recommended_fix']}\n\n")

    # dataset_validation_report.md
    validation_md_path = os.path.join(output_dir, "dataset_validation_report.md")

    # Feature schema validation (ensure FEATURE_COLUMNS match actual feature columns)
    from analysis.entry_v2.feature_schema import FEATURE_COLUMNS  # local import

    schema_missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    schema_extra = [c for c in df.columns if c in FEATURE_COLUMNS and c not in feature_columns]

    final_reject = (len(constant_features) > 0) or leakage_detected or imbalance_exceeds
    final_verdict = "PASS" if not final_reject else "FAIL"

    with open(validation_md_path, "w", encoding="utf-8") as f:
        f.write("# Entry v2 Dataset Validation Report\n\n")
        f.write(f"Verdict: {final_verdict}\n\n")

        f.write("## Missing values\n\n")
        f.write(f"Missing cells total: {missing_cells_total}\n")
        # print top missing columns
        if missing_cells_per_col:
            items = sorted(missing_cells_per_col.items(), key=lambda kv: kv[1], reverse=True)
            f.write("Top missing columns:\n")
            for c, m in items[:30]:
                f.write(f"- {c}: {m}\n")
        f.write("\n")

        f.write("## Duplicates\n\n")
        f.write(f"Duplicate rows (full row): {duplicate_rows_count}\n")
        f.write(f"Duplicate (symbol,t): {duplicate_symbol_t_count}\n\n")

        f.write("## Constant / Near-constant features\n\n")
        f.write(f"Constant features: {len(constant_features)}\n")
        f.write(f"Near constant features: {len(near_constant_features)}\n\n")
        if constant_features:
            f.write("Constant features (first 50):\n")
            for c in constant_features[:50]:
                f.write(f"- {c}\n")
            if len(constant_features) > 50:
                f.write(f"... and {len(constant_features)-50} more\n")
            f.write("\n")

        if near_constant_features:
            f.write("Near-constant features (first 50):\n")
            for c in near_constant_features[:50]:
                f.write(f"- {c}\n")
            if len(near_constant_features) > 50:
                f.write(f"... and {len(near_constant_features)-50} more\n")
            f.write("\n")

        f.write("## Leakage detection\n\n")
        f.write(f"leakage_detected: {leakage_detected}\n")
        if leakage_detected:
            for r in leakage_reasons[:50]:
                f.write(f"- {r}\n")
        f.write("\n")

        f.write("## Label distribution\n\n")
        if has_label:
            f.write(f"wins: {win_count}\n")
            f.write(f"losses: {loss_count}\n")
            f.write(f"win_rate: {win_count/max(win_count+loss_count,1)}\n")
            f.write(f"majority_ratio: {majority_ratio}\n")
        else:
            f.write("No label column present.\n")
        f.write("\n")

        f.write("## Feature schema validation\n\n")
        f.write(f"Missing schema columns (in dataset): {len(schema_missing)}\n")
        for c in schema_missing[:50]:
            f.write(f"- {c}\n")
        if len(schema_missing) > 50:
            f.write(f"... and {len(schema_missing)-50} more\n")
        f.write("\n")

        f.write("## Checks summary\n\n")
        f.write(f"constant_features_exist: {len(constant_features) > 0}\n")
        f.write(f"near_constant_features_exist: {len(near_constant_features) > 0}\n")
        f.write(f"leakage_detected: {leakage_detected}\n")
        f.write(f"class_imbalance_exceeds: {imbalance_exceeds}\n")

    # Final return summary
    verdict_ok = final_verdict == "PASS"
    return {
        "ok": verdict_ok,
        "reject": not verdict_ok,
        "verdict": final_verdict,
        "counts": {
            "rows": int(getattr(df, "shape", (0, 0))[0]),
            "cols": int(getattr(df, "shape", (0, 0))[1]),
            "constant_features": len(constant_features),
            "near_constant_features": len(near_constant_features),
            "duplicate_rows": duplicate_rows_count,
            "duplicate_symbol_t": duplicate_symbol_t_count,
            "missing_cells_total": missing_cells_total,
        },
        "paths": {
            "feature_statistics_csv": feat_stats_path,
            "dataset_report_md": dataset_report_path,
            "constant_feature_report_csv": constant_csv_path,
            "constant_feature_report_md": constant_md_path,
            "dataset_validation_report_md": validation_md_path,
            "correlation_matrix_csv": corr_matrix_path,
        },
    }


if __name__ == "__main__":
    raise SystemExit("Use audit_dataset(dataset_csv_path=..., output_dir=...)")

