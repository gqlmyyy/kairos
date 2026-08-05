from __future__ import annotations

"""analysis/entry_v2/run_pipeline.py

Single-run Entry v2 dataset pipeline (NO training, NO Optuna, NO model creation).

Flow:
1) dataset_builder.build_dataset(output_dir="data/entry_v2/")
   -> produces entry_v2_dataset_*.csv (and parquet + metadata)
2) Locate the most recent CSV in data/entry_v2/ matching entry_v2_dataset_*.csv
3) feature_engineering.generate_features(dataset_csv_path=latest_csv, output_dir="data/entry_v2/")
   -> produces data/entry_v2/features_dataset.parquet, features_dataset.csv, feature_schema.json
4) entry_labels.generate_entry_labels_v2()
   -> produces data/entry_v2/labeled_dataset.parquet, labeled_dataset.csv
5) dataset_audit.audit_dataset()
   -> prints report and PASS/FAIL verdict

Stops after audit.

Important:
- Does not touch Exit model code.
- Does not start training.
- If any step fails, raises with a clear error.
"""

import glob
import os
import sys
from typing import Any, Dict, Optional

try:
    from utils.logger import get_logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    def get_logger(name: str):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)


from analysis.entry_v2.dataset_builder import build_dataset
from analysis.entry_v2.feature_engineering import generate_features
from analysis.entry_v2.entry_labels import generate_entry_labels_v2
from analysis.entry_v2.dataset_audit import audit_dataset


logger = get_logger("entry_v2.run_pipeline")


def _require_dir(path: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Required directory does not exist: {path}")


def _find_latest_csv(pattern: str) -> str:
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    # latest by mtime
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def _ensure_exit_after_audit() -> None:
    # Explicit stop; avoids future accidental code additions.
    raise SystemExit(0)


def main() -> None:
    out_dir = "data/entry_v2/"
    _require_dir(os.path.dirname(os.path.abspath(out_dir)) if out_dir.endswith(os.sep) else os.path.abspath("data") )

    # Ensure directory exists
    os.makedirs(out_dir, exist_ok=True)

    logger.info("[Entry v2 run_pipeline] Step 1: build_dataset -> unified CSV/Parquet")
    builder_meta = build_dataset(output_dir=out_dir)
    logger.info("[Entry v2 run_pipeline] dataset_builder meta: row_count=%s exports=%s", builder_meta.get("row_count"), builder_meta.get("exports"))

    # Step 2: latest unified CSV
    logger.info("[Entry v2 run_pipeline] Step 2: locate latest entry_v2_dataset_*.csv")
    latest_csv = _find_latest_csv(os.path.join(out_dir, "entry_v2_dataset_*.csv"))
    logger.info("[Entry v2 run_pipeline] latest unified CSV: %s", latest_csv)

    # Step 3: feature engineering
    logger.info("[Entry v2 run_pipeline] Step 3: generate_features from latest unified CSV")
    feat_result = generate_features(
        dataset_csv_path=latest_csv,
        output_dir=out_dir,
        # feature_engineering already writes to data/entry_v2/features_dataset.* by default
    )
    logger.info("[Entry v2 run_pipeline] feature_engineering done: generated_rows=%s parquet_path=%s csv_path=%s",
                feat_result.get("generated_rows"), feat_result.get("parquet_path"), feat_result.get("csv_path"))

    # Step 4: labeling
    logger.info("[Entry v2 run_pipeline] Step 4: generate_entry_labels_v2 -> labeled_dataset.*")
    label_report: Dict[str, Any] = generate_entry_labels_v2(
        engineered_parquet_path="data/entry_v2/features_dataset.parquet",
        engineered_csv_path="data/entry_v2/features_dataset.csv",
        output_parquet_path="data/entry_v2/labeled_dataset.parquet",
        output_csv_path="data/entry_v2/labeled_dataset.csv",
    )
    logger.info("[Entry v2 run_pipeline] label_report: win_rate=%s total=%s output=%s/%s",
                label_report.get("win_rate"), label_report.get("total_samples"),
                label_report.get("output_parquet_path"), label_report.get("output_csv_path"))

    # Step 5: audit
    logger.info("[Entry v2 run_pipeline] Step 5: dataset_audit")
    audit_report = audit_dataset(
        dataset_csv_path="data/entry_v2/labeled_dataset.csv",
        output_dir="data/entry_v2/audit_reports/",
        label_column="label",
    )

    # Print PASS/FAIL + full audit report already writes files; also return structure.
    verdict = "PASS" if audit_report.get("ok") else "FAIL"
    logger.info("[Entry v2 run_pipeline] Dataset audit verdict=%s reason=%s", verdict, audit_report.get("reason"))

    # Additionally print the markdown path if present
    out_files = audit_report.get("output", {}) or {}
    md_path = out_files.get("dataset_report_md")
    if md_path:
        logger.info("[Entry v2 run_pipeline] dataset_report.md: %s", md_path)

    _ensure_exit_after_audit()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("[Entry v2 run_pipeline] FAILED: %s", str(e))
        # Non-zero exit
        sys.exit(1)

