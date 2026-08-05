from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

from utils.logger import get_logger
from data.storage.database import get_conn
from analysis.features.ml_dataset_builder import build_ml_row
from analysis.models.xgboost_trainer import train_model_from_db
from data_quality import validate_execution_row

logger = get_logger("train_pipeline")


def _get_execution_dataset_rows() -> List[Dict[str, Any]]:
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("SELECT * FROM execution_dataset")
        return [dict(r) for r in c.fetchall()]
    finally:
        conn.close()


def build_dataset_strict(min_rows: int = 50) -> Tuple[List[List[float]], List[float], Dict[str, Any]]:
    """Read execution_dataset and build a strict clean dataset.

    Rules:
      - must be accepted by validate_execution_row() (no placeholders, no None)
      - build_ml_row() must also accept (STRICT_MODE)
    """
    rows = _get_execution_dataset_rows()

    accepted: List[List[float]] = []
    y_train: List[float] = []
    rejected = 0
    rejection_reasons: List[str] = []

    for r in rows:
        ok, reasons = validate_execution_row(r)
        if not ok:
            rejected += 1
            # keep limited size to avoid huge logs
            if len(rejection_reasons) < 50:
                rejection_reasons.append(",".join(reasons))
            continue

        built = build_ml_row(r)
        if built is None:
            rejected += 1
            if len(rejection_reasons) < 50:
                rejection_reasons.append("build_ml_row_rejected")
            continue

        X, y = built
        accepted.append(X)
        y_train.append(y)

    stats = {
        "total_rows": len(rows),
        "accepted_rows": len(accepted),
        "rejected_rows": rejected,
        "rejection_examples": rejection_reasons[:10],
    }

    if len(accepted) <= 0:
        raise RuntimeError("No accepted dataset rows (DATASET CLEAN=0)")

    if len(accepted) < min_rows:
        # start training as soon as we reach min_rows (>= min_rows)
        raise RuntimeError(f"Not enough clean rows: {len(accepted)} < min_rows={min_rows}")


    return accepted, y_train, stats


def main() -> None:
    # Strict closed-loop: require enough clean real rows before training.
    min_rows = 50

    logger.info("DATASET SCAN started")

    X_clean, y_clean, stats = build_dataset_strict(min_rows=min_rows)
    logger.info("DATASET CLEAN")
    logger.info("dataset_stats=%s", stats)

    logger.info("TRAINING STARTED")
    # train_model_from_db reads from DB again, but uses STRICT_MODE.
    # We already validated strict acceptance; if this fails it indicates inconsistency.
    result = train_model_from_db(strict_mode=True, min_rows=min_rows)


    if not result.get("ok"):
        raise RuntimeError(f"Training failed: {result}")

    logger.info("MODEL SAVED")
    logger.info("train_result=%s", result)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("train_pipeline failed: %s", str(e))
        # ensure non-zero exit for automation
        sys.exit(1)

