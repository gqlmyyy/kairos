from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

from utils.logger import get_logger
from analysis.models.xgboost_trainer import train_model_from_db, should_retrain
from analysis.models.model_manager import load_latest_model

# DB access is via execution_dataset
from data.storage.database import get_conn

logger = get_logger("system_orchestrator")


def _get_execution_dataset_stats() -> Dict[str, Any]:
    """Return lightweight stats for execution_dataset.

    Used to decide whether we have enough new data to retrain.
    """
    conn = get_conn()
    try:
        c = conn.cursor()
        row = c.execute(
            """SELECT 
                COUNT(*) as total_rows,
                MAX(dataset_updated_at) as last_updated_at
            FROM execution_dataset
            WHERE order_id IS NOT NULL"""
        ).fetchone()
        total = int(row[0] or 0)
        last_updated_at = row[1]

        # new_rows_count is approximated using updated time; we keep it simple.
        # If last_updated_at is null => 0.
        # NOTE: For a precise delta, we'd need an additional cursor/table.
        # Here we approximate by using total as proxy.
        # This is enough to trigger retrain frequently in early stages.
        new_rows_count = total

        return {
            "total_rows": total,
            "new_rows_count": new_rows_count,
            "last_updated_at": last_updated_at,
        }
    finally:
        conn.close()


def _parse_datetime_or_none(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        # dataset_updated_at stored as isoformat
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def run_daily_cycle() -> None:
    """Daily orchestrator cycle.

    Steps:
      1) read execution_dataset stats
      2) (optional) performance snapshot placeholder
      3) if should_retrain(): train + overwrite models/entry/entry_model.json + reload model
      4) reset daily metrics (not implemented; depends on DB schema)
      5) log cycle summary
    """

    stats = _get_execution_dataset_stats()
    total_rows = stats.get("total_rows", 0)
    new_rows_count = stats.get("new_rows_count", 0)
    last_updated_at = stats.get("last_updated_at")

    last_ts = _parse_datetime_or_none(last_updated_at)

    logger.info(
        "Daily cycle started | total_rows=%s new_rows_count=%s last_updated_at=%s last_ts=%s",
        str(total_rows),
        str(new_rows_count),
        str(last_updated_at),
        str(last_ts),
    )

    # Performance snapshot: optional hook point.
    # We currently rely on performance_monitor global buffer elsewhere.
    perf_snapshot: Dict[str, Any] = {}

    try:
        if should_retrain(
            new_rows_count=int(new_rows_count),
            last_train_ts=last_ts,
        ):
            logger.info("Retrain triggered by should_retrain()")
            result = train_model_from_db(strict_mode=True)

            # Ensure overwrite pointer happens inside trainer.
            # Reload model with hot reload.
            _model, _ver = load_latest_model(force_reload=True)

            logger.info(
                "Retrain completed | ok=%s reason=%s version=%s",
                str(result.get("ok")),
                str(result.get("reason")) if "reason" in result else "",
                str(_ver),
            )
        else:
            logger.info("Retrain not needed today")
    except Exception as e:
        logger.error("Daily cycle retrain error: %s", e)

    logger.info(
        "Daily cycle completed | total_rows=%s new_rows_count=%s perf_snapshot_keys=%s",
        str(total_rows),
        str(new_rows_count),
        str(list(perf_snapshot.keys())),
    )


def daily_thread_runner(hour: int = 0, minute: int = 5, interval_sec: int = 60) -> None:
    """Run run_daily_cycle() once per day at HH:MM local time.

    Uses a lightweight polling loop.
    """
    while True:
        now = datetime.now()
        if now.hour == hour and now.minute == minute:
            try:
                run_daily_cycle()
            except Exception as e:
                logger.error("Daily cycle failed: %s", e)
            # avoid running multiple times within the same minute
            time.sleep(61)
            continue
        time.sleep(interval_sec)


def start_daily_orchestrator_thread(hour: int = 0, minute: int = 5) -> threading.Thread:
    t = threading.Thread(
        target=daily_thread_runner,
        args=(hour, minute),
        daemon=True,
        name="daily_orchestrator_thread",
    )
    t.start()
    logger.info("Daily orchestrator thread started (at %02d:%02d)", hour, minute)
    return t

