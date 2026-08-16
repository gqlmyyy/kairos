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

AUTO_RETRAIN_ENV_VAR = "KAIROS_ENABLE_AUTO_RETRAIN"


def auto_retrain_enabled() -> bool:
    """Whether the daily cycle may train at all. Off unless explicitly set."""
    import os

    return os.environ.get(AUTO_RETRAIN_ENV_VAR, "").strip() in {"1", "true", "TRUE", "yes"}


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

        # A delta, not the total. Passing the total made should_retrain() return
        # True on every call once the table held 50 rows, which is how a
        # "retrain when there is new data" policy became "retrain always".
        #
        # Without a watermark table there is no way to compute a true delta, so
        # report 0 — no new data can be demonstrated. That is the fail-closed
        # answer: retraining replaces a model, and "I don't know" must not
        # authorise that.
        new_rows_count = 0

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

    # Automatic retraining is DISABLED and opt-in.
    #
    # This block used to call train_model_from_db(strict_mode=True), which wrote
    # a 12-feature model straight over models/entry/entry_model.json and then
    # hot-reloaded it — no schema check, no backup, no promotion gate, and no
    # human. The 12-feature schema does not match the 10 features the live path
    # sends, so a single firing would have swapped the entry model for one the
    # gate must reject, on a target (`actual_pnl > 0`) that is not the target
    # the entry model is meant to predict.
    #
    # It never fired only because start_daily_orchestrator_thread is imported
    # and never called. That is luck, not design. Retraining now requires
    # KAIROS_ENABLE_AUTO_RETRAIN=1, and even then it may not touch production:
    # promotion goes through analysis.models.production_model_guard.install(),
    # which needs its own separate opt-in.
    if not auto_retrain_enabled():
        logger.info(
            "Automatic retraining disabled (set %s=1 to enable). "
            "total_rows=%s — no model was trained or replaced.",
            AUTO_RETRAIN_ENV_VAR, str(total_rows),
        )
        return

    try:
        if should_retrain(
            new_rows_count=int(new_rows_count),
            last_train_ts=last_ts,
        ):
            logger.info("Retrain triggered by should_retrain()")
            result = train_model_from_db(strict_mode=True)

            # Deliberately NOT reloading production here. The trainer writes a
            # research artifact; replacing what is served is a separate act.
            logger.info(
                "Retrain completed into a research artifact | ok=%s reason=%s. "
                "Production model unchanged — promote via production_model_guard.install().",
                str(result.get("ok")),
                str(result.get("reason")) if "reason" in result else "",
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

