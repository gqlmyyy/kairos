"""Watchdog for the bot's own MT5 session.

The previous implementation watched the wrong thing. It polled
``{QUANTDINGER_URL}/api/mt5/status``, i.e. whether *QuantDinger* was connected
to MT5 — not whether the bot's own session was alive. A healthy QuantDinger
with a dead bot session read as "connected", and a reconnect attempt drove
QuantDinger's session rather than the bot's.

This version checks the session the bot actually trades through, via
``mt5_session.is_healthy()`` (a local ``account_info()`` read, no broker round
trip), and recovers it through the same module that owns it — so recovery takes
the shared lock instead of racing the other threads.
"""

from __future__ import annotations

import threading
import time

from config import WATCHDOG_FAIL_LIMIT, WATCHDOG_INTERVAL
from data.market.mt5_session import ensure_session, is_available, is_healthy
from utils.logger import get_logger

logger = get_logger("mt5_watchdog")

# Grace period before the first check, so startup has finished.
_STARTUP_GRACE_SEC = 60


def check_mt5_connection() -> bool:
    """True when the bot's MT5 session is usable right now."""
    return is_healthy()


def reconnect_mt5(max_retries: int = 3, delay: float = 2.0) -> bool:
    """Re-establish the session, forcing a fresh login."""
    for attempt in range(1, max_retries + 1):
        try:
            if ensure_session(force_relogin=True):
                logger.info("MT5 session re-established (attempt %d/%d)", attempt, max_retries)
                return True
            logger.warning("MT5 reconnect attempt %d/%d failed", attempt, max_retries)
        except Exception as exc:
            logger.warning("MT5 reconnect attempt %d/%d raised: %s", attempt, max_retries, exc)

        if attempt < max_retries:
            time.sleep(delay)

    logger.error("MT5 reconnect failed after %d attempts", max_retries)
    return False


def watchdog_loop() -> None:
    from telegram.notifier import notify_alert, notify_status

    if not is_available():
        logger.error("MetaTrader5 library unavailable - watchdog will not run")
        return

    consecutive_failures = 0
    time.sleep(_STARTUP_GRACE_SEC)

    while True:
        try:
            if check_mt5_connection():
                if consecutive_failures > 0:
                    logger.info("MT5 session restored")
                    notify_status("✅ MT5 session restored")
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                logger.warning(
                    "MT5 session unhealthy! Attempt %d/%d",
                    consecutive_failures, WATCHDOG_FAIL_LIMIT,
                )

                if consecutive_failures == 1:
                    notify_alert("⚠️ MT5 session lost - attempting to reconnect...")

                if reconnect_mt5():
                    consecutive_failures = 0
                    logger.info("MT5 session recovered")
                    notify_status("✅ MT5 session recovered")
                elif consecutive_failures >= WATCHDOG_FAIL_LIMIT:
                    notify_alert(
                        f"❌ MT5 session could not be recovered after "
                        f"{WATCHDOG_FAIL_LIMIT} attempts.\nCheck the terminal manually."
                    )
        except Exception as exc:
            logger.error("Watchdog error: %s", exc)

        time.sleep(WATCHDOG_INTERVAL)


def start_mt5_watchdog() -> threading.Thread:
    thread = threading.Thread(target=watchdog_loop, daemon=True, name="mt5_watchdog")
    thread.start()
    logger.info("MT5 watchdog started (interval=%ss)", WATCHDOG_INTERVAL)
    return thread
