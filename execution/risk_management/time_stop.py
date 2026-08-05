"""Time Stop

Closes a trade if it stays open longer than max_duration_minutes.

Defensive: never raises out of check_time_stop.
"""

from __future__ import annotations

import time
from typing import Optional, Callable

from utils.logger import get_logger

from config import TIME_STOP_ENABLED, TIME_STOP_MAX_MINUTES

logger = get_logger("time_stop")


def check_time_stop(
    order_id: str,
    symbol: str,
    open_time: Optional[float],
    max_duration_minutes: Optional[float],
    close_trade_fn: Callable[..., bool],
) -> bool:
    """Return True if time stop executed else False.

    Args:
        order_id: MT5 ticket / order id
        symbol: symbol name
        open_time: epoch seconds when trade opened (preferred)
        max_duration_minutes: override; if None uses TIME_STOP_MAX_MINUTES
        close_trade_fn: callable like _close_trade_mt5(ticket)
    """
    try:
        if not TIME_STOP_ENABLED:
            return False

        if order_id is None or str(order_id).strip() == "":
            return False

        if not symbol:
            return False

        if open_time is None:
            return False

        max_min = float(max_duration_minutes) if max_duration_minutes is not None else float(TIME_STOP_MAX_MINUTES)
        if max_min <= 0:
            return False

        elapsed_sec = float(time.time()) - float(open_time)
        elapsed_min = elapsed_sec / 60.0

        if elapsed_min < max_min:
            return False

        ok = bool(close_trade_fn(order_id))
        if not ok:
            return False

        logger.info(f"[TIME_STOP] Closed {symbol} after {elapsed_min:.2f} minutes")
        return True

    except Exception as e:
        logger.error(f"check_time_stop error: {e}")
        return False

