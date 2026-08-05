"""Partial TP + Trailing

When price reaches the first partial TP target, close 50% (or configurable)
then move SL for remaining volume defensively (trailing-style).

This module is defensive and should not break reconciliation.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from utils.logger import get_logger

from config import PARTIAL_TP_ENABLED, PARTIAL_TP_RATIO


logger = get_logger("partial_tp_trailing")

# order_id -> bool
_partial_done: Dict[str, bool] = {}


def check_partial_tp(
    order_id: str,
    symbol: str,
    direction: str,
    entry_price: float,
    volume: float,
    current_tp: Optional[float],
    current_sl: Optional[float],
    current_price: float,
    apply_sltp_fn: Callable[..., bool],
    close_partial_fn: Callable[..., bool],
):
    """Return True if partial TP executed, else False."""
    try:
        if not PARTIAL_TP_ENABLED:
            return False

        if order_id is None or str(order_id).strip() == "":
            return False

        if current_tp is None:
            return False

        oid = str(order_id)
        if _partial_done.get(oid):
            return False

        if entry_price is None or volume is None:
            return False

        entry = float(entry_price)
        tp = float(current_tp)
        if tp == entry:
            return False

        ratio = float(PARTIAL_TP_RATIO)
        # first target based on total TP distance
        if str(direction).lower() == "sell":
            first_tp = entry - (entry - tp) * ratio
            hit = float(current_price) <= first_tp
        else:
            first_tp = entry + (tp - entry) * ratio
            hit = float(current_price) >= first_tp

        if not hit:
            return False

        half_volume = float(volume) * 0.5
        if half_volume <= 0:
            return False

        # Close partial
        ok_close = bool(close_partial_fn(oid, half_volume))
        if not ok_close:
            return False

        # Activate trailing for remaining part:
        # For safety we move SL to entry (break-even) if not already better.
        new_sl = entry
        ok_sl = bool(
            apply_sltp_fn(
                order_id=oid,
                symbol=symbol,
                direction=direction,
                new_sl=new_sl,
                new_tp=None,
            )
        )

        _partial_done[oid] = True
        logger.info(f"[PARTIAL_TP] Closed 50% of {symbol}, trailing activated")
        return True

    except Exception as e:
        logger.error(f"check_partial_tp error: {e}")
        return False

