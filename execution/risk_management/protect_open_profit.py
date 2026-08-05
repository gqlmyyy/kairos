"""Protect Open Profit

Ensures SL guarantees a minimum amount of open-profit once profit points
reach a configured threshold.

Defensive: never raises out of check_protect_open_profit.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from utils.logger import get_logger

from config import (
    PROFIT_PROTECT_ENABLED,
    PROFIT_PROTECT_TRIGGER_POINTS,
    PROFIT_PROTECT_LOCK_POINTS,
)

logger = get_logger("protect_open_profit")

# order_id -> bool
_profit_protected: Dict[str, bool] = {}


def check_protect_open_profit(
    order_id: str,
    symbol: str,
    direction: str,
    current_sl: Optional[float],
    current_price: float,
    entry_price: float,
    profit_points: float,
    apply_sltp_fn: Callable[..., bool],
) -> bool:
    """Return True if SL updated (profit protected)."""
    try:
        if not PROFIT_PROTECT_ENABLED:
            return False

        if order_id is None or str(order_id).strip() == "":
            return False

        if _profit_protected.get(str(order_id)):
            return False

        if entry_price is None or profit_points is None:
            return False

        trigger = float(PROFIT_PROTECT_TRIGGER_POINTS)
        if float(profit_points) < trigger:
            return False

        lock_pts = float(PROFIT_PROTECT_LOCK_POINTS)
        if lock_pts <= 0:
            return False

        d = str(direction).lower()
        is_sell = d == "sell"

        # Requirement: new_sl = entry +/- lock_points (no pip conversion here;
        # caller provides profit_points in points but offset in same point-space).
        # This is defensive and assumes points==price units used by strategy.
        new_sl = float(entry_price - lock_pts) if is_sell else float(entry_price + lock_pts)
        cur_sl = float(current_sl) if current_sl is not None else 0.0

        # BUY: higher SL is better; SELL: lower SL is better.
        better = (cur_sl == 0) or ((not is_sell and new_sl > cur_sl) or (is_sell and new_sl < cur_sl))
        if not better:
            _profit_protected[str(order_id)] = True
            return False

        ok = bool(
            apply_sltp_fn(
                order_id=str(order_id),
                symbol=symbol,
                direction=direction,
                new_sl=new_sl,
                new_tp=None,
            )
        )
        if not ok:
            return False

        _profit_protected[str(order_id)] = True
        logger.info(
            f"[PROTECT_PROFIT] Locked {PROFIT_PROTECT_LOCK_POINTS} pts profit for {symbol}"
        )
        return True

    except Exception as e:
        logger.error(f"check_protect_open_profit error: {e}")
        return False

