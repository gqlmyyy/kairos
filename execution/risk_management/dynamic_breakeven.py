"""Dynamic Breakeven

Moves SL to entry dynamically based on EMA crossover.

Defensive: never raises out of apply_dynamic_breakeven.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from utils.logger import get_logger

from config import (
    DYNAMIC_BREAKEVEN_ENABLED,
    DYNAMIC_BE_FAST_EMA,
    DYNAMIC_BE_SLOW_EMA,
    DYNAMIC_BE_FALLBACK_POINTS,
)

logger = get_logger("dynamic_breakeven")

# order_id -> bool
_dynamic_be_done: Dict[str, bool] = {}


def apply_dynamic_breakeven(
    order_id: str,
    symbol: str,
    direction: str,
    entry_price: float,
    current_sl: Optional[float],
    current_price: float,
    apply_sltp_fn: Callable[..., bool],
    # Optional EMAs injected via apply_sltp_fn call context
    ema_fast: Optional[float] = None,
    ema_slow: Optional[float] = None,
    profit_points: Optional[float] = None,
) -> bool:
    """Return True if SL updated.

    Note: reconciliation passes only required args; ema_fast/ema_slow/profit_points
    are accepted for compatibility with future improvements.
    """
    try:
        if not DYNAMIC_BREAKEVEN_ENABLED:
            return False

        if order_id is None or str(order_id).strip() == "":
            return False

        pid = str(order_id)
        if _dynamic_be_done.get(pid):
            return False

        if entry_price is None or current_price is None:
            return False

        # Determine if we should move to breakeven:
        # Primary: EMA trend confirmation (fast crosses slow in trade direction)
        direction_l = str(direction).lower()
        is_sell = direction_l == "sell"

        should_move = False
        if ema_fast is not None and ema_slow is not None:
            if is_sell:
                # For sell: expect fast < slow (bearish momentum)
                should_move = float(ema_fast) < float(ema_slow)
            else:
                # For buy: expect fast > slow (bullish momentum)
                should_move = float(ema_fast) > float(ema_slow)

        # Fallback: profit_points reached
        if not should_move and profit_points is not None:
            try:
                should_move = float(profit_points) >= float(DYNAMIC_BE_FALLBACK_POINTS)
            except Exception:
                should_move = False

        if not should_move:
            return False

        cur_sl = float(current_sl) if current_sl is not None else 0.0
        new_sl = float(entry_price)

        # Do not worsen SL
        better = (cur_sl == 0) or ((not is_sell and new_sl > cur_sl) or (is_sell and new_sl < cur_sl))
        if not better:
            _dynamic_be_done[pid] = True
            return False

        ok = bool(
            apply_sltp_fn(
                order_id=pid,
                symbol=symbol,
                direction=direction,
                new_sl=new_sl,
                new_tp=None,
            )
        )
        if not ok:
            return False

        _dynamic_be_done[pid] = True
        logger.info(f"[DYNAMIC_BE] Moved SL to entry for {symbol} (dynamic trigger)")
        return True

    except Exception as e:
        logger.error(f"apply_dynamic_breakeven error: {e}")
        return False

