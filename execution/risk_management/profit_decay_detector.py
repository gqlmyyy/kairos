"""Profit Decay Detector

Feature #9 (partial):
- Closes a trade if its current profit decays below a fraction of its peak profit.

We store peak profit in-memory keyed by order_id.
This is deterministic within a single process lifetime.
"""
from config import (
    PROFIT_DECAY_ENABLED,
    PROFIT_DECAY_TRIGGER,
    PROFIT_DECAY_PWIN_THRESHOLD,
    PROFIT_DECAY_MIN_PEAK,

)

from typing import Dict, Optional

from utils.logger import get_logger


from config import (
    PROFIT_DECAY_ENABLED,
    PROFIT_DECAY_TRIGGER,
    PROFIT_DECAY_PWIN_THRESHOLD,
)



logger = get_logger("profit_decay_detector")

# In-memory peak profit tracking per order_id
_peak_profit_by_order: Dict[str, float] = {}


def _update_peak(order_id: str, current_profit: float) -> float:
    prev = _peak_profit_by_order.get(order_id)
    if prev is None or current_profit > prev:
        _peak_profit_by_order[order_id] = float(current_profit)
        return float(current_profit)
    return float(prev)


def check_profit_decay(
    order_id: str,
    current_profit: float,
    close_trade_fn,
    p_win: Optional[float] = None,
):

    """Check profit decay condition.

    Condition:
        if current_profit < peak_profit * PROFIT_DECAY_TRIGGER -> close.

    Args:
        order_id: MT5 ticket/order position id.
        current_profit: current unrealized/online profit.
        close_trade_fn: function(ticket)->bool, typically _close_trade_mt5

    Returns:
        True if closure triggered.
    """
    try:
        if not PROFIT_DECAY_ENABLED:
            return False

        if order_id is None or str(order_id).strip() == "":
            return False

        oid = str(order_id)
        cur = float(current_profit)
        peak = _update_peak(oid, cur)

        # if peak is non-positive, decay logic is not meaningful; do nothing
        # لا نفعل Profit Decay إذا لم تحقق الصفقة ربحاً كافياً
        if peak <= 0:
         return False

        if peak < float(PROFIT_DECAY_MIN_PEAK):
            return False

        # Primary: XGBoost p_win based early decay
        if p_win is not None:
            try:
                p = float(p_win)
                if p < float(PROFIT_DECAY_PWIN_THRESHOLD):
                    logger.warning(
                        f"Profit decay triggered by p_win: order={oid} p_win={p:.3f} "
                        f"< {float(PROFIT_DECAY_PWIN_THRESHOLD)}"
                    )
                    ok = bool(close_trade_fn(oid))
                    if ok:
                        _peak_profit_by_order.pop(oid, None)
                    return ok
            except Exception:
                pass

        # Fallback: purely profit-vs-peak rule
        if cur < peak * float(PROFIT_DECAY_TRIGGER):

            logger.warning(
                f"Profit decay triggered: order={oid} current={cur:.2f} peak={peak:.2f} "
                f"threshold={float(PROFIT_DECAY_TRIGGER)}"
            )
            ok = bool(close_trade_fn(oid))
            if ok:
                # cleanup peak after closing attempt
                _peak_profit_by_order.pop(oid, None)
            return ok

    except Exception as e:
        logger.error(f"check_profit_decay error: {e}")

    return False

