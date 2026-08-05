"""Dynamic TP updater (XGBoost-driven)

Updates TP during reconciliation when model confidence (p_win) is available.
Defensive: never raises out of reconciliation.
"""

from __future__ import annotations

from typing import Optional

from utils.logger import get_logger

from config import DYNAMIC_TP_ENABLED, DYNAMIC_TP_AGGRESSION


logger = get_logger("dynamic_tp")

# In-memory dedup: order_id -> bool
_tp_updated_by_order_id = set()


def calculate_dynamic_tp(
    order_id: str,
    symbol: str,
    direction: str,
    entry_price: float,
    current_tp: Optional[float],
    p_win: Optional[float],
    *,
    apply_tp_fn,
    tp_ratio_override: Optional[float] = None,
):
    """Calculate and apply a dynamic TP.

    Args:
        order_id: MT5 ticket/order id.
        symbol: trading symbol.
        direction: 'buy' or 'sell'.
        entry_price: trade entry price.
        current_tp: existing TP price (can be None).
        p_win: XGBoost p_win (0..1).
        apply_tp_fn: callback to apply sl/tp modifications.
            Signature: apply_tp_fn(order_id: str, new_tp: float) -> bool

    Returns:
        new_tp if updated, else None.
    """
    try:
        if not DYNAMIC_TP_ENABLED:
            return None

        oid = str(order_id) if order_id is not None else ""
        if not oid or oid in _tp_updated_by_order_id:
            return None

        if current_tp is None:
            return None

        if p_win is None:
            return None

        p = float(p_win)
        if p <= 0:
            return None

        ep = float(entry_price)
        tp = float(current_tp)

        # Expand target based on p_win.
        # buy: new_tp above current_tp when p_win>0.6
        # sell: new_tp below current_tp when p_win>0.6
        factor = 1.0 + (p - 0.6) * float(DYNAMIC_TP_AGGRESSION)
        if tp_ratio_override is not None:
            # best-effort regime modifier: scale how aggressive TP expansion is
            factor = 1.0 + (factor - 1.0) * float(tp_ratio_override)

        d = str(direction).strip().lower()
        if d == "buy":
            new_tp = ep + (tp - ep) * factor
            if new_tp <= tp:
                return None
        else:
            # treat anything else as sell
            new_tp = ep - (ep - tp) * factor
            if new_tp >= tp:
                return None

        ok = bool(apply_tp_fn(oid, new_tp))
        if ok:
            _tp_updated_by_order_id.add(oid)
            logger.info(f"[DYNAMIC_TP] Updated TP for {symbol}: {new_tp}")
            return new_tp

    except Exception as e:
        logger.error(f"calculate_dynamic_tp error: {e}")

    return None

