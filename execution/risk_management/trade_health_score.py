"""Trade Health Score (XGBoost)

Uses p_win from XGBoost inference to compute a health score.
If health is too low, we close the trade defensively.
"""

from __future__ import annotations

from typing import Optional

from utils.logger import get_logger

from config import TRADE_HEALTH_ENABLED, TRADE_HEALTH_MIN_SCORE


logger = get_logger("trade_health_score")


def check_trade_health(order_id: str, symbol: str, p_win: Optional[float], close_trade_fn):
    """Close trade when health_score < TRADE_HEALTH_MIN_SCORE.

    Args:
        order_id: MT5 ticket/order id.
        symbol: trading symbol.
        p_win: XGBoost p_win (0..1). If None -> do nothing.
        close_trade_fn: function(ticket)->bool, typically _close_trade_mt5.

    Returns:
        True if closure triggered.
    """
    try:
        if not TRADE_HEALTH_ENABLED:
            return False

        if p_win is None:
            return False

        if order_id is None or str(order_id).strip() == "":
            return False

        p = float(p_win)
        health_score = p * 100.0

        if health_score < float(TRADE_HEALTH_MIN_SCORE):
            ok = bool(close_trade_fn(str(order_id)))
            if ok:
                logger.info(f"[HEALTH] Closed {symbol} (health={health_score:.1f})")
                return True
    except Exception as e:
        logger.error(f"check_trade_health error: {e}")

    return False

