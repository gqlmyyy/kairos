"""ML Exit Model

Feature #15 (partial):
- Uses XGBoost p_win to decide early exit.
- If p_win is below a threshold, close the trade in MT5.

This module is defensive and should never break reconciliation.
"""

from typing import Optional

from utils.logger import get_logger

from config import ML_EXIT_THRESHOLD

logger = get_logger("ml_exit_model")


def check_ml_exit(order_id: str, symbol: str, p_win: Optional[float], close_trade_fn):
    """Check ML exit condition.

    Args:
        order_id: MT5 ticket/order position id.
        symbol: trading symbol.
        p_win: probability of success (0..1). If None, do nothing.
        close_trade_fn: function(ticket)->bool, typically _close_trade_mt5

    Side effect:
        If p_win is not None and p_win < ML_EXIT_THRESHOLD -> close_trade_fn(order_id)
    """
    try:
        if p_win is None:
            return False
        p = float(p_win)
        if p < float(ML_EXIT_THRESHOLD):
            logger.warning(f"ML Exit triggered: {symbol} order={order_id} p_win={p:.3f} < {ML_EXIT_THRESHOLD}")
            return bool(close_trade_fn(order_id))
    except Exception as e:
        logger.error(f"check_ml_exit error: {e}")
    return False

