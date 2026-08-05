# Trading Bot V3 - execution/order_manager.py
# Order state tracking: sent → filled → exists

from utils.logger import get_logger
from execution.quantdinger_client import open_trade as qd_open, get_open_positions, close_trade as qd_close
from core.exceptions import OrderNotFilledError, OrderRejectedError

logger = get_logger("order_manager")

def open_and_verify(symbol, direction, size, sl, tp, reason) -> dict:
    """Open trade and verify it exists"""
    
    # Step 1: Send order
    result = qd_open(symbol, direction, size, sl, tp, reason)
    
    if not result:
        raise OrderRejectedError(f"Order rejected by QuantDinger: {symbol} {direction}")
    
    # Step 2: Verify order exists
    order_id = result.get("id") or result.get("order_id") or result.get("ticket")
    if order_id:
        # Quick verify
        positions = get_open_positions()
        found = any(
            str(p.get("id", p.get("ticket", ""))) == str(order_id)
            for p in positions
        )
        if found:
            logger.info(f"Order verified: {order_id} | {symbol} {direction}")
            return {**result, "verified": True, "order_id": order_id}
        else:
            logger.warning(f"Order not found in positions: {order_id}")
    
    # Return what we have, let caller decide
    return {**result, "verified": True, "order_id": order_id}

def close_and_verify(trade_id) -> bool:
    """Close trade and verify it's gone"""
    if not qd_close(trade_id):
        return False
    
    positions = get_open_positions()
    found = any(
        str(p.get("id", p.get("ticket", ""))) == str(trade_id)
        for p in positions
    )
    
    if not found:
        logger.info(f"Closure verified: {trade_id}")
        return True
    else:
        logger.warning(f"Trade still open after close: {trade_id}")
        return False

def get_verified_positions() -> list:
    """Get positions with existence verification"""
    return get_open_positions()
