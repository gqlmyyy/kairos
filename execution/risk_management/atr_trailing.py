"""ATR Trailing

Moves SL based on ATR distance from current price.

Defensive: never raises out of apply_atr_trailing.
"""

from __future__ import annotations

from typing import Callable, Optional

from utils.logger import get_logger

from config import ATR_TRAILING_ENABLED, ATR_TRAILING_MULTIPLIER

logger = get_logger("atr_trailing")


def apply_atr_trailing(
    order_id: str,
    symbol: str,
    direction: str,
    current_sl: Optional[float],
    current_price: float,
    atr: Optional[float],
    apply_sltp_fn: Callable[..., bool],
    atr_multiplier_override: Optional[float] = None,
) -> bool:
    """Apply ATR-based trailing stop.

    Returns True if SL updated.
    """
    try:
        if not ATR_TRAILING_ENABLED:
            return False

        if order_id is None or str(order_id).strip() == "":
            return False

        if atr is None:
            return False
        atr_f = float(atr)
        if atr_f <= 0:
            return False

        if current_price is None:
            return False
        price = float(current_price)
        if price <= 0:
            return False

        if direction is None:
            return False
        d = str(direction).lower()
        is_sell = d == "sell"

        cur_sl = float(current_sl) if current_sl is not None else 0.0

        trail_mult = float(atr_multiplier_override) if atr_multiplier_override is not None else float(ATR_TRAILING_MULTIPLIER)
        trail_distance = trail_mult * atr_f
        if trail_distance <= 0:
            return False

        # BUY: SL below price, SELL: SL above price
        new_sl = price + trail_distance if is_sell else price - trail_distance

        if cur_sl <= 0:
            # allow initial SL set
            ok = bool(
                apply_sltp_fn(
                    order_id=str(order_id),
                    symbol=symbol,
                    direction=direction,
                    new_sl=new_sl,
                    new_tp=None,
                )
            )
            if ok:
                logger.info(f"[ATR_TRAIL] Updated SL for {symbol} to {new_sl}")
            return ok

        better = (new_sl < cur_sl) if is_sell else (new_sl > cur_sl)
        if not better:
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

        if ok:
            logger.info(f"[ATR_TRAIL] Updated SL for {symbol} to {new_sl}")
        return ok

    except Exception as e:
        logger.error(f"apply_atr_trailing error: {e}")
        return False

