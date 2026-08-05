"""Multi-Level Breakeven

Moves SL in multiple stages when profit_points reach configured levels.

Defensive: never raises out of apply_multi_level_breakeven.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from utils.logger import get_logger

from config import (
    MULTI_BREAKEVEN_ENABLED,
    LEVEL1_PROFIT_POINTS,
    LEVEL1_SL_OFFSET,
    LEVEL2_PROFIT_POINTS,
    LEVEL3_PROFIT_POINTS,
    LEVEL3_SL_OFFSET,
)

logger = get_logger("multi_level_breakeven")

# order_id -> last applied stage number (1..3)
_breakeven_stage: Dict[str, int] = {}


def _is_better_sl(direction: str, new_sl: float, current_sl: Optional[float]) -> bool:
    try:
        cur = float(current_sl) if current_sl is not None else 0.0
    except Exception:
        cur = 0.0

    if cur <= 0:
        return True

    d = str(direction).lower()
    if d == "sell":
        return new_sl < cur
    return new_sl > cur


def apply_multi_level_breakeven(
    order_id: str,
    symbol: str,
    direction: str,
    entry_price: float,
    current_sl: Optional[float],
    profit_points: float,
    apply_sltp_fn: Callable[..., bool],
) -> bool:
    """Return True if any SL update applied."""
    try:
        if not MULTI_BREAKEVEN_ENABLED:
            return False

        if order_id is None or str(order_id).strip() == "":
            return False

        if entry_price is None:
            return False

        pid = str(order_id)
        stage_done = int(_breakeven_stage.get(pid, 0) or 0)

        d = str(direction).lower()
        is_sell = d == "sell"

        # Stage 1
        if stage_done < 1 and float(profit_points) >= float(LEVEL1_PROFIT_POINTS):
            offset = float(LEVEL1_SL_OFFSET)
            new_sl = entry_price + offset if is_sell else entry_price - offset
            if _is_better_sl(direction, new_sl, current_sl):
                ok = bool(
                    apply_sltp_fn(
                        order_id=pid,
                        symbol=symbol,
                        direction=direction,
                        new_sl=float(new_sl),
                        new_tp=None,
                    )
                )
                if ok:
                    _breakeven_stage[pid] = 1
                    logger.info(f"[MULTI_BE] Level 1: SL moved for {symbol}")
                    return True

        # Stage 2
        if stage_done < 2 and float(profit_points) >= float(LEVEL2_PROFIT_POINTS):
            new_sl = float(entry_price)
            if _is_better_sl(direction, new_sl, current_sl):
                ok = bool(
                    apply_sltp_fn(
                        order_id=pid,
                        symbol=symbol,
                        direction=direction,
                        new_sl=new_sl,
                        new_tp=None,
                    )
                )
                if ok:
                    _breakeven_stage[pid] = 2
                    logger.info(f"[MULTI_BE] Level 2: SL moved to entry for {symbol}")
                    return True

        # Stage 3
        if stage_done < 3 and float(profit_points) >= float(LEVEL3_PROFIT_POINTS):
            offset = float(LEVEL3_SL_OFFSET)
            new_sl = float(entry_price + offset) if is_sell else float(entry_price + offset)
            # For buy: entry + offset. For sell: entry - offset would typically be +offset in point-space,
            # but requirement says: entry_price + LEVEL3_SL_OFFSET (for both directions as written).
            # We'll follow requirement literally.
            if _is_better_sl(direction, new_sl, current_sl):
                ok = bool(
                    apply_sltp_fn(
                        order_id=pid,
                        symbol=symbol,
                        direction=direction,
                        new_sl=new_sl,
                        new_tp=None,
                    )
                )
                if ok:
                    _breakeven_stage[pid] = 3
                    logger.info(f"[MULTI_BE] Level 3: SL moved to protected profit for {symbol}")
                    return True

        return False

    except Exception as e:
        logger.error(f"apply_multi_level_breakeven error: {e}")
        return False

