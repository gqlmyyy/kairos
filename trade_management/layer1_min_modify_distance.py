"""Layer 1 - Minimum modify distance.

The last filter before any SL/TP write reaches the broker. Nothing else in the
system may call the executor's modify path directly; everything funnels here.

Input : ModifyRequest + TradeContext
Output : FilterVerdict(approved, reason, request)

Three independent reasons to reject a modification:

1. The change is smaller than ``MIN_MODIFY_DISTANCE_POINTS`` — modify-spam on a
   loop that ticks every few seconds, with no protective value.
2. The new stop is on the wrong side of price, or would move protection
   backwards.
3. The new stop sits inside the broker's stop level, which the broker would
   reject anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

from . import tm_config as C
from .types import ModifyRequest, TradeContext

logger = get_logger("tm.min_modify")

LAYER = "min_modify_distance"


@dataclass(frozen=True)
class FilterVerdict:
    approved: bool
    reason: str
    request: Optional[ModifyRequest] = None

    def __bool__(self) -> bool:
        return self.approved


def _points(distance: float, point_size: float) -> float:
    if point_size <= 0:
        point_size = C.DEFAULT_POINT_SIZE
    return abs(distance) / point_size


def filter_modification(
    request: ModifyRequest,
    ctx: TradeContext,
    settings: Optional[dict] = None,
) -> FilterVerdict:
    """Approve or reject a pending SL/TP modification."""
    settings = settings or {}

    if not bool(settings.get("MIN_MODIFY_ENABLED", C.MIN_MODIFY_ENABLED)):
        return FilterVerdict(True, "filter_disabled", request)

    if request.new_sl is None and request.new_tp is None:
        return FilterVerdict(False, "nothing_to_modify")

    point_size = ctx.point_size if ctx.point_size > 0 else C.DEFAULT_POINT_SIZE
    min_points = float(settings.get("MIN_MODIFY_DISTANCE_POINTS", C.MIN_MODIFY_DISTANCE_POINTS))

    approved_sl = request.new_sl
    approved_tp = request.new_tp

    # ---- Stop loss checks ----
    if approved_sl is not None:
        if approved_sl <= 0:
            return FilterVerdict(False, "invalid_sl")

        # Never move protection backwards.
        if not ctx.sl_is_improvement(approved_sl):
            return FilterVerdict(False, "sl_not_an_improvement")

        # Wrong side of current price would close instantly.
        if ctx.is_buy and approved_sl >= ctx.current_price:
            return FilterVerdict(False, "sl_above_price_for_buy")
        if not ctx.is_buy and approved_sl <= ctx.current_price:
            return FilterVerdict(False, "sl_below_price_for_sell")

        # Broker stop-level buffer.
        buffer_points = max(
            float(ctx.broker_stop_level_points or 0.0),
            float(settings.get("MIN_BROKER_STOP_BUFFER_POINTS", C.MIN_BROKER_STOP_BUFFER_POINTS)),
        )
        gap_points = _points(ctx.current_price - approved_sl, point_size)
        if gap_points < buffer_points:
            return FilterVerdict(False, f"inside_broker_stop_level({gap_points:.1f}<{buffer_points:.1f})")

        # Minimum travel since the current stop.
        if ctx.sl and ctx.sl > 0:
            moved_points = _points(approved_sl - ctx.sl, point_size)
            if moved_points < min_points:
                return FilterVerdict(False, f"sl_move_too_small({moved_points:.1f}<{min_points:.1f})")

    # ---- Take profit checks ----
    if approved_tp is not None:
        if approved_tp <= 0:
            approved_tp = None  # clearing the TP is legitimate for open-ended profiles
        elif ctx.tp and ctx.tp > 0:
            moved_points = _points(approved_tp - ctx.tp, point_size)
            if moved_points < min_points:
                approved_tp = None  # drop the TP part, keep any SL part

    if approved_sl is None and approved_tp is None:
        return FilterVerdict(False, "all_changes_below_threshold")

    final = ModifyRequest(
        order_id=request.order_id,
        symbol=request.symbol,
        direction=request.direction,
        new_sl=approved_sl,
        new_tp=approved_tp,
        reasons=request.reasons,
    )
    logger.info(
        "[TM_L1_MINMOD] approved order=%s sl=%s tp=%s reasons=%s",
        final.order_id, final.new_sl, final.new_tp, ",".join(final.reasons),
    )
    return FilterVerdict(True, "approved", final)
