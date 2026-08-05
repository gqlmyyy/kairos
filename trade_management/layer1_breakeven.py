"""Layer 1 - Break-even.

Once open profit reaches a configurable R multiple, the stop moves to entry
(plus a small cushion so the exit is genuinely flat after costs rather than a
few points negative).

Input : TradeContext, resolved settings
Output : LayerResult with new_sl, or a no-op

Runs at most once per trade: after ``breakeven_done`` is set the layer stands
down and leaves the stop to Layer 3.
"""

from __future__ import annotations

from typing import Optional

from utils.logger import get_logger

from . import tm_config as C
from .types import LayerResult, TradeContext

logger = get_logger("tm.breakeven")

LAYER = "breakeven"


def apply_breakeven(ctx: TradeContext, settings: Optional[dict] = None) -> LayerResult:
    settings = settings or {}

    if not bool(settings.get("BREAKEVEN_ENABLED", C.BREAKEVEN_ENABLED)):
        return LayerResult.noop(LAYER, "disabled")

    if ctx.breakeven_done:
        return LayerResult.noop(LAYER, "already_done")

    if ctx.r_distance <= 0:
        return LayerResult.noop(LAYER, "unknown_risk")

    trigger_r = float(settings.get("BREAKEVEN_TRIGGER_R", C.BREAKEVEN_TRIGGER_R))
    profit_r = ctx.profit_r
    if profit_r < trigger_r:
        return LayerResult.noop(LAYER, f"profit_r={profit_r:.2f}<{trigger_r}")

    offset = float(settings.get("BREAKEVEN_OFFSET_ATR", C.BREAKEVEN_OFFSET_ATR)) * max(ctx.atr_now, 0.0)
    target_sl = ctx.entry_price + offset if ctx.is_buy else ctx.entry_price - offset

    if not ctx.sl_is_improvement(target_sl):
        return LayerResult.noop(LAYER, "sl_already_beyond_breakeven")

    logger.info(
        "[TM_L1_BE] order=%s profit_r=%.2f>=%.2f sl %.5f -> %.5f",
        ctx.order_id, profit_r, trigger_r, ctx.sl, target_sl,
    )
    return LayerResult(
        layer=LAYER,
        new_sl=target_sl,
        reasons=[f"breakeven@{profit_r:.2f}R"],
        meta={"profit_r": profit_r, "trigger_r": trigger_r, "offset": offset},
    )
