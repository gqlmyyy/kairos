"""Layer 4 - Trade Age Management.

Age changes how aggressively the trade is managed, and the time stop is one
*condition inside this layer* rather than a system of its own.

Phases (bars = closed candles since entry, boundaries in tm_config):

    0-5    settle   : no big changes, let the trade develop
    6-12   trail    : trailing ramps in, scale slides 1.0 -> 0.8
    12-15  tighten  : stop tightens harder, scale slides 0.8 -> 0.5
    >15    expired  : tightest scale

Time-stop condition: past TIME_STOP_MIN_BARS and still below
TIME_STOP_MIN_PROFIT_R, the trade is closed outright. Capital that has not
produced 0.3R in ten-plus candles is better redeployed.

Output is both a decision (possible full close) and an ``age_scale`` that
Layer 3 multiplies into its trailing distance — that is how age "ramps"
trailing without owning a second trailing implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

from . import tm_config as C
from .types import LayerResult, TradeContext

logger = get_logger("tm.trade_age")

LAYER = "trade_age"

# Tolerance for R-multiple comparisons at phase/threshold boundaries.
_EPSILON = 1e-9

PHASE_SETTLE = "settle"
PHASE_TRAIL = "trail"
PHASE_TIGHTEN = "tighten"
PHASE_EXPIRED = "expired"


@dataclass(frozen=True)
class AgeAssessment:
    phase: str
    bars_open: int
    age_scale: float
    allow_large_changes: bool
    time_stop_triggered: bool
    reason: str


def _lerp(t: float, a: float, b: float) -> float:
    t = max(0.0, min(1.0, t))
    return a + t * (b - a)


def assess_age(ctx: TradeContext, settings: Optional[dict] = None) -> AgeAssessment:
    """Classify the trade's age and derive the trailing scale."""
    settings = settings or {}

    if not bool(settings.get("TRADE_AGE_ENABLED", C.TRADE_AGE_ENABLED)):
        return AgeAssessment(PHASE_SETTLE, ctx.bars_open, 1.0, True, False, "disabled")

    bars = int(ctx.bars_open or 0)
    settle_max = int(settings.get("AGE_PHASE_SETTLE_MAX_BARS", C.AGE_PHASE_SETTLE_MAX_BARS))
    trail_max = int(settings.get("AGE_PHASE_TRAIL_MAX_BARS", C.AGE_PHASE_TRAIL_MAX_BARS))
    tighten_max = int(settings.get("AGE_PHASE_TIGHTEN_MAX_BARS", C.AGE_PHASE_TIGHTEN_MAX_BARS))

    # --- time-stop condition (internal to this layer) ---
    min_bars = int(settings.get("TIME_STOP_MIN_BARS", C.TIME_STOP_MIN_BARS))
    max_bars = int(settings.get("TIME_STOP_MAX_BARS", C.TIME_STOP_MAX_BARS))
    min_profit_r = float(settings.get("TIME_STOP_MIN_PROFIT_R", C.TIME_STOP_MIN_PROFIT_R))

    # Float tolerance: profit_r is a division of price differences, so a trade
    # sitting exactly on the floor can read 0.29999999999999716 and be closed
    # by an exact comparison. The epsilon keeps the boundary inclusive.
    time_stop = bars >= min_bars and ctx.profit_r < (min_profit_r - _EPSILON)
    # Beyond the hard ceiling the trade goes regardless of how it is doing.
    if bars > max_bars:
        time_stop = True

    # --- phase + scale ---
    if bars <= settle_max:
        phase = PHASE_SETTLE
        scale = float(settings.get("AGE_SETTLE_TRAIL_SCALE", C.AGE_SETTLE_TRAIL_SCALE))
        allow_large = False
    elif bars <= trail_max:
        phase = PHASE_TRAIL
        span = max(1, trail_max - settle_max)
        scale = _lerp(
            (bars - settle_max) / span,
            float(settings.get("AGE_TRAIL_SCALE_START", C.AGE_TRAIL_SCALE_START)),
            float(settings.get("AGE_TRAIL_SCALE_END", C.AGE_TRAIL_SCALE_END)),
        )
        allow_large = True
    elif bars <= tighten_max:
        phase = PHASE_TIGHTEN
        span = max(1, tighten_max - trail_max)
        scale = _lerp(
            (bars - trail_max) / span,
            float(settings.get("AGE_TIGHTEN_SCALE_START", C.AGE_TIGHTEN_SCALE_START)),
            float(settings.get("AGE_TIGHTEN_SCALE_END", C.AGE_TIGHTEN_SCALE_END)),
        )
        allow_large = True
    else:
        phase = PHASE_EXPIRED
        scale = float(settings.get("AGE_TIGHTEN_SCALE_END", C.AGE_TIGHTEN_SCALE_END))
        allow_large = True

    reason = f"phase={phase} bars={bars} scale={scale:.2f}"
    return AgeAssessment(phase, bars, scale, allow_large, time_stop, reason)


def evaluate(ctx: TradeContext, settings: Optional[dict] = None) -> LayerResult:
    """Layer entry point: close on the time-stop condition, else report the scale."""
    assessment = assess_age(ctx, settings)

    meta = {
        "phase": assessment.phase,
        "bars_open": assessment.bars_open,
        "age_scale": assessment.age_scale,
        "allow_large_changes": assessment.allow_large_changes,
        "profit_r": ctx.profit_r,
    }

    if assessment.time_stop_triggered:
        logger.warning(
            "[TM_L4_AGE] order=%s time stop: bars=%d profit_r=%.2f -> close",
            ctx.order_id, assessment.bars_open, ctx.profit_r,
        )
        return LayerResult(
            layer=LAYER,
            close_full=True,
            reasons=[f"time_stop(bars={assessment.bars_open},profit_r={ctx.profit_r:.2f})"],
            meta=meta,
        )

    result = LayerResult.noop(LAYER, assessment.reason)
    result.meta = meta
    return result
