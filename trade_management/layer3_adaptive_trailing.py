"""Layer 3 - Adaptive Trailing Stop.

One trailing system that absorbs what used to be separate "dynamic volatility
management" and "dynamic take profit" modules. Trend strength and volatility
feed a single distance equation:

    distance = ATR_now x base_multiplier x trend_factor x volatility_factor

  - strong trend      -> trend_factor up to WIDE   (let the move breathe)
  - choppy market     -> trend_factor down to TIGHT (protect quickly)
  - ATR rising        -> volatility_factor > 1      (widen automatically)
  - ATR falling       -> volatility_factor < 1      (tighten automatically)

The result is clamped to [MIN, MAX] ATR multiples so no combination of inputs
can produce a nonsensical stop.

This layer is also the target-extension mechanism: for profiles with
``USE_FIXED_TP = False`` it clears the fixed take profit and lets the trail
decide when the trade ends. There is no separate dynamic-TP system.

MAE/MFE statistics only *calibrate sensitivity* here — how much pullback from
peak counts as normal before the trail tightens. They never emit an exit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

from . import tm_config as C
from .types import LayerResult, TradeContext

logger = get_logger("tm.adaptive_trailing")

LAYER = "adaptive_trailing"


@dataclass(frozen=True)
class TrailingPlan:
    active: bool
    distance: float
    atr_multiplier: float
    trend_factor: float
    volatility_factor: float
    age_scale: float
    calibration_factor: float
    reason: str


def _lerp(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    """Linear map of ``value`` from [lo, hi] onto [out_lo, out_hi], clamped."""
    if hi <= lo:
        return out_lo
    t = (value - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    return out_lo + t * (out_hi - out_lo)


def compute_trend_factor(trend_strength: float, settings: Optional[dict] = None) -> float:
    settings = settings or {}
    low = float(settings.get("TRAILING_TREND_LOW", C.TRAILING_TREND_LOW))
    high = float(settings.get("TRAILING_TREND_HIGH", C.TRAILING_TREND_HIGH))
    tight = float(settings.get("TRAILING_TREND_TIGHT_FACTOR", C.TRAILING_TREND_TIGHT_FACTOR))
    wide = float(settings.get("TRAILING_TREND_WIDE_FACTOR", C.TRAILING_TREND_WIDE_FACTOR))
    return _lerp(float(trend_strength or 0.0), low, high, tight, wide)


def compute_volatility_factor(
    atr_now: float, atr_at_entry: float, settings: Optional[dict] = None
) -> float:
    """ATR expansion ratio, clamped so a data spike cannot blow the stop out."""
    settings = settings or {}
    lo = float(settings.get("TRAILING_VOL_RATIO_MIN", C.TRAILING_VOL_RATIO_MIN))
    hi = float(settings.get("TRAILING_VOL_RATIO_MAX", C.TRAILING_VOL_RATIO_MAX))

    if atr_at_entry <= 0 or atr_now <= 0:
        return 1.0
    ratio = float(atr_now) / float(atr_at_entry)
    return max(lo, min(hi, ratio))


def compute_calibration_factor(
    ctx: TradeContext,
    pullback_tolerance: Optional[float] = None,
    settings: Optional[dict] = None,
) -> float:
    """Stretch or shrink the trail based on historical MFE giveback behaviour.

    If this symbol/profile historically gives back a large share of its peak
    before continuing, the trail is loosened slightly so normal breathing does
    not stop the trade out. Bounded by MFE_CALIBRATION_MAX_ADJUST so a bad
    statistic can only nudge, never dominate.
    """
    settings = settings or {}
    tolerance = pullback_tolerance
    if tolerance is None:
        tolerance = float(settings.get("MFE_PULLBACK_TOLERANCE", C.MFE_PULLBACK_TOLERANCE))

    max_adjust = float(settings.get("MFE_CALIBRATION_MAX_ADJUST", C.MFE_CALIBRATION_MAX_ADJUST))
    baseline = float(C.MFE_PULLBACK_TOLERANCE)
    if baseline <= 0:
        return 1.0

    delta = (float(tolerance) - baseline) / baseline
    delta = max(-max_adjust, min(max_adjust, delta))
    return 1.0 + delta


def compute_trailing_plan(
    ctx: TradeContext,
    age_scale: float = 1.0,
    pullback_tolerance: Optional[float] = None,
    settings: Optional[dict] = None,
) -> TrailingPlan:
    """Compute the trailing distance without deciding anything yet."""
    settings = settings or {}

    if not bool(settings.get("ADAPTIVE_TRAILING_ENABLED", C.ADAPTIVE_TRAILING_ENABLED)):
        return TrailingPlan(False, 0.0, 0.0, 1.0, 1.0, age_scale, 1.0, "disabled")

    activate_r = float(settings.get("TRAILING_ACTIVATE_R", C.TRAILING_ACTIVATE_R))
    if ctx.profit_r < activate_r:
        return TrailingPlan(
            False, 0.0, 0.0, 1.0, 1.0, age_scale, 1.0,
            f"profit_r={ctx.profit_r:.2f}<{activate_r}",
        )

    if ctx.atr_now <= 0:
        return TrailingPlan(False, 0.0, 0.0, 1.0, 1.0, age_scale, 1.0, "atr_unavailable")

    base = float(settings.get("TRAILING_BASE_ATR_MULTIPLIER", C.TRAILING_BASE_ATR_MULTIPLIER))
    trend_factor = compute_trend_factor(ctx.trend_strength, settings)
    vol_factor = compute_volatility_factor(ctx.atr_now, ctx.atr_at_entry, settings)
    calibration = compute_calibration_factor(ctx, pullback_tolerance, settings)

    multiplier = base * trend_factor * vol_factor * float(age_scale) * calibration

    lo = float(settings.get("TRAILING_MIN_ATR_MULTIPLIER", C.TRAILING_MIN_ATR_MULTIPLIER))
    hi = float(settings.get("TRAILING_MAX_ATR_MULTIPLIER", C.TRAILING_MAX_ATR_MULTIPLIER))
    multiplier = max(lo, min(hi, multiplier))

    return TrailingPlan(
        active=True,
        distance=ctx.atr_now * multiplier,
        atr_multiplier=multiplier,
        trend_factor=trend_factor,
        volatility_factor=vol_factor,
        age_scale=float(age_scale),
        calibration_factor=calibration,
        reason="active",
    )


def evaluate(
    ctx: TradeContext,
    age_scale: float = 1.0,
    pullback_tolerance: Optional[float] = None,
    settings: Optional[dict] = None,
) -> LayerResult:
    """Layer entry point: propose a trailed SL (and clear TP for open-ended profiles)."""
    settings = settings or {}
    plan = compute_trailing_plan(ctx, age_scale, pullback_tolerance, settings)

    if not plan.active:
        return LayerResult.noop(LAYER, plan.reason)

    candidate = (
        ctx.current_price - plan.distance if ctx.is_buy else ctx.current_price + plan.distance
    )

    if not ctx.sl_is_improvement(candidate):
        result = LayerResult.noop(LAYER, "trail_would_not_improve_sl")
        result.meta = {"candidate_sl": candidate, "distance": plan.distance}
        return result

    # Open-ended profiles let the trail define the exit: drop the fixed target.
    use_fixed_tp = bool(settings.get("USE_FIXED_TP", C.USE_FIXED_TP))
    new_tp = None
    reasons = [f"trail@{plan.atr_multiplier:.2f}xATR"]
    if not use_fixed_tp and ctx.tp:
        new_tp = 0.0
        reasons.append("target_extended(fixed_tp_cleared)")

    logger.info(
        "[TM_L3_TRAIL] order=%s mult=%.2f (trend=%.2f vol=%.2f age=%.2f cal=%.2f) "
        "sl %.5f -> %.5f",
        ctx.order_id, plan.atr_multiplier, plan.trend_factor, plan.volatility_factor,
        plan.age_scale, plan.calibration_factor, ctx.sl, candidate,
    )
    return LayerResult(
        layer=LAYER,
        new_sl=candidate,
        new_tp=new_tp,
        reasons=reasons,
        meta={
            "distance": plan.distance,
            "atr_multiplier": plan.atr_multiplier,
            "trend_factor": plan.trend_factor,
            "volatility_factor": plan.volatility_factor,
            "age_scale": plan.age_scale,
            "calibration_factor": plan.calibration_factor,
        },
    )
