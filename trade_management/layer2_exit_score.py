"""Layer 2b - Unified Exit Score (soft exit).

One weighted score, one threshold, one decision. There is no separate
"probability exit" system: the ML probability is a *component* here and nothing
more.

Components (weights in tm_config):
  - probability        : the exit model's opinion, qualified per its own rules
  - trend_reversal     : the entry's directional premise breaking down
  - momentum_weakness  : momentum draining out of the move
  - volume_weakness    : currently weight 0 (see tm_config for why and how to
                         re-enable it)

Each component is normalised to 0..1 where 1 means "strongest reason to exit".
The score is the weighted mean over the components that actually have data, so
a missing component redistributes rather than silently scoring zero.

Input : component readings + optional ProbabilityAssessment
Output : LayerResult (close_full when score > threshold)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from utils.logger import get_logger

from . import tm_config as C
from .layer2_exit_probability import ProbabilityAssessment
from .types import LayerResult, TradeContext

logger = get_logger("tm.exit_score")

LAYER = "exit_score"


@dataclass(frozen=True)
class ExitScoreBreakdown:
    score: float
    threshold: float
    components: Dict[str, float]
    weights: Dict[str, float]
    should_close: bool


def _clamp01(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, v))


def compute_trend_reversal(ctx: TradeContext, readings: Dict[str, Any]) -> Optional[float]:
    """0 = trend still supports the trade, 1 = fully reversed against it.

    Uses the directional trend score (0..100, where >50 favours long) relative
    to the trade's direction.
    """
    raw = readings.get("trend_score")
    if raw is None:
        return None
    try:
        trend_score = float(raw)
    except (TypeError, ValueError):
        return None

    # Convert to "how much the trend opposes this position", 0..1.
    favourable = trend_score if ctx.is_buy else (100.0 - trend_score)
    return _clamp01((50.0 - favourable) / 50.0)


def compute_momentum_weakness(ctx: TradeContext, readings: Dict[str, Any]) -> Optional[float]:
    """0 = momentum intact, 1 = momentum fully drained.

    Prefers an explicit 0..100 momentum score; falls back to MFE giveback, which
    is a decent proxy for a stalling move.
    """
    raw = readings.get("momentum_score")
    if raw is not None:
        try:
            momentum = float(raw)
        except (TypeError, ValueError):
            momentum = None
        if momentum is not None:
            favourable = momentum if ctx.is_buy else (100.0 - momentum)
            return _clamp01((50.0 - favourable) / 50.0)

    # Fallback: how much of the peak has been handed back.
    if ctx.mfe_r > 0:
        giveback = (ctx.mfe_r - ctx.profit_r) / ctx.mfe_r
        return _clamp01(giveback)
    return None


def compute_exit_score(
    ctx: TradeContext,
    readings: Optional[Dict[str, Any]] = None,
    probability: Optional[ProbabilityAssessment] = None,
    settings: Optional[dict] = None,
) -> ExitScoreBreakdown:
    """Blend the components into a single 0..1 exit score."""
    readings = readings or {}
    settings = settings or {}

    threshold = float(settings.get("EXIT_SCORE_THRESHOLD", C.EXIT_SCORE_THRESHOLD))

    components: Dict[str, float] = {}
    weights: Dict[str, float] = {}

    # --- probability ---
    w_prob = float(settings.get("EXIT_WEIGHT_PROBABILITY", C.EXIT_WEIGHT_PROBABILITY))
    if probability and probability.available and probability.influences_decision:
        value = _clamp01(probability.probability)
        if value is not None:
            components["probability"] = value
            # A single unqualified reading is damped rather than dropped.
            weights["probability"] = w_prob * probability.weight_multiplier

    # --- trend reversal ---
    trend_reversal = compute_trend_reversal(ctx, readings)
    if trend_reversal is not None:
        components["trend_reversal"] = trend_reversal
        weights["trend_reversal"] = float(
            settings.get("EXIT_WEIGHT_TREND_REVERSAL", C.EXIT_WEIGHT_TREND_REVERSAL)
        )

    # --- momentum weakness ---
    momentum_weakness = compute_momentum_weakness(ctx, readings)
    if momentum_weakness is not None:
        components["momentum_weakness"] = momentum_weakness
        weights["momentum_weakness"] = float(
            settings.get("EXIT_WEIGHT_MOMENTUM_WEAKNESS", C.EXIT_WEIGHT_MOMENTUM_WEAKNESS)
        )

    # --- volume weakness (weight 0 by default; see tm_config) ---
    w_volume = float(settings.get("EXIT_WEIGHT_VOLUME_WEAKNESS", C.EXIT_WEIGHT_VOLUME_WEAKNESS))
    if w_volume > 0:
        volume_weakness = _clamp01(readings.get("volume_weakness"))
        if volume_weakness is not None:
            components["volume_weakness"] = volume_weakness
            weights["volume_weakness"] = w_volume

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return ExitScoreBreakdown(0.0, threshold, components, weights, False)

    # Weighted mean over available components: a missing component redistributes
    # its share instead of dragging the score to zero.
    score = sum(components[k] * weights[k] for k in components) / total_weight
    score = max(0.0, min(1.0, score))

    return ExitScoreBreakdown(
        score=score,
        threshold=threshold,
        components=components,
        weights=weights,
        should_close=score > threshold,
    )


def evaluate(
    ctx: TradeContext,
    readings: Optional[Dict[str, Any]] = None,
    probability: Optional[ProbabilityAssessment] = None,
    settings: Optional[dict] = None,
) -> LayerResult:
    """Layer entry point: compute the score and turn it into a decision."""
    breakdown = compute_exit_score(ctx, readings, probability, settings)

    meta = {
        "score": breakdown.score,
        "threshold": breakdown.threshold,
        "components": breakdown.components,
        "weights": breakdown.weights,
    }

    if not breakdown.should_close:
        result = LayerResult.noop(
            LAYER, f"score={breakdown.score:.3f}<={breakdown.threshold:.2f}"
        )
        result.meta = meta
        return result

    logger.warning(
        "[TM_L2_SCORE] order=%s exit_score=%.3f>%.2f components=%s -> close",
        ctx.order_id, breakdown.score, breakdown.threshold, breakdown.components,
    )
    return LayerResult(
        layer=LAYER,
        close_full=True,
        reasons=[f"exit_score={breakdown.score:.3f}"],
        meta=meta,
    )
