"""Layer 5 - Partial Take Profit, with MAE/MFE as a calibrator.

Ladder (all values from tm_config):

    +1R -> stop to break-even   (owned by Layer 1, nothing closed here)
    +2R -> close 30% of the ORIGINAL volume
    +3R -> close a further 30%

Fractions are of the original volume so the ladder does not drift as the
position shrinks. Whatever remains after the ladder is managed exclusively by
Layer 3's adaptive trailing — this layer never trails and never fully closes.

MAE/MFE statistics are used here only to derive a *pullback tolerance* that is
handed to Layer 3 to tune its sensitivity on the remainder. This layer does not
turn excursion statistics into an exit decision of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from utils.logger import get_logger

from . import tm_config as C
from .types import LayerResult, TradeContext

logger = get_logger("tm.partial_tp")

LAYER = "partial_tp"


@dataclass(frozen=True)
class PartialLevel:
    index: int
    trigger_r: float
    fraction: float
    volume: float


def _ladder(settings: Optional[dict] = None) -> Sequence[Tuple[float, float]]:
    settings = settings or {}
    return tuple(settings.get("PARTIAL_TP_LADDER", C.PARTIAL_TP_LADDER))


def next_due_level(
    ctx: TradeContext, settings: Optional[dict] = None
) -> Optional[PartialLevel]:
    """The highest ladder level reached but not yet taken, if any."""
    settings = settings or {}
    done = set(ctx.partial_levels_done or ())
    profit_r = ctx.profit_r

    due: Optional[PartialLevel] = None
    for index, (trigger_r, fraction) in enumerate(_ladder(settings)):
        if index in done:
            continue
        if profit_r < float(trigger_r):
            continue
        volume = float(ctx.initial_volume) * float(fraction)
        due = PartialLevel(index, float(trigger_r), float(fraction), volume)

    return due


def compute_pullback_tolerance(
    mfe_samples: Optional[Iterable[float]] = None,
    settings: Optional[dict] = None,
) -> float:
    """Normal giveback from peak, as a fraction of MFE.

    Fed to Layer 3 so trailing sensitivity reflects how this strategy actually
    behaves. With too few samples the configured default stands — a statistic
    from three trades is noise, not calibration.
    """
    settings = settings or {}
    default = float(settings.get("MFE_PULLBACK_TOLERANCE", C.MFE_PULLBACK_TOLERANCE))
    min_samples = int(settings.get("MFE_CALIBRATION_MIN_SAMPLES", C.MFE_CALIBRATION_MIN_SAMPLES))

    samples: List[float] = []
    for value in mfe_samples or ():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= v <= 1.0:
            samples.append(v)

    if len(samples) < min_samples:
        return default

    samples.sort()
    # Median is the robust choice: a couple of runaway trades should not drag
    # the tolerance for every subsequent position.
    mid = len(samples) // 2
    if len(samples) % 2:
        return samples[mid]
    return (samples[mid - 1] + samples[mid]) / 2.0


def evaluate(ctx: TradeContext, settings: Optional[dict] = None) -> LayerResult:
    """Layer entry point: propose at most one partial close per pass."""
    settings = settings or {}

    if not bool(settings.get("PARTIAL_TP_ENABLED", C.PARTIAL_TP_ENABLED)):
        return LayerResult.noop(LAYER, "disabled")

    if ctx.r_distance <= 0:
        return LayerResult.noop(LAYER, "unknown_risk")

    level = next_due_level(ctx, settings)
    if level is None:
        return LayerResult.noop(LAYER, f"no_level_due(profit_r={ctx.profit_r:.2f})")

    min_partial = float(settings.get("MIN_PARTIAL_VOLUME", C.MIN_PARTIAL_VOLUME))
    min_remaining = float(settings.get("MIN_REMAINING_VOLUME", C.MIN_REMAINING_VOLUME))

    if level.volume < min_partial:
        return LayerResult.noop(LAYER, f"partial_below_min_lot({level.volume:.4f}<{min_partial})")

    remaining = float(ctx.volume) - level.volume
    if remaining < min_remaining:
        # Closing this slice would leave an untradeable stub. Skip the partial
        # and let the trail manage the whole remainder instead.
        return LayerResult.noop(
            LAYER, f"remainder_below_min_lot({remaining:.4f}<{min_remaining})"
        )

    logger.info(
        "[TM_L5_PARTIAL] order=%s level=%d @%.1fR closing %.2f of %.2f (remaining %.2f)",
        ctx.order_id, level.index, level.trigger_r, level.volume, ctx.volume, remaining,
    )
    return LayerResult(
        layer=LAYER,
        close_fraction=level.fraction,
        reasons=[f"partial_tp_L{level.index}@{level.trigger_r:.1f}R"],
        meta={
            "level_index": level.index,
            "trigger_r": level.trigger_r,
            "fraction": level.fraction,
            "close_volume": level.volume,
            "remaining_volume": remaining,
        },
    )
