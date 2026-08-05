"""Layer 2a - Signal Flip Exit (hard override).

The only unconditional exit in the system. When the entry engine now produces a
*fully confirmed* signal in the opposite direction to an open trade, that trade
closes immediately and no further layer is consulted.

Input : TradeContext + current signal snapshot
Output : LayerResult(terminal=True) on flip, otherwise a no-op

"Fully confirmed" deliberately requires all of: opposite direction, score above
threshold, AI confidence above threshold, and (by default) multi-timeframe
alignment. A merely weakening signal is Layer 2b's business, not a hard close.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from utils.logger import get_logger

from . import tm_config as C
from .types import LayerResult, TradeContext

logger = get_logger("tm.signal_flip")

LAYER = "signal_flip"

_OPPOSITE = {"buy": "sell", "sell": "buy"}


def _normalise_direction(raw: Any) -> str:
    d = str(raw or "").strip().lower()
    if d in {"buy", "long", "0"}:
        return "buy"
    if d in {"sell", "short", "1"}:
        return "sell"
    return ""


def check_signal_flip(
    ctx: TradeContext,
    signal: Optional[Dict[str, Any]] = None,
    settings: Optional[dict] = None,
) -> LayerResult:
    """Close the trade when a confirmed opposite signal exists.

    ``signal`` is the latest decision snapshot for this symbol:
        {"direction": "SELL", "final_score": 62.0,
         "ai_confidence": 0.71, "mtf_aligned": True}

    A missing or stale snapshot is treated as "no flip" — absence of evidence
    must never trigger a close.
    """
    settings = settings or {}

    if not bool(settings.get("SIGNAL_FLIP_ENABLED", C.SIGNAL_FLIP_ENABLED)):
        return LayerResult.noop(LAYER, "disabled")

    if not signal:
        return LayerResult.noop(LAYER, "no_signal_snapshot")

    position_dir = "buy" if ctx.is_buy else "sell"
    signal_dir = _normalise_direction(signal.get("direction"))

    if not signal_dir:
        return LayerResult.noop(LAYER, "signal_direction_unknown")

    if signal_dir != _OPPOSITE[position_dir]:
        return LayerResult.noop(LAYER, "signal_not_opposite")

    min_score = float(settings.get("SIGNAL_FLIP_MIN_SCORE", C.SIGNAL_FLIP_MIN_SCORE))
    min_conf = float(settings.get("SIGNAL_FLIP_MIN_CONFIDENCE", C.SIGNAL_FLIP_MIN_CONFIDENCE))
    require_mtf = bool(
        settings.get("SIGNAL_FLIP_REQUIRE_MTF_ALIGNED", C.SIGNAL_FLIP_REQUIRE_MTF_ALIGNED)
    )

    try:
        score = float(signal.get("final_score") or 0.0)
        confidence = float(signal.get("ai_confidence") or 0.0)
    except (TypeError, ValueError):
        return LayerResult.noop(LAYER, "signal_fields_unparseable")

    if score < min_score:
        return LayerResult.noop(LAYER, f"flip_score_too_low({score:.1f}<{min_score})")

    if confidence < min_conf:
        return LayerResult.noop(LAYER, f"flip_confidence_too_low({confidence:.2f}<{min_conf})")

    if require_mtf and not bool(signal.get("mtf_aligned")):
        return LayerResult.noop(LAYER, "flip_not_mtf_aligned")

    logger.warning(
        "[TM_L2_FLIP] order=%s %s position vs confirmed %s signal "
        "(score=%.1f conf=%.2f) -> immediate close",
        ctx.order_id, position_dir, signal_dir, score, confidence,
    )
    return LayerResult(
        layer=LAYER,
        close_full=True,
        terminal=True,
        reasons=[f"signal_flip:{position_dir}->{signal_dir}"],
        meta={"score": score, "confidence": confidence, "signal_direction": signal_dir},
    )
