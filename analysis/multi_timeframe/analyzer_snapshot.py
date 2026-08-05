from __future__ import annotations

from utils.logger import get_logger

from config import TF_TREND, TF_DECISION, TF_TIMING
from core.models import MultiTimeframeData
from data.market.market_snapshot import MarketSnapshot
from analysis.technical.indicators import (
    get_trend_score_from_snapshot,
    get_momentum_score_from_snapshot,
)

logger = get_logger("mtf")


def get_multi_timeframe_analysis_from_snapshot(
    snapshot: MarketSnapshot,
    symbol: str,
) -> MultiTimeframeData:
    """Snapshot-only MTF analysis (H4 + H1 + M15).

    No network calls here.
    """

    # H4 - Trend direction
    h4_score, h4_dir = get_trend_score_from_snapshot(snapshot, symbol)

    # H1 - Momentum / decision
    h1_score, h1_dir = get_momentum_score_from_snapshot(snapshot, symbol, timeframe=TF_DECISION)

    # M15 - Timing (short-term momentum)
    m15_score, m15_dir = get_momentum_score_from_snapshot(snapshot, symbol, timeframe=TF_TIMING)

    directions = [h4_dir, h1_dir, m15_dir]
    non_neutral = [d for d in directions if d != "neutral"]

    if len(non_neutral) == 3 and all(d == non_neutral[0] for d in non_neutral):
        aligned = True
        strength = "strong"
    elif len(non_neutral) >= 2 and all(d == non_neutral[0] for d in non_neutral):
        aligned = True
        strength = "moderate"
    elif h4_dir != "neutral" and h1_dir == h4_dir:
        aligned = True
        strength = "moderate"
    elif h1_dir != "neutral" and m15_dir == h1_dir:
        aligned = True
        strength = "weak"
    else:
        aligned = False
        strength = "weak"

    mtf = MultiTimeframeData(
        h4_direction=h4_dir,
        h4_score=h4_score,
        h1_direction=h1_dir,
        h1_score=h1_score,
        m15_direction=m15_dir,
        m15_score=m15_score,
        aligned=aligned,
        strength=strength,
    )

    logger.debug(
        "MTF (snapshot) %s: H4=%s H1=%s M15=%s aligned=%s (%s)",
        symbol,
        h4_dir,
        h1_dir,
        m15_dir,
        aligned,
        strength,
    )

    return mtf

