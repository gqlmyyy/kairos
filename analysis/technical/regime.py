# Trading Bot V3 - analysis/technical/regime.py
# Market regime detection

from utils.logger import get_logger

from data.market.market_snapshot import MarketSnapshot
from config import TF_DECISION
from analysis.technical.indicators import (
    get_volatility_score_from_snapshot,
    get_trend_score_from_snapshot,
)

logger = get_logger("regime")


def get_market_regime_from_snapshot(snapshot: MarketSnapshot, symbol: str) -> str:
    """Snapshot-only regime: HIGH_VOLATILITY / LOW_VOLATILITY / TRENDING / RANGING"""

    _, trend_dir = get_trend_score_from_snapshot(snapshot, symbol)
    vol_score = get_volatility_score_from_snapshot(snapshot, symbol)

    if vol_score < 30:
        return "HIGH_VOLATILITY"
    elif trend_dir != "neutral":
        return "TRENDING"
    elif vol_score > 70:
        return "LOW_VOLATILITY"
    else:
        return "RANGING"


def get_market_regime(
    snapshot: MarketSnapshot | None = None,
    symbol: str | None = None
) -> str:
    """
    Compatibility wrapper for older imports.

    Expected values:
    - "TRENDING"
    - "RANGING"
    - "LOW_VOLATILITY"
    - "HIGH_VOLATILITY"

    If called without complete data, returns a neutral default ("RANGING").
    """
    try:
        if snapshot is None or symbol is None:
            return "RANGING"
        return get_market_regime_from_snapshot(snapshot, symbol)
    except Exception:
        return "RANGING"


def is_safe_to_trade_from_snapshot(snapshot: MarketSnapshot, symbol: str) -> bool:
    regime = get_market_regime_from_snapshot(snapshot, symbol)
    return regime not in ["HIGH_VOLATILITY", "UNKNOWN"]




