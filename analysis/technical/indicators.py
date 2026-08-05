# Trading Bot V3 - analysis/technical/indicators.py
# Technical indicator analysis from QuantDinger

from utils.logger import get_logger
from config import TF_TREND, TF_DECISION
from data.market.market_snapshot import MarketSnapshot

logger = get_logger("indicators")


def _get(snapshot: MarketSnapshot, symbol: str, timeframe: str) -> dict:
    data = snapshot.get(symbol, timeframe)
    if not isinstance(data, dict):
        return {}
    return data


def get_trend_score(*args):
    """Return trend score.

    Supported call signatures:
      - get_trend_score(snapshot, symbol)
      - get_trend_score(symbol)  (backward-compat for telegram_bot.py)

    Note: the telegram handler currently calls this without a snapshot.
    In that case we return a neutral default.
    """
    if len(args) == 2:
        snapshot, symbol = args
        return get_trend_score_from_snapshot(snapshot, symbol)
    if len(args) == 1:
        _symbol = args[0]
        return 40, "neutral"
    raise TypeError("get_trend_score expects (snapshot, symbol) or (symbol)")



def get_trend_score_from_snapshot(snapshot: MarketSnapshot, symbol: str) -> tuple:
    """Returns (score 0-100, direction: bullish/bearish/neutral) using snapshot H4."""
    data = _get(snapshot, symbol, TF_TREND)
    if not data or (not data.get("ma_trend") and not data.get("rsi")):
        logger.debug(f"No trend data for {symbol}, using neutral")
        return 40, "neutral"

    ma_trend = str(data.get("ma_trend", "")).lower()
    rsi = float(data.get("rsi", 50))

    if "strong uptrend" in ma_trend:
        return 85, "bullish"
    elif "uptrend" in ma_trend:
        return 70, "bullish"
    elif "strong downtrend" in ma_trend:
        return 85, "bearish"
    elif "downtrend" in ma_trend:
        return 70, "bearish"

    # RSI fallback
    if rsi > 65:
        return 75, "bullish"
    elif rsi > 55:
        return 65, "bullish"
    elif rsi < 35:
        return 75, "bearish"
    elif rsi < 45:
        return 65, "bearish"
    else:
        return 40, "neutral"


def get_momentum_score(*args, timeframe: str = None):
    """Return momentum score.

    Supported call signatures:
      - get_momentum_score(snapshot, symbol, timeframe?)
      - get_momentum_score(symbol) (backward-compat for telegram_bot.py)
    """
    if len(args) >= 2:
        snapshot, symbol = args[0], args[1]
        timeframe = args[2] if len(args) >= 3 else TF_DECISION
        return get_momentum_score_from_snapshot(snapshot=snapshot, symbol=symbol, timeframe=timeframe)
    if len(args) == 1:
        return 40, "neutral"
    raise TypeError("get_momentum_score expects (snapshot, symbol, ...) or (symbol)")


def get_momentum_score_from_snapshot(
    snapshot: MarketSnapshot,
    symbol: str,
    timeframe: str = TF_DECISION,
) -> tuple:
    """Returns (score 0-100, direction) using RSI thresholds for the provided timeframe."""
    data = _get(snapshot, symbol, timeframe)

    if not data:
        logger.debug(f"No momentum data for {symbol}, using neutral")
        return 40, "neutral"

    rsi = float(data.get("rsi", 50))
    if rsi < 30:
        return 85, "bullish"
    elif rsi > 70:
        return 85, "bearish"
    elif rsi < 45:
        return 65, "bearish"
    elif rsi > 55:
        return 65, "bullish"
    else:
        return 40, "neutral"


def get_volatility_score(*args):
    """Return volatility score (0-100).

    Supported call signatures:
      - get_volatility_score(snapshot, symbol)
      - get_volatility_score(symbol) (backward-compat for telegram_bot.py)
    """
    if len(args) == 2:
        snapshot, symbol = args
        return get_volatility_score_from_snapshot(snapshot, symbol)
    if len(args) == 1:
        return 50
    raise TypeError("get_volatility_score expects (snapshot, symbol) or (symbol)")


def get_volatility_score_from_snapshot(snapshot: MarketSnapshot, symbol: str) -> float:
    """Returns 0-100 (higher = safer/less volatile)"""
    data = _get(snapshot, symbol, TF_DECISION)
    if not data:
        logger.debug(f"No volatility data for {symbol}, using neutral")
        return 50

    vol = str(data.get("volatility", "")).lower()
    if "very high" in vol:
        return 20
    elif "high" in vol:
        return 35
    elif "low" in vol:
        return 80
    else:
        return 55


