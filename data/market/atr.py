# Trading Bot V3 - data/market/atr.py
# ATR calculation and management

from data.market.client import get_candles
from utils.logger import get_logger

logger = get_logger("atr")

def calculate_atr_from_candles(candles: list, period: int = 14) -> float:
    """Calculate ATR from OHLC candle data"""
    if len(candles) < period + 1:
        return 0.0
    tr_values = []
    for i in range(1, len(candles)):
        high = float(candles[i].get("high", 0))
        low = float(candles[i].get("low", 0))
        close_prev = float(candles[i-1].get("close", 0))
        tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
        tr_values.append(tr)
    if len(tr_values) < period:
        return 0.0
    atr = sum(tr_values[-period:]) / period
    return atr

def get_atr_from_qd(symbol: str, timeframe: str = "H1") -> float:
    try:
        candles = get_candles(symbol, timeframe, 20)
        if candles:
            atr = calculate_atr_from_candles(candles)
            if atr > 0:
                return atr
    except Exception as e:
        logger.error(f"ATR calculation error: {e}")
    return 0.0

