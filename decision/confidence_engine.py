# Trading Bot V3 - decision/confidence_engine.py
# Overall confidence calculation from multiple components

from utils.logger import get_logger

logger = get_logger("confidence_engine")

def calculate_confidence(
    ai_confidence: float,
    mtf_aligned: bool,
    trend_direction: str,
    ai_bias: str,
    regime: str
) -> float:
    """Calculate overall confidence 0-1 based on all factors"""
    
    base_confidence = ai_confidence
    
    # Multi-timeframe alignment bonus/penalty
    if mtf_aligned:
        base_confidence *= 1.2
    else:
        base_confidence *= 0.7
    
    # Trend-AI agreement bonus
    if trend_direction != "neutral" and trend_direction == ai_bias:
        base_confidence *= 1.15
    elif trend_direction != "neutral" and trend_direction != ai_bias:
        base_confidence *= 0.6  # Big penalty for disagreement
    
    # Market regime adjustment
    regime_factors = {
        "TRENDING": 1.1,
        "RANGING": 0.75,
        "HIGH_VOLATILITY": 0.5,
        "LOW_VOLATILITY": 1.0,
        "UNKNOWN": 0.7
    }
    base_confidence *= regime_factors.get(regime, 0.7)
    
    return round(min(1.0, max(0.0, base_confidence)), 3)
