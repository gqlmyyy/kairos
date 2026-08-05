# Trading Bot V3 - risk/drawdown.py
# Drawdown protection: 5% → stop, 10% → half risk, 20% → full stop

from utils.logger import get_logger
from config import MAX_DRAWDOWN_HALT, ACCOUNT_DRAWDOWN_HALF, ACCOUNT_DRAWDOWN_STOP, BASE_RISK_PERCENT

logger = get_logger("drawdown")

def check_drawdown(daily_pnl: float, equity: float) -> dict:
    """Check drawdown levels and return action"""
    if equity <= 0:
        return {"action": "stop", "risk_multiplier": 0, "reason": "No equity"}
    
    daily_dd = abs(daily_pnl) / equity if daily_pnl < 0 else 0
    
    # Tier 1: Daily DD > 5% → stop
    if daily_dd >= MAX_DRAWDOWN_HALT:
        logger.warning(f"Daily DD halt: {daily_dd:.2%}")
        return {"action": "halt_day", "risk_multiplier": 0, 
                "reason": f"Daily drawdown {daily_dd:.1%} > {MAX_DRAWDOWN_HALT:.0%}"}
    
    # Tier 2: Account DD > 20% → full stop
    if daily_dd >= ACCOUNT_DRAWDOWN_STOP:
        logger.warning(f"Account DD stop: {daily_dd:.2%}")
        return {"action": "full_stop", "risk_multiplier": 0,
                "reason": f"Account drawdown {daily_dd:.1%} > {ACCOUNT_DRAWDOWN_STOP:.0%}"}
    
    # Tier 3: Account DD > 10% → half risk
    if daily_dd >= ACCOUNT_DRAWDOWN_HALF:
        logger.warning(f"Half risk: DD {daily_dd:.2%}")
        return {"action": "half_risk", "risk_multiplier": 0.5,
                "reason": f"Account drawdown {daily_dd:.1%} > {ACCOUNT_DRAWDOWN_HALF:.0%}"}
    
    return {"action": "ok", "risk_multiplier": 1.0, "reason": ""}
