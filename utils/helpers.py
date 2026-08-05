# Trading Bot V3 - utils/helpers.py

from datetime import datetime
from typing import Optional

def now_iso() -> str:
    return datetime.now().isoformat()

def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def time_str() -> str:
    return datetime.now().strftime("%H:%M:%S")

def clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))

def pct_diff(a: float, b: float) -> float:
    """Percentage difference between two values"""
    if b == 0:
        return 0.0
    return (a - b) / b

def pip_value(symbol: str) -> float:
    from config import PIP_VALUES
    return PIP_VALUES.get(symbol, 0.0001)

def pip_distance(price1: float, price2: float, symbol: str) -> float:
    pip = pip_value(symbol)
    return abs(price1 - price2) / pip if pip > 0 else 0

def format_pnl(pnl: float) -> str:
    prefix = "+" if pnl >= 0 else ""
    return f"{prefix}{pnl:.2f}$"

def safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

