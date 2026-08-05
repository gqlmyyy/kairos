# Trading Bot V3 - risk/position_sizing.py
# Equity-based position sizing with ATR-aware lot sizing

from utils.logger import get_logger
from config import BASE_RISK_PERCENT, MAX_OPEN_TRADES, STOP_AFTER_LOSSES, PIP_VALUES
from data.storage.database import get_daily_stats, get_total_open_trades

logger = get_logger("position_sizing")

# الحد الأقصى للوت لكل زوج (سقف أمان)
MAX_LOT_PER_SYMBOL = {
    "XAUUSD": 0.10,
    "EURUSD": 0.50,
    "GBPUSD": 0.50,
    "USDJPY": 0.50,
}

# الحد الأدنى للوت
MIN_LOT = 0.01

# Pip value per 1.0 lot (in account currency) for each symbol.
# For most forex pairs: 1 lot = 100,000 units, pip value = $10 per pip.
# For XAUUSD: 1 lot = 100 oz, pip = 0.1, pip value = 100 * 0.1 = $10 per pip.
# For USDJPY: 1 lot = 100,000 units, pip = 0.01, pip value = $10 per pip (approx).
PIP_VALUE_PER_LOT = {
    "EURUSD": 10.0,
    "GBPUSD": 10.0,
    "USDJPY": 10.0,
    "XAUUSD": 10.0,
    "USDCAD": 10.0,
    "AUDUSD": 10.0,
    "NZDUSD": 10.0,
    "USDCHF": 10.0,
}


def get_pip_value(symbol: str) -> float:
    return PIP_VALUES.get(symbol, 0.0001)


def get_pip_value_per_lot(symbol: str) -> float:
    """Return pip value per 1.0 lot in account currency."""
    return PIP_VALUE_PER_LOT.get(symbol, 10.0)


def calculate_position_size(
    equity: float,
    sl_distance: float,
    symbol: str,
    consecutive_losses: int = 0,
    score: float = 65.0,
) -> float:
    """
    Calculate position size based on:
    - Equity (account balance)
    - Score (signal strength)
    - SL distance (in price units)
    - Consecutive losses

    Correct formula (MT5):
        lots = risk_amount / (sl_pips * pip_value_per_lot)

    Where:
        sl_pips = sl_distance / pip_value
        risk_amount = equity * risk_percent
    """
    if sl_distance <= 0:
        logger.warning("SL distance is 0, using minimum")
        sl_distance = 0.001

    # ==============================
    # 1. Risk % بناءً على قوة الإشارة
    # ==============================
    if score >= 90:
        risk_percent = BASE_RISK_PERCENT * 2.0    # إشارة قوية جداً → 1.0%
    elif score >= 80:
        risk_percent = BASE_RISK_PERCENT * 1.5    # إشارة قوية → 0.75%
    elif score >= 70:
        risk_percent = BASE_RISK_PERCENT * 1.0    # إشارة عادية → 0.5%
    else:
        risk_percent = BASE_RISK_PERCENT * 0.5    # إشارة ضعيفة → 0.25%

    # ==============================
    # 2. تقليل الخطر بعد خسائر متتالية
    # ==============================
    if consecutive_losses >= 3:
        risk_percent *= 0.25
    elif consecutive_losses >= 2:
        risk_percent *= 0.50
    elif consecutive_losses == 1:
        risk_percent *= 0.75

    # ==============================
    # 3. تقليل الخطر بناءً على Drawdown
    # ==============================
    if equity < 80000:      # خسر أكثر من 20%
        risk_percent *= 0.25
    elif equity < 90000:    # خسر أكثر من 10%
        risk_percent *= 0.50
    elif equity < 95000:    # خسر أكثر من 5%
        risk_percent *= 0.75

    # ==============================
    # 4. حساب حجم الصفقة (الصيغة الصحيحة)
    # ==============================
    risk_amount = equity * risk_percent

    # Convert SL distance to pips
    pip = get_pip_value(symbol)
    sl_pips = sl_distance / pip if pip > 0 else 0.0

    # Pip value per 1.0 lot
    pip_value_per_lot = get_pip_value_per_lot(symbol)

    # Correct formula: lots = risk_amount / (sl_pips * pip_value_per_lot)
    if sl_pips > 0 and pip_value_per_lot > 0:
        size = risk_amount / (sl_pips * pip_value_per_lot)
    else:
        # Fallback: use price-based approximation (legacy)
        size = risk_amount / sl_distance

    # ==============================
    # 5. تطبيق الحدود الآمنة
    # ==============================
    max_lot = MAX_LOT_PER_SYMBOL.get(symbol, 0.10)
    size = round(max(MIN_LOT, min(size, max_lot)), 2)

    logger.info(
        f"Position size: {size} (equity={equity:.0f}, risk={risk_percent:.3f}, "
        f"sl_dist={sl_distance:.5f}, sl_pips={sl_pips:.2f}, pip_val_per_lot={pip_value_per_lot}, "
        f"score={score:.0f}, cons_loss={consecutive_losses})"
    )
    return size


def get_dynamic_risk_multiplier() -> float:
    """Dynamic risk based on recent performance"""
    stats = get_daily_stats()
    cons = stats.get("consecutive_losses", 0)
    open_trades = get_total_open_trades()

    if cons >= 3:
        return 0.25
    elif cons >= 2:
        return 0.5
    elif cons >= 1:
        return 0.75
    elif open_trades >= MAX_OPEN_TRADES:
        return 0.5
    else:
        return 1.0