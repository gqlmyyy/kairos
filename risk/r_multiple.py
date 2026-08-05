"""Shared R-multiple calculation module.

Provides a single, canonical implementation for converting SL price distance
to dollar risk and computing R-multiple from PnL.

All callers (reconciliation, post_entry_manager, risk_governor) must use
this module to ensure consistent units and avoid the dollar/price-distance
mixing bug that was previously fixed in multiple places.
"""

from __future__ import annotations

from typing import Optional

from utils.logger import get_logger
from config import PIP_VALUES

logger = get_logger("r_multiple")


def calculate_risk_amount_usd(
    sl_distance_price: float,
    symbol: str,
    trade_size: float,
) -> Optional[float]:
    """Calculate the dollar amount risked on a trade.

    Converts SL price distance to dollars using:
        sl_pips = sl_distance / pip
        risk_amount_usd = sl_pips * pip_value_per_lot * trade_size

    Uses the SAME source for pip and pip_value_per_lot as position_sizing.py
    to ensure consistency.

    Args:
        sl_distance_price: SL distance in price units (e.g. 0.0030 for EURUSD).
        symbol: Trading symbol (e.g. "EURUSD", "XAUUSD").
        trade_size: Lot size (e.g. 0.50).

    Returns:
        Dollar amount risked, or None if inputs are invalid.
    """
    try:
        from risk.position_sizing import get_pip_value_per_lot

        sym_upper = str(symbol).upper()
        pip = float(PIP_VALUES.get(sym_upper, 0.0001))
        pip_value_per_lot = get_pip_value_per_lot(sym_upper)

        if pip <= 0 or pip_value_per_lot <= 0 or trade_size is None or trade_size <= 0:
            return None

        sl_distance = float(sl_distance_price)
        if sl_distance <= 0:
            return None

        sl_pips = sl_distance / pip
        risk_amount_usd = sl_pips * pip_value_per_lot * trade_size

        return risk_amount_usd
    except Exception as e:
        logger.error(f"calculate_risk_amount_usd error: {e}")
        return None


def calculate_r_multiple(
    pnl_usd: float,
    sl_distance_price: float,
    symbol: str,
    trade_size: float,
) -> float:
    """Calculate R-multiple from PnL and SL distance.

    Formula:
        risk_amount_usd = calculate_risk_amount_usd(...)
        r_multiple = pnl_usd / risk_amount_usd

    Fallback: if risk_amount_usd cannot be computed (None or <= 0),
    returns 1.0 for losses (conservative: 1R per loss) and 0.0 for wins.

    Args:
        pnl_usd: Realized PnL in account currency (negative = loss).
        sl_distance_price: SL distance in price units.
        symbol: Trading symbol.
        trade_size: Lot size.

    Returns:
        R-multiple (float). Negative for losses, positive for wins.
    """
    risk = calculate_risk_amount_usd(sl_distance_price, symbol, trade_size)
    pnl = float(pnl_usd) if pnl_usd is not None else 0.0

    if risk is not None and risk > 0:
        return pnl / risk
    else:
        # Fallback: 1R per loss, 0R per win (conservative)
        return 1.0 if pnl < 0 else 0.0