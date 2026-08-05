"""Symbol Info Utility - General MAX_SL computation from broker data.

Computes the maximum SL distance for any symbol using MT5's symbol_info:
  - point: the smallest price increment
  - trade_stops_level: minimum distance (in points) between price and SL/TP

This replaces hardcoded per-symbol overrides with a general formula.
"""

from __future__ import annotations

from typing import Optional

from utils.logger import get_logger
from config import PIP_VALUES

logger = get_logger("symbol_info")

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    mt5 = None

# Fallback point values if MT5 is unavailable (used only as last resort)
_FALLBACK_POINT = {
    "XAUUSD": 0.01,
    "EURUSD": 0.00001,
    "GBPUSD": 0.00001,
    "USDJPY": 0.001,
    "USDCAD": 0.00001,
    "AUDUSD": 0.00001,
    "NZDUSD": 0.00001,
    "USDCHF": 0.00001,
}

# Fallback stops level (in points) if MT5 is unavailable
_FALLBACK_STOPS_LEVEL = 0


def get_symbol_point(symbol: str) -> float:
    """Return the symbol's point size (smallest price increment).

    Prefers MT5 symbol_info.point. Falls back to a static map.
    """
    if mt5 is not None:
        try:
            info = mt5.symbol_info(symbol)
            if info is not None and getattr(info, "point", None):
                p = float(info.point)
                if p > 0:
                    return p
        except Exception:
            pass
    return _FALLBACK_POINT.get(symbol, 0.00001)


def get_symbol_stops_level(symbol: str) -> int:
    """Return the symbol's trade_stops_level (minimum SL/TP distance in points).

    Prefers MT5 symbol_info.trade_stops_level. Falls back to 0.
    """
    if mt5 is not None:
        try:
            info = mt5.symbol_info(symbol)
            if info is not None and getattr(info, "trade_stops_level", None) is not None:
                return int(info.trade_stops_level)
        except Exception:
            pass
    return _FALLBACK_STOPS_LEVEL


# Absolute cap on SL distance as a fraction of account equity.
# This is an ULTIMATE safety net independent of ATR: even if ATR spikes 5x,
# the SL cannot exceed this percentage of account equity risked per trade.
# 0.05 = 5% of account equity as the absolute max risk in DOLLARS.
# The dollar amount is then converted to a PRICE DISTANCE using lot_size
# and pip_value_per_lot (same source as position_sizing.py).
_ABSOLUTE_SL_ACCOUNT_FRACTION = 0.05

# Max lot per symbol fallback (must match position_sizing.py).
_MAX_LOT_FALLBACK = 0.10


def _get_pip_value_per_lot(symbol: str) -> float:
    """Return pip value per 1.0 lot in account currency.

    Uses the SAME source as position_sizing.py (PIP_VALUE_PER_LOT dict)
    to ensure get_max_sl_distance and calculate_position_size stay consistent.
    Imported lazily to avoid circular imports.
    """
    try:
        from risk.position_sizing import get_pip_value_per_lot
        return get_pip_value_per_lot(symbol)
    except Exception:
        return 10.0


def _get_max_lot(symbol: str) -> float:
    """Return max allowed lot for symbol (same source as position_sizing.py)."""
    try:
        from risk.position_sizing import MAX_LOT_PER_SYMBOL
        return MAX_LOT_PER_SYMBOL.get(str(symbol).upper(), _MAX_LOT_FALLBACK)
    except Exception:
        return _MAX_LOT_FALLBACK


def get_max_sl_distance(symbol: str, max_sl_pips: float = 100.0, atr: float = None,
                        account_equity: float = None) -> float:
    """Compute the maximum SL distance in price units for a symbol.

    Formula:
        max_sl_distance = min(
            max(
                max_sl_pips * pip,            # user-configured pip cap
                atr * 3.0 if atr else 0,       # ATR-based cap (for high-volatility symbols)
                stops_level * point            # broker minimum stops level
            ),
            equity_cap_price_distance          # ABSOLUTE cap based on account equity
        )

    Where:
        pip = PIP_VALUES[symbol] (same source as position_sizing.py)
        stops_level = symbol_info.trade_stops_level (in points)
        point = symbol_info.point
        atr = current ATR value (optional)
        account_equity = current account equity (optional, account currency).

    The equity_cap_price_distance is computed by converting the dollar cap
    (equity * 5%) to a price distance:

        max_risk_dollars = account_equity * _ABSOLUTE_SL_ACCOUNT_FRACTION
        max_lot = MAX_LOT_PER_SYMBOL[symbol]
        pip_value_per_lot = PIP_VALUE_PER_LOT[symbol]  (dollars per pip per 1.0 lot)
        dollars_per_pip = max_lot * pip_value_per_lot
        max_sl_pips_from_cap = max_risk_dollars / dollars_per_pip
        equity_cap_price_distance = max_sl_pips_from_cap * pip

    This ensures the SL never risks more than 5% of equity even during an
    extreme ATR spike, AND the units are consistent (price distance vs price distance).

    Example for $500 account, XAUUSD (max_lot=0.10, pip_value_per_lot=$10, pip=0.1):
        max_risk_dollars = $500 * 0.05 = $25
        dollars_per_pip = 0.10 * $10 = $1.00/pip
        max_sl_pips_from_cap = $25 / $1.00 = 25 pips
        equity_cap_price_distance = 25 * 0.1 = 2.5 (price units)

    If ATR=5.0, atr*3=15.0 > 2.5, so the cap triggers and limits SL to 2.5.
    """
    point = get_symbol_point(symbol)
    stops_level = get_symbol_stops_level(symbol)

    # Determine pip value — use PIP_VALUES from config (same source as
    # position_sizing.py and sltp.py) to avoid any mismatch.
    sym_upper = str(symbol).upper()
    pip = float(PIP_VALUES.get(sym_upper, 0.0001))

    # Broker minimum SL distance (in price units)
    min_sl_from_stops = stops_level * point

    # User-configured max SL (in price units)
    max_sl_from_pips = max_sl_pips * pip

    # ATR-based cap: allow SL up to 3x ATR (generous but bounded)
    max_sl_from_atr = 0.0
    if atr is not None:
        try:
            atr_f = float(atr)
            if atr_f > 0:
                max_sl_from_atr = atr_f * 3.0
        except Exception:
            max_sl_from_atr = 0.0

    # The effective max SL from all constraints (before absolute cap)
    effective_max = max(min_sl_from_stops, max_sl_from_pips, max_sl_from_atr)

    # ==============================
    # ABSOLUTE CAP (account-equity based) — UNITS FIX
    # ==============================
    # Convert the dollar cap (equity * 5%) to a PRICE DISTANCE using
    # lot_size and pip_value_per_lot — the SAME source as position_sizing.py.
    #
    # OLD (BUGGY): min(effective_max, equity * 0.05)
    #   -> compares price distance (e.g. 0.0030) with dollars (e.g. $5000)
    #   -> cap NEVER triggers because $5000 >> 0.0030
    #
    # NEW (FIXED): convert dollars -> price distance first, then compare
    if account_equity is not None:
        try:
            eq = float(account_equity)
            if eq > 0 and pip > 0:
                max_risk_dollars = eq * _ABSOLUTE_SL_ACCOUNT_FRACTION
                max_lot = _get_max_lot(sym_upper)
                pip_value_per_lot = _get_pip_value_per_lot(sym_upper)
                dollars_per_pip = max_lot * pip_value_per_lot
                if dollars_per_pip > 0:
                    max_sl_pips_from_cap = max_risk_dollars / dollars_per_pip
                    equity_cap_price_distance = max_sl_pips_from_cap * pip
                    effective_max = min(effective_max, equity_cap_price_distance)
                    logger.debug(
                        f"[EQUITY_CAP] {symbol}: equity={eq:.0f} max_risk=${max_risk_dollars:.2f} "
                        f"max_lot={max_lot} pip_val_per_lot={pip_value_per_lot} "
                        f"dollars_per_pip={dollars_per_pip:.2f} "
                        f"cap_pips={max_sl_pips_from_cap:.1f} "
                        f"cap_price_dist={equity_cap_price_distance:.5f} "
                        f"effective_max={effective_max:.5f}"
                    )
        except Exception as e:
            logger.error(f"[EQUITY_CAP] {symbol}: conversion error: {e}")

    return effective_max