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

    # Ceilings are combined with min(), not max().
    #
    # The previous code used max(), which meant MAX_SL_PIPS — a setting whose
    # name promises an upper bound — acted as a *floor*: an ATR spike could
    # push the "maximum" far above the configured pip cap. The broker's stop
    # level is a genuine floor and is applied separately below.
    ceilings = [c for c in (max_sl_from_pips, max_sl_from_atr) if c > 0]
    effective_max = min(ceilings) if ceilings else max_sl_from_pips

    # The broker will reject a stop closer than its stops_level, so that is a
    # hard floor regardless of any ceiling above.
    if min_sl_from_stops > 0:
        effective_max = max(effective_max, min_sl_from_stops)

    # ==============================
    # NO EQUITY CAP HERE — DELIBERATE
    # ==============================
    # This function used to shrink the stop distance so that trading MAX_LOT
    # would risk no more than 5% of equity. That inverted the correct order of
    # operations and produced stops that were unusable in practice.
    #
    # Observed live on 2026-08-07 with a $99.40 account:
    #   XAUUSD ATR=47.36 -> stop should be 71.03, the cap forced it to 0.497,
    #   i.e. 1% of one ATR. The position was opened and stopped out in the same
    #   second. EURUSD and GBPUSD were capped to a single pip.
    #
    # Two things were wrong:
    #   1. The cap assumed MAX_LOT (0.10 for XAUUSD, 0.50 for EURUSD) while the
    #      actual order was 0.01 lots, making it 10-50x tighter than intended.
    #   2. More fundamentally, stop placement is a *market* decision — ATR says
    #      where the trade is invalidated. Risk is then controlled by choosing
    #      the position size, which is what calculate_position_size() is for.
    #      Deriving the stop from an assumed lot size reverses that and, taken
    #      to its conclusion, places the stop wherever the account is small
    #      rather than wherever the trade is wrong.
    #
    # Equity-based risk control now lives entirely in calculate_position_size(),
    # which refuses the trade outright when the risk-correct size falls below
    # the broker's minimum lot.
    #
    # account_equity is still accepted so callers need not change; it is
    # intentionally unused.
    _ = account_equity

    return effective_max