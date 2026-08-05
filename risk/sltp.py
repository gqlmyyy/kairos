# Trading Bot V3 - risk/sltp.py
# ATR-based Stop Loss and Take Profit calculation (regime-aware)

from utils.logger import get_logger
from config import (
    MAX_SL_PIPS,
    PIP_VALUES,
    ATR_SL_BASE_MULTIPLIER,
    ATR_TP_BASE_MULTIPLIER,
)
from risk.symbol_info import get_max_sl_distance

logger = get_logger("sltp")


def get_pip_value(symbol: str) -> float:
    return PIP_VALUES.get(symbol, 0.0001)


def _regime_multipliers(regime: str) -> tuple[float, float]:
    """Return (sl_multiplier, tp_multiplier) adjusted for market regime.

    Regime adaptation:
      - high_volatility / volatile: wider SL (protect against noise), normal TP
      - trending: normal SL, extended TP (let winners run)
      - mean_reversion / ranging: tighter SL, tighter TP (quick scalps)
      - normal / unknown: base multipliers
    """
    r = str(regime or "").strip().lower()

    if r in ("high_volatility", "volatile"):
        return ATR_SL_BASE_MULTIPLIER * 1.3, ATR_TP_BASE_MULTIPLIER * 1.0
    if r in ("trending", "trend", "strong uptrend", "strong downtrend"):
        return ATR_SL_BASE_MULTIPLIER * 1.0, ATR_TP_BASE_MULTIPLIER * 1.3
    if r in ("mean_reversion", "ranging", "sideways"):
        return ATR_SL_BASE_MULTIPLIER * 0.8, ATR_TP_BASE_MULTIPLIER * 0.8
    if r in ("weak trend", "weak_trend"):
        return ATR_SL_BASE_MULTIPLIER * 0.9, ATR_TP_BASE_MULTIPLIER * 1.1

    # Normal / unknown -> base
    return ATR_SL_BASE_MULTIPLIER, ATR_TP_BASE_MULTIPLIER


def calculate_sl_tp_distances(
    symbol: str,
    atr: float,
    regime: str = "Normal",
    account_equity: float = None,
) -> tuple:
    """Calculate SL/TP as DISTANCES (not absolute prices).
    
    This is the preferred method for live trading - calculates SL/TP as distances
    from entry price, which can then be applied to the live MT5 price at execution time.
    This eliminates price discrepancy issues between data sources.

    Args:
        symbol: trading symbol (e.g. XAUUSD)
        atr: ATR value
        regime: market regime string (e.g. "Trending", "Volatile", "Ranging", "Normal")
        account_equity: current account equity (for absolute SL cap)

    Returns:
        (sl_distance, tp_distance) tuple in price units
        These distances should be applied to live_price at execution time:
        - BUY: sl = live_price - sl_distance, tp = live_price + tp_distance
        - SELL: sl = live_price + sl_distance, tp = live_price - tp_distance
    """
    # General symbol-aware max SL cap (uses broker symbol_info.point + trade_stops_level + ATR + absolute cap)
    max_sl_distance = get_max_sl_distance(
        symbol,
        max_sl_pips=MAX_SL_PIPS,
        atr=atr,
        account_equity=account_equity,
    )

    # Regime-aware multipliers
    sl_mult, tp_mult = _regime_multipliers(regime)

    # SL = ATR × sl_mult (capped)
    sl_distance = min(atr * sl_mult, max_sl_distance)
    # TP = ATR × tp_mult
    tp_distance = atr * tp_mult

    logger.info(
        f"SL/TP DISTANCES {symbol}: atr={atr:.5f} regime={regime} "
        f"sl_mult={sl_mult:.2f} tp_mult={tp_mult:.2f} "
        f"sl_distance={sl_distance:.5f} tp_distance={tp_distance:.5f}"
    )
    return sl_distance, tp_distance


def calculate_sl_tp(
    symbol: str,
    entry_price: float,
    direction: str,
    atr: float,
    regime: str = "Normal",
    account_equity: float = None,
) -> tuple:
    """Calculate SL/TP using real ATR with max SL cap, adapted to market regime.
    
    LEGACY FUNCTION - Uses absolute prices based on entry_price.
    For live trading, prefer calculate_sl_tp_distances() to avoid price discrepancy issues.

    Args:
        symbol: trading symbol (e.g. XAUUSD)
        entry_price: entry price
        direction: "BUY" or "SELL"
        atr: ATR value
        regime: market regime string (e.g. "Trending", "Volatile", "Ranging", "Normal")
        account_equity: current account equity (for absolute SL cap)

    Returns:
        (sl, tp) tuple
    """
    # Use the new distance-based calculation
    sl_distance, tp_distance = calculate_sl_tp_distances(
        symbol, atr, regime, account_equity
    )

    # Apply distances to entry price
    if direction == "BUY":
        sl = round(entry_price - sl_distance, 5)
        tp = round(entry_price + tp_distance, 5)
    else:
        sl = round(entry_price + sl_distance, 5)
        tp = round(entry_price - tp_distance, 5)

    logger.info(
        f"SL/TP {symbol}: entry={entry_price} atr={atr:.5f} regime={regime} "
        f"sl_mult={_regime_multipliers(regime)[0]:.2f} tp_mult={_regime_multipliers(regime)[1]:.2f} "
        f"sl_distance={sl_distance:.5f} tp_distance={tp_distance:.5f} "
        f"sl={sl} tp={tp}"
    )
    return sl, tp
