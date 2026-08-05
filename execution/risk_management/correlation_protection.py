"""Correlation Protection

Prevents opening new trades that are too correlated with existing open positions.
Defensive: should never raise.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from utils.logger import get_logger

from config import (
    CORRELATION_PROTECTION_ENABLED,
    CORRELATION_ACTION,
    CORRELATED_PAIRS,
)

logger = get_logger("correlation_protection")


def check_correlation(symbol1: str, symbol2: str) -> bool:
    """Return True if the pair is considered correlated."""
    if not symbol1 or not symbol2:
        return False
    s1 = str(symbol1).upper()
    s2 = str(symbol2).upper()
    for a, b in CORRELATED_PAIRS:
        if (s1 == a.upper() and s2 == b.upper()) or (s1 == b.upper() and s2 == a.upper()):
            return True
    return False


def _get_symbol_direction(pos: dict) -> tuple[str, str]:
    sym = str(pos.get("symbol", "") or pos.get("Symbol", "") or "").upper()
    d_raw = pos.get("direction", pos.get("type", ""))
    d = str(d_raw).strip().lower()
    if d in ["buy", "0", "long", "1"]:
        return sym, "buy"
    if d in ["sell", "short", "-1"]:
        return sym, "sell"
    # best-effort
    return sym, "buy" if "buy" in d else ("sell" if "sell" in d else "buy")


def is_correlated_open(
    symbol: str,
    direction: str,
    open_positions: Iterable[dict],
    *,
    action: Optional[str] = None,
) -> bool:
    """Return True if a correlated open trade exists with same direction.

    If action == "close_old", caller should close old trades; this function only reports.
    For now: for "close_old" we also return True (signal to caller to handle closing).
    """
    try:
        if not CORRELATION_PROTECTION_ENABLED:
            return False

        if not symbol:
            return False

        action_eff = (action or CORRELATION_ACTION or "block").lower()
        new_symbol = str(symbol).upper()
        new_dir = str(direction).strip().lower()
        if new_dir in ["sell", "short", "-1", "1"]:
            new_dir = "sell"
        else:
            new_dir = "buy"

        for pos in open_positions or []:
            try:
                existing_symbol, existing_dir = _get_symbol_direction(pos)
                if not existing_symbol:
                    continue

                if existing_dir != new_dir:
                    continue

                if check_correlation(new_symbol, existing_symbol):
                    logger.info(
                        f"[CORRELATION_PROTECTION] Block correlated {new_symbol} {new_dir} with {existing_symbol} {existing_dir} action={action_eff}"
                    )
                    return True
            except Exception:
                continue

        return False

    except Exception as e:
        logger.error(f"is_correlated_open error: {e}")
        return False

