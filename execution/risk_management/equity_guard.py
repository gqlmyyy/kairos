"""Equity Guard

Stops trading / closes all open positions when account equity drop or
consecutive losses exceed configured thresholds.

Defensive: never raises out of check_equity_guard.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from utils.logger import get_logger

from config import (
    EQUITY_GUARD_ENABLED,
    MAX_DAILY_LOSS_PCT,
    MAX_CONSECUTIVE_LOSSES,
)

logger = get_logger("equity_guard")


def check_equity_guard(
    current_equity: float,
    starting_equity: float,
    consecutive_losses: int,
    open_positions: List[dict],
    close_all_fn: Optional[Callable[[], bool]] = None,
) -> List[str]:
    """Return order_ids to close (or close_all_fn result).

    Requirements (from spec):
    - Daily loss limit: (starting_equity - current_equity) / starting_equity >= MAX_DAILY_LOSS_PCT
    - Consecutive losses limit: consecutive_losses >= MAX_CONSECUTIVE_LOSSES

    Args:
        open_positions: list of qd-like dicts that include at least {'id' or 'ticket'}
        close_all_fn: optional callable to close all positions directly.

    Returns:
        List[str] of order_ids/tickets to close.
    """
    try:
        if not EQUITY_GUARD_ENABLED:
            return []

        if starting_equity is None or starting_equity <= 0:
            return []

        cur_eq = float(current_equity) if current_equity is not None else None
        if cur_eq is None:
            return []

        daily_loss_pct = (float(starting_equity) - cur_eq) / float(starting_equity)
        daily_loss_hit = daily_loss_pct >= float(MAX_DAILY_LOSS_PCT)
        consecutive_hit = int(consecutive_losses or 0) >= int(MAX_CONSECUTIVE_LOSSES)

        if not (daily_loss_hit or consecutive_hit):
            return []

        logger.warning("[EQUITY_GUARD] Daily loss limit reached. Closing all trades.")

        if close_all_fn is not None:
            # If close_all_fn works, return empty list (already closed)
            ok = bool(close_all_fn())
            return [] if ok else _extract_order_ids(open_positions)

        return _extract_order_ids(open_positions)

    except Exception as e:
        logger.error(f"check_equity_guard error: {e}")
        return []


def _extract_order_ids(open_positions: List[dict]) -> List[str]:
    out: List[str] = []
    try:
        for p in open_positions or []:
            if not isinstance(p, dict):
                continue
            oid = p.get("id", p.get("ticket", ""))
            if oid is None:
                continue
            s = str(oid)
            if s.strip():
                out.append(s)
    except Exception:
        pass
    # de-dup while preserving order
    seen = set()
    unique = []
    for x in out:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    return unique

