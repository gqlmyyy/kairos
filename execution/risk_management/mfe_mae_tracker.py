"""MFE/MAE Tracker

Tracks maximum favorable excursion (peak_profit) and maximum adverse excursion
(max_loss) for each order_id during its lifetime.

This is intentionally defensive and does not raise.

Currently, it logs the values at the moment it detects that an order is no longer
present in the latest qd_positions snapshot.

Future improvement: Persist mfe/mae into DB when schema supports it.
"""

from __future__ import annotations

from typing import Dict, Optional

from utils.logger import get_logger

from config import MFE_MAE_ENABLED


logger = get_logger("mfe_mae_tracker")

# in-memory tracking
# order_id -> {peak_profit, max_loss}
_state: Dict[str, Dict[str, float]] = {}


def track_mfe_mae(order_id: str, current_profit: float) -> None:
    """Update peak_profit and max_loss for an order.

    Args:
        order_id: MT5/QuantDinger order position id
        current_profit: current profit (unrealized)
    """
    try:
        if not MFE_MAE_ENABLED:
            return
        if order_id is None:
            return
        oid = str(order_id).strip()
        if not oid:
            return

        cur = float(current_profit)

        s = _state.get(oid)
        if s is None:
            _state[oid] = {
                "peak_profit": cur,
                "max_loss": cur,
            }
            return

        if cur > s["peak_profit"]:
            s["peak_profit"] = cur
        if cur < s["max_loss"]:
            s["max_loss"] = cur
    except Exception as e:
        logger.error(f"track_mfe_mae error: {e}")


def flush_closed_mfe_mae(current_open_order_ids: set):
    """Detect closed orders (not in current snapshot) and log their MFE/MAE.

    Args:
        current_open_order_ids: set of currently open position ids
    """
    if not MFE_MAE_ENABLED:
        return

    try:
        # anything in _state but not in current_open_order_ids is considered closed
        to_flush = [oid for oid in list(_state.keys()) if oid not in current_open_order_ids]
        for oid in to_flush:
            s = _state.pop(oid, None)
            if not s:
                continue
            logger.info(
                f"[MFE/MAE] Closed order={oid} mfe(peak)={s['peak_profit']:.2f} mae(max_loss)={s['max_loss']:.2f}"
            )
    except Exception as e:
        logger.error(f"flush_closed_mfe_mae error: {e}")


def _reset_mfe_mae_state():
    """Testing helper."""
    _state.clear()

