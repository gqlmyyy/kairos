"""Layer 1 - Risk Governor gate.

Entry-side only. This layer answers one question: may a *new* trade be opened
right now? It never inspects, modifies or closes an open position.

Input : open position count (optional)
Output : GateDecision(allowed, reason)

The underlying RiskGovernor still records closed trades to accumulate the daily
loss in R — that bookkeeping is what feeds this gate — but the gate itself is
read-only with respect to live trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

logger = get_logger("tm.risk_gate")


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str = ""
    halt_sources: tuple = ()

    def __bool__(self) -> bool:
        return self.allowed


def check_entry_allowed(open_position_count: Optional[int] = None) -> GateDecision:
    """Return whether a new entry may proceed.

    Fails open on internal error: a broken governor must not silently block all
    trading, but it is logged loudly.
    """
    try:
        from risk.risk_governor import get_risk_governor

        governor = get_risk_governor()

        if governor.is_halted():
            reason = governor.get_halt_reason() or "risk governor halted"
            sources = tuple(governor.get_halt_sources() or ())
            logger.warning("[TM_L1_GATE] entry blocked: %s sources=%s", reason, sources)
            return GateDecision(False, reason, sources)

        ok, reason = governor.can_open_new_position(open_position_count)
        if not ok:
            logger.warning("[TM_L1_GATE] entry blocked: %s", reason)
            return GateDecision(False, reason)

        return GateDecision(True, "")
    except Exception as exc:
        logger.error("[TM_L1_GATE] governor check failed, allowing entry: %s", exc)
        return GateDecision(True, f"governor_error:{exc}")


def record_closed_trade(order_id: str, pnl_usd: float, risk_amount_usd: Optional[float]) -> None:
    """Feed a closed trade's result into the governor's daily accounting.

    Deduplicated by order_id inside the governor, so repeated calls are safe.
    """
    try:
        from risk.risk_governor import get_risk_governor

        get_risk_governor().record_trade_close(
            pnl_usd=float(pnl_usd or 0.0),
            risk_amount_usd=risk_amount_usd,
            order_id=str(order_id),
        )
    except Exception as exc:
        logger.error("[TM_L1_GATE] record_trade_close failed order_id=%s: %s", order_id, exc)
