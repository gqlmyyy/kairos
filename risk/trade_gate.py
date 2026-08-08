"""The single authoritative pre-trade validation gate.

Why this exists
---------------
Protections were spread across the entry path and some were simply not wired
in. Measured before this module existed:

* ``TradeManagementOrchestrator.check_entry_gate`` — 0 call sites
* ``layer1_intrabar.can_open_new_entry`` — 0 call sites
* ``RiskGovernor.can_open_new_position`` — reachable only through the unwired
  gate above, so the governor's MAX_OPEN_TRADES ceiling never ran

``main.py`` did check ``governor.is_halted()``, but once per *cycle* rather than
per trade, and it assembled ``final_decision_valid`` from a boolean expression
that mixed several unrelated conditions — easy to extend wrongly and impossible
to test in isolation.

Every entry now passes through :func:`validate_trade_request`, which returns
ALLOW or REJECT(reason). It is the only place that decides, and it fails closed:
anything it cannot verify is a rejection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger("trade_gate")


class GateDecision(str, Enum):
    ALLOW = "ALLOW"
    REJECT = "REJECT"


@dataclass(frozen=True)
class TradeRequest:
    """Everything the gate needs. Nothing is fetched inside the gate itself,
    so it stays pure and fully testable."""

    symbol: str
    direction: str
    final_score: float
    ai_confidence: float
    confidence: float
    equity: float
    position_size: Optional[float]
    sl_distance: Optional[float]
    tp_distance: Optional[float]
    signal_is_valid: bool
    ml_available: bool
    ml_p_win: Optional[float]
    ml_threshold: float
    ml_status: str = ""
    size_multiplier: float = 1.0
    open_position_count: Optional[int] = None
    risk_passed: bool = True
    risk_reason: str = ""


@dataclass(frozen=True)
class GateResult:
    decision: GateDecision
    reason: str = ""
    checks: tuple = field(default_factory=tuple)

    @property
    def allowed(self) -> bool:
        return self.decision is GateDecision.ALLOW


def _finite(value: Any) -> bool:
    """True only for a real, finite number."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(numeric) or math.isinf(numeric))


def _reject(reason: str, checks: list) -> GateResult:
    checks.append(f"REJECT:{reason}")
    logger.warning("[TRADE_GATE] REJECT — %s", reason)
    return GateResult(GateDecision.REJECT, reason, tuple(checks))


def validate_trade_request(
    request: TradeRequest,
    governor: Any = None,
) -> GateResult:
    """Decide whether this entry may reach the broker.

    Checks run cheapest-first, and every one of them is blocking. ``governor``
    is injected so the gate can be tested without global state; when omitted the
    process-wide RiskGovernor is used.
    """
    checks: list = []

    # --- 1. Signal validity -------------------------------------------------
    if not request.signal_is_valid:
        return _reject("signal_invalid", checks)
    if str(request.direction).upper() not in {"BUY", "SELL"}:
        return _reject(f"direction_invalid:{request.direction}", checks)
    checks.append("signal_valid")

    # --- 2. Numeric sanity --------------------------------------------------
    # NaN/Inf must never reach order construction. A NaN comparison is False,
    # so an unguarded value can slip through a naive `> 0` test.
    for name, value in (
        ("final_score", request.final_score),
        ("ai_confidence", request.ai_confidence),
        ("confidence", request.confidence),
        ("equity", request.equity),
        ("sl_distance", request.sl_distance),
        ("tp_distance", request.tp_distance),
        ("position_size", request.position_size),
        ("size_multiplier", request.size_multiplier),
    ):
        if value is None:
            return _reject(f"{name}_missing", checks)
        if not _finite(value):
            return _reject(f"{name}_not_finite:{value}", checks)
    checks.append("numerics_finite")

    if request.equity <= 0:
        return _reject(f"equity_not_positive:{request.equity}", checks)

    # --- 3. Stop / target ---------------------------------------------------
    if request.sl_distance <= 0:
        return _reject(f"sl_distance_not_positive:{request.sl_distance}", checks)
    if request.tp_distance < 0:
        return _reject(f"tp_distance_negative:{request.tp_distance}", checks)
    checks.append("sl_tp_valid")

    # --- 4. Position size ---------------------------------------------------
    # 0.0 is how calculate_position_size signals "this account cannot take this
    # trade at this stop" — it is a rejection, not a clamp.
    if request.position_size <= 0:
        return _reject(f"position_size_not_positive:{request.position_size}", checks)
    if request.size_multiplier <= 0:
        return _reject(f"size_multiplier_not_positive:{request.size_multiplier}", checks)
    checks.append("size_valid")

    # --- 5. Risk engine -----------------------------------------------------
    if not request.risk_passed:
        return _reject(f"risk_engine:{request.risk_reason or 'blocked'}", checks)
    checks.append("risk_engine_passed")

    # --- 6. ML gate ---------------------------------------------------------
    # available=False covers ML_GATE_INVALID, model missing and prediction
    # errors. A trade must never proceed on an unverified probability.
    if not request.ml_available:
        return _reject(f"ml_unavailable:{request.ml_status or 'no_status'}", checks)
    if request.ml_p_win is None or not _finite(request.ml_p_win):
        return _reject(f"ml_p_win_invalid:{request.ml_p_win}", checks)
    if request.ml_p_win < request.ml_threshold:
        return _reject(
            f"ml_below_threshold:{request.ml_p_win:.3f}<{request.ml_threshold}", checks
        )
    checks.append("ml_gate_passed")

    # --- 7. Risk Governor ---------------------------------------------------
    # Previously unreachable: this is the halt state AND the MAX_OPEN_TRADES
    # ceiling the governor owns.
    gov = governor
    if gov is None:
        try:
            from risk.risk_governor import get_risk_governor

            gov = get_risk_governor()
        except Exception as exc:
            # Fail closed. A governor that cannot be consulted is not a
            # governor that approved the trade.
            return _reject(f"risk_governor_unavailable:{exc}", checks)

    try:
        if gov.is_halted():
            return _reject(f"risk_governor_halted:{gov.get_halt_reason()}", checks)
        ok, gov_reason = gov.can_open_new_position(request.open_position_count)
        if not ok:
            return _reject(f"risk_governor:{gov_reason}", checks)
    except Exception as exc:
        return _reject(f"risk_governor_error:{exc}", checks)
    checks.append("risk_governor_passed")

    logger.info(
        "[TRADE_GATE] ALLOW %s %s size=%s p_win=%.3f checks=%d",
        request.symbol, request.direction, request.position_size,
        request.ml_p_win, len(checks),
    )
    return GateResult(GateDecision.ALLOW, "", tuple(checks))
