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


def resolve_ml_mode(mode: Optional[str] = None) -> str:
    """The effective ENTRY_ML_MODE. Reads config when not injected.

    Fails closed: an unreadable config yields `required`, the strictest mode.
    A mode setting that cannot be read must never be the reason a trade slips
    past the filter.
    """
    if mode is not None:
        return mode
    try:
        from config import ENTRY_ML_MODE

        return ENTRY_ML_MODE
    except Exception:  # noqa: BLE001 - fail closed, never open
        return "required"


def absent_size_multiplier(value: Optional[float] = None) -> float:
    """Sizing to use when the ML gate is not sizing the trade."""
    if value is not None:
        return float(value)
    try:
        from config import ENTRY_ML_ABSENT_SIZE_MULT

        return float(ENTRY_ML_ABSENT_SIZE_MULT)
    except Exception:  # noqa: BLE001
        return 0.5


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
    ml_mode: Optional[str] = None,
) -> GateResult:
    """Decide whether this entry may reach the broker.

    Checks run cheapest-first, and every one of them is blocking. ``governor``
    is injected so the gate can be tested without global state; when omitted the
    process-wide RiskGovernor is used. ``ml_mode`` likewise defaults to
    ``config.ENTRY_ML_MODE`` and exists so tests need no environment.
    """
    checks: list = []
    mode = resolve_ml_mode(ml_mode)

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

    # --- 4. ML availability -------------------------------------------------
    # Checked BEFORE the size check, and that ordering is the whole point.
    #
    # `size_multiplier` is DERIVED from `ml_p_win` — main.py computes it via
    # get_size_multiplier(p_win), which yields 0.0 when there is no
    # probability to size from. So an absent model produces a zero multiplier
    # as a downstream *symptom*. With the size check first, every rejection
    # caused by a missing model was reported as
    # `size_multiplier_not_positive:0.0`, which points at position sizing and
    # sends the reader to the risk engine — while the real cause is that
    # models/entry/entry_model.json has no metadata sidecar and was never
    # loaded. Reporting a symptom as the cause cost real debugging time; the
    # gate now names the cause.
    #
    # available=False covers ML_GATE_INVALID, model missing and prediction
    # errors. In `required` a trade must never proceed on an unverified
    # probability. In `advisory` the absence is logged and the trade proceeds
    # on signal + MTF alone; in `off` the gate is bypassed outright.
    if mode == "off":
        checks.append("ml_bypassed:off")
        logger.warning(
            "[ML_MODE] mode=off — ML gate BYPASSED for %s %s. This trade is not "
            "filtered by any model.", request.symbol, request.direction)
    elif not request.ml_available:
        if mode == "advisory":
            checks.append(f"ml_absent_advisory:{request.ml_status or 'no_status'}")
            logger.warning(
                "[ML_MODE] mode=advisory — model unavailable (%s) for %s %s. "
                "Trading on signal + MTF WITHOUT ML filtering, at reduced size.",
                request.ml_status or "no_status", request.symbol, request.direction)
        else:
            return _reject(f"ml_unavailable:{request.ml_status or 'no_status'}", checks)
    else:
        checks.append("ml_available")

    # --- 5. Position size ---------------------------------------------------
    # 0.0 is how calculate_position_size signals "this account cannot take this
    # trade at this stop" — it is a rejection, not a clamp.
    if request.position_size <= 0:
        return _reject(f"position_size_not_positive:{request.position_size}", checks)
    if request.size_multiplier <= 0:
        return _reject(f"size_multiplier_not_positive:{request.size_multiplier}", checks)
    checks.append("size_valid")

    # --- 6. Risk engine -----------------------------------------------------
    if not request.risk_passed:
        return _reject(f"risk_engine:{request.risk_reason or 'blocked'}", checks)
    checks.append("risk_engine_passed")

    # --- 7. ML probability --------------------------------------------------
    # The rest of the ML gate: these judge the probability's VALUE, which is
    # only meaningful once sizing and risk have passed — and only when there
    # is a probability at all.
    #
    # Reached in `required` always (availability above guaranteed a model),
    # and in `advisory` only when a model IS serving — in which case its
    # threshold applies in full, exactly as in `required`. `advisory` softens
    # what happens when the model is ABSENT, never what a present model says.
    if mode == "off" or (mode == "advisory" and not request.ml_available):
        checks.append(f"ml_threshold_skipped:{mode}")
    else:
        if request.ml_p_win is None or not _finite(request.ml_p_win):
            return _reject(f"ml_p_win_invalid:{request.ml_p_win}", checks)
        if request.ml_p_win < request.ml_threshold:
            return _reject(
                f"ml_below_threshold:{request.ml_p_win:.3f}<{request.ml_threshold}", checks
            )
        checks.append("ml_gate_passed")

    # --- 8. Risk Governor ---------------------------------------------------
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

    # p_win is legitimately None in advisory/off, so it is formatted rather
    # than passed to %.3f — which would raise on None and turn an allowed
    # trade into a crash inside the logging call.
    p_win_text = (f"{request.ml_p_win:.3f}"
                  if isinstance(request.ml_p_win, float) and _finite(request.ml_p_win)
                  else "n/a")
    logger.info(
        "[TRADE_GATE] ALLOW %s %s size=%s p_win=%s ml_mode=%s checks=%d",
        request.symbol, request.direction, request.position_size,
        p_win_text, mode, len(checks),
    )
    return GateResult(GateDecision.ALLOW, "", tuple(checks))
