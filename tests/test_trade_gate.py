"""H-04 regression: exactly one authoritative pre-trade gate.

Before this, protections were assembled inline in main.py and two were never
reachable at all — ``check_entry_gate`` and ``can_open_new_position`` had zero
live call sites, so the Risk Governor's per-trade ceiling never ran.

Properties proved here:
  PROPERTY 4 — no order can bypass the Risk Governor
  PROPERTY 5 — no order can bypass MAX_OPEN_TRADES
  PROPERTY 6 — no order can bypass final entry validation
  PROPERTY 7 — NaN/Inf cannot reach order construction
"""

from __future__ import annotations

import pytest

from risk.trade_gate import GateDecision, TradeRequest, validate_trade_request


class FakeGovernor:
    def __init__(self, halted=False, reason="", can_open=True, open_reason=""):
        self._halted = halted
        self._reason = reason
        self._can_open = can_open
        self._open_reason = open_reason

    def is_halted(self):
        return self._halted

    def get_halt_reason(self):
        return self._reason

    def can_open_new_position(self, count=None):
        return self._can_open, self._open_reason


class ExplodingGovernor:
    def is_halted(self):
        raise RuntimeError("governor exploded")

    def get_halt_reason(self):
        return ""

    def can_open_new_position(self, count=None):
        raise RuntimeError("governor exploded")


def good_request(**overrides) -> TradeRequest:
    base = dict(
        symbol="EURUSD", direction="BUY",
        final_score=70.0, ai_confidence=0.75, confidence=0.8,
        equity=10_000.0, position_size=0.05,
        sl_distance=0.0025, tp_distance=0.0060,
        signal_is_valid=True,
        ml_available=True, ml_p_win=0.72, ml_threshold=0.60, ml_status="OK",
        size_multiplier=1.0, open_position_count=0,
        risk_passed=True, risk_reason="",
    )
    base.update(overrides)
    return TradeRequest(**base)


class TestAllowPath:
    def test_all_gates_pass(self):
        result = validate_trade_request(good_request(), governor=FakeGovernor())
        assert result.decision is GateDecision.ALLOW
        assert result.allowed is True
        assert result.reason == ""

    def test_allow_records_every_check(self):
        result = validate_trade_request(good_request(), governor=FakeGovernor())
        for expected in ("signal_valid", "numerics_finite", "sl_tp_valid",
                         "size_valid", "risk_engine_passed", "ml_gate_passed",
                         "risk_governor_passed"):
            assert expected in result.checks


class TestSignalGate:
    def test_invalid_signal_rejected(self):
        r = validate_trade_request(good_request(signal_is_valid=False), FakeGovernor())
        assert r.reason == "signal_invalid"

    @pytest.mark.parametrize("direction", ["NEUTRAL", "", "HOLD", "buy_maybe"])
    def test_bad_direction_rejected(self, direction):
        r = validate_trade_request(good_request(direction=direction), FakeGovernor())
        assert not r.allowed


class TestNumericGate:
    """PROPERTY 7 — NaN/Inf cannot reach order construction."""

    @pytest.mark.parametrize(
        "field",
        ["final_score", "ai_confidence", "confidence", "equity",
         "sl_distance", "tp_distance", "position_size", "size_multiplier"],
    )
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_rejected(self, field, bad):
        r = validate_trade_request(good_request(**{field: bad}), FakeGovernor())
        assert not r.allowed
        assert field in r.reason

    @pytest.mark.parametrize(
        "field", ["position_size", "sl_distance", "tp_distance", "equity"]
    )
    def test_none_rejected(self, field):
        r = validate_trade_request(good_request(**{field: None}), FakeGovernor())
        assert not r.allowed
        assert "missing" in r.reason

    def test_nan_position_size_cannot_slip_through_a_naive_comparison(self):
        """`nan > 0` is False, but the message must name the real cause."""
        r = validate_trade_request(good_request(position_size=float("nan")), FakeGovernor())
        assert "not_finite" in r.reason


class TestStopAndSizeGates:
    @pytest.mark.parametrize("sl", [0.0, -0.001])
    def test_invalid_sl_rejected(self, sl):
        r = validate_trade_request(good_request(sl_distance=sl), FakeGovernor())
        assert "sl_distance" in r.reason

    def test_negative_tp_rejected(self):
        r = validate_trade_request(good_request(tp_distance=-1.0), FakeGovernor())
        assert "tp_distance" in r.reason

    def test_zero_tp_allowed_for_open_ended_profiles(self):
        """Trend profiles clear the fixed target and let trailing exit."""
        r = validate_trade_request(good_request(tp_distance=0.0), FakeGovernor())
        assert r.allowed

    @pytest.mark.parametrize("size", [0.0, -0.01])
    def test_non_positive_size_rejected(self, size):
        r = validate_trade_request(good_request(position_size=size), FakeGovernor())
        assert "position_size" in r.reason

    def test_zero_size_from_risk_rejection_blocks(self):
        """calculate_position_size returns 0.0 to refuse — honour it."""
        r = validate_trade_request(good_request(position_size=0.0), FakeGovernor())
        assert not r.allowed

    def test_zero_multiplier_rejected(self):
        r = validate_trade_request(good_request(size_multiplier=0.0), FakeGovernor())
        assert "size_multiplier" in r.reason


class TestRiskEngineGate:
    def test_risk_engine_rejection_honoured(self):
        r = validate_trade_request(
            good_request(risk_passed=False, risk_reason="daily loss limit"),
            FakeGovernor(),
        )
        assert "risk_engine" in r.reason
        assert "daily loss limit" in r.reason


class TestMLGate:
    """PROPERTY 1 reinforced at the gate level."""

    def test_unavailable_ml_rejected(self):
        r = validate_trade_request(
            good_request(ml_available=False, ml_status="ML_GATE_INVALID"), FakeGovernor()
        )
        assert "ml_unavailable" in r.reason
        assert "ML_GATE_INVALID" in r.reason

    def test_none_p_win_rejected(self):
        r = validate_trade_request(good_request(ml_p_win=None), FakeGovernor())
        assert "ml_p_win_invalid" in r.reason

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_p_win_rejected(self, bad):
        r = validate_trade_request(good_request(ml_p_win=bad), FakeGovernor())
        assert not r.allowed

    def test_below_threshold_rejected(self):
        r = validate_trade_request(good_request(ml_p_win=0.55), FakeGovernor())
        assert "ml_below_threshold" in r.reason

    def test_exactly_at_threshold_allowed(self):
        r = validate_trade_request(
            good_request(ml_p_win=0.60, ml_threshold=0.60), FakeGovernor()
        )
        assert r.allowed


class TestRiskGovernorGate:
    """PROPERTY 4 and PROPERTY 5 — previously unreachable."""

    def test_halted_governor_blocks(self):
        r = validate_trade_request(
            good_request(), FakeGovernor(halted=True, reason="6R cumulative loss")
        )
        assert "risk_governor_halted" in r.reason
        assert "6R" in r.reason

    def test_max_open_trades_ceiling_blocks(self):
        r = validate_trade_request(
            good_request(open_position_count=3),
            FakeGovernor(can_open=False, open_reason="max open positions 3 >= 3"),
        )
        assert "risk_governor" in r.reason
        assert "max open positions" in r.reason

    def test_governor_exception_fails_closed(self):
        """An unconsultable governor is not an approving governor."""
        r = validate_trade_request(good_request(), ExplodingGovernor())
        assert not r.allowed
        assert "risk_governor_error" in r.reason

    def test_governor_runs_even_when_everything_else_passes(self):
        """Regression: the governor used to be skipped entirely."""
        result = validate_trade_request(good_request(), FakeGovernor())
        assert "risk_governor_passed" in result.checks


class TestGateOrdering:
    def test_first_failure_short_circuits(self):
        r = validate_trade_request(
            good_request(signal_is_valid=False, position_size=0.0, ml_available=False),
            FakeGovernor(halted=True),
        )
        assert r.reason == "signal_invalid"

    def test_only_allow_can_reach_execution(self):
        """PROPERTY 6 — sweep of every single-failure variant."""
        variants = [
            good_request(signal_is_valid=False),
            good_request(direction="NEUTRAL"),
            good_request(equity=0.0),
            good_request(sl_distance=0.0),
            good_request(position_size=0.0),
            good_request(size_multiplier=0.0),
            good_request(risk_passed=False),
            good_request(ml_available=False),
            good_request(ml_p_win=0.1),
            good_request(final_score=float("nan")),
        ]
        for request in variants:
            assert not validate_trade_request(request, FakeGovernor()).allowed

        for governor in (
            FakeGovernor(halted=True),
            FakeGovernor(can_open=False, open_reason="limit"),
            ExplodingGovernor(),
        ):
            assert not validate_trade_request(good_request(), governor).allowed
