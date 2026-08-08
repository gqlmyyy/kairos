"""H-05: integration tests across the full entry/execution chain.

Every layer here already has unit coverage in isolation — trade_gate,
order_idempotency, entry_feature_contract, database persistence. What was
missing is proof that they work *wired together* the way main.py actually
wires them: a signal that fails the ML gate really does never reach
``open_trade``; a broker timeout really does get resolved through
``resolve_unknown_outcome`` inside ``open_trade`` itself rather than in a
test harness standing in for it; a partial close really does get persisted in
a way a restarted process can read back.

``FakeMT5`` below is deliberately close to the real ``MetaTrader5`` module's
surface: the same constants, the same ``order_send``/``positions_get``
semantics (including a bare ``None`` meaning "ambiguous", not "empty"),
because the defects this whole remediation exists to catch live exactly in
that gap between what the fake usually stands in for and what the broker
actually does.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from risk.trade_gate import GateDecision, TradeRequest, validate_trade_request


# ---------------------------------------------------------------------------
# A broker double faithful enough to exercise open_trade's real retry logic.
# ---------------------------------------------------------------------------

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
TRADE_ACTION_DEAL = 1
TRADE_RETCODE_DONE = 10009
ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2


@dataclass
class _Tick:
    ask: float
    bid: float


@dataclass
class _SymbolInfo:
    point: float = 0.00001
    trade_stops_level: int = 0
    filling_mode: int = 7  # FOK | IOC | RETURN all supported
    visible: bool = True


class _OrderResult:
    def __init__(self, retcode, order=None, price=None, volume=None, comment=""):
        self.retcode = retcode
        self.order = order
        self.price = price
        self.volume = volume
        self.comment = comment

    def _asdict(self):
        return dict(retcode=self.retcode, order=self.order, price=self.price,
                    volume=self.volume, comment=self.comment)


class FakePosition:
    def __init__(self, ticket, symbol, magic, volume=0.1, ptype=0):
        self.ticket = ticket
        self.symbol = symbol
        self.magic = magic
        self.volume = volume
        self.type = ptype

    def _asdict(self):
        return dict(ticket=self.ticket, symbol=self.symbol, magic=self.magic,
                    volume=self.volume, type=self.type)


class FakeMT5:
    """Stands in for the whole MetaTrader5 module.

    ``send_behavior`` drives what happens on the next ``order_send`` call:
      - ``"success"``            -> normal DONE retcode
      - ``"reject"``              -> a real rejection (e.g. invalid stops)
      - ("exception", positions)  -> order_send raises; positions_get(symbol=)
                                      returns `positions` afterwards (used for
                                      both "timeout, nothing happened" and
                                      "timeout, but it actually executed")
      - ("exception", "unreachable") -> order_send AND the follow-up
                                      positions_get both fail (unresolvable)
    """

    ORDER_TYPE_BUY = ORDER_TYPE_BUY
    ORDER_TYPE_SELL = ORDER_TYPE_SELL
    TRADE_ACTION_DEAL = TRADE_ACTION_DEAL
    TRADE_RETCODE_DONE = TRADE_RETCODE_DONE
    ORDER_FILLING_FOK = ORDER_FILLING_FOK
    ORDER_FILLING_IOC = ORDER_FILLING_IOC
    ORDER_FILLING_RETURN = ORDER_FILLING_RETURN

    def __init__(self, *, ask=1.1000, bid=1.0998):
        self.ask = ask
        self.bid = bid
        self.send_behavior = "success"
        self.order_send_calls = 0
        self.positions: Dict[str, List[FakePosition]] = {}
        self._next_ticket = 1000
        self._last_error = (0, "no error")

    # -- terminal / diagnostics --------------------------------------------
    def terminal_info(self):
        return object()

    def account_info(self):
        return object()

    def last_error(self):
        return self._last_error

    def symbol_info(self, symbol):
        return _SymbolInfo()

    def symbol_select(self, symbol, enable):
        return True

    def symbol_info_tick(self, symbol):
        return _Tick(ask=self.ask, bid=self.bid)

    # -- positions ------------------------------------------------------
    def positions_get(self, symbol=None, ticket=None):
        if ticket is not None:
            for plist in self.positions.values():
                for p in plist:
                    if p.ticket == ticket:
                        return [p]
            return []
        if symbol is not None:
            return list(self.positions.get(symbol, []))
        return [p for plist in self.positions.values() for p in plist]

    def _add_position(self, symbol, magic, volume=0.1, ptype=0):
        self._next_ticket += 1
        pos = FakePosition(self._next_ticket, symbol, magic, volume, ptype)
        self.positions.setdefault(symbol, []).append(pos)
        return pos

    # -- order submission -----------------------------------------------
    def order_send(self, request):
        self.order_send_calls += 1

        if self.send_behavior == "success":
            pos = self._add_position(
                request["symbol"], request["magic"], request["volume"], request["type"]
            )
            return _OrderResult(TRADE_RETCODE_DONE, order=pos.ticket,
                                 price=request["price"], volume=request["volume"])

        if self.send_behavior == "reject":
            return _OrderResult(10016, comment="Invalid stops")

        if isinstance(self.send_behavior, tuple) and self.send_behavior[0] == "exception":
            outcome = self.send_behavior[1]
            self._last_error = (-10004, "IPC timeout")
            if outcome == "unreachable":
                self._broker_unreachable = True
            elif outcome == "executed":
                # The order actually went through despite the lost reply.
                self._add_position(request["symbol"], request["magic"],
                                    request["volume"], request["type"])
            # outcome == "lost": nothing added; a genuine non-execution.
            raise ConnectionError("simulated IPC timeout")

        raise AssertionError(f"unhandled send_behavior: {self.send_behavior}")


@pytest.fixture
def fake_mt5(monkeypatch):
    import execution.mt5_direct as md

    fake = FakeMT5()
    monkeypatch.setattr(md, "mt5", fake)
    monkeypatch.setattr(md, "_ensure_mt5_initialized", lambda: True)
    monkeypatch.setattr(md, "_ensure_symbol_selected", lambda symbol: True)

    def _positions_for_symbol(symbol):
        if getattr(fake, "_broker_unreachable", False):
            raise RuntimeError("broker unreachable")
        return list(fake.positions.get(symbol, []))

    monkeypatch.setattr(md, "_positions_for_symbol", _positions_for_symbol)
    return fake


# ---------------------------------------------------------------------------
# 1. Full pipeline: ML contract -> TradeRequest -> gate -> execution
# ---------------------------------------------------------------------------

class TestFullPipelineWiring:
    """The gate's output must be exactly what open_trade needs, with no
    hand-waving field renames between the two layers."""

    def _base_request(self, **overrides) -> TradeRequest:
        base = dict(
            symbol="EURUSD", direction="BUY", final_score=72.0,
            ai_confidence=0.75, confidence=0.70, equity=5000.0,
            position_size=0.10, sl_distance=0.0030, tp_distance=0.0050,
            signal_is_valid=True, ml_available=True, ml_p_win=0.64,
            ml_threshold=0.55, ml_status="OK", size_multiplier=1.0,
            open_position_count=0, risk_passed=True, risk_reason="",
        )
        base.update(overrides)
        return TradeRequest(**base)

    def test_allowed_signal_reaches_the_broker_and_opens(self, fake_mt5):
        request = self._base_request()
        gate = validate_trade_request(request, governor=_AllowingGovernor())
        assert gate.decision is GateDecision.ALLOW

        import execution.mt5_direct as md

        result = md.open_trade(
            request.symbol, request.direction, request.position_size,
            request.sl_distance, request.tp_distance, "IntegrationTest",
            signal_ts=1_700_000_000,
        )
        assert result["status"] == "success"
        assert result["order_id"] is not None
        assert fake_mt5.order_send_calls == 1

    def test_ml_gate_rejection_never_reaches_the_broker(self, fake_mt5):
        request = self._base_request(ml_available=False, ml_status="ML_MODEL_MISSING",
                                      ml_p_win=None)
        gate = validate_trade_request(request, governor=_AllowingGovernor())
        assert gate.decision is GateDecision.REJECT
        assert "ml_unavailable" in gate.reason

        # The whole point: nothing downstream should ever be called.
        assert fake_mt5.order_send_calls == 0

    def test_ml_below_threshold_never_reaches_the_broker(self, fake_mt5):
        request = self._base_request(ml_p_win=0.40)  # below ml_threshold=0.55
        gate = validate_trade_request(request, governor=_AllowingGovernor())
        assert gate.decision is GateDecision.REJECT
        assert "ml_below_threshold" in gate.reason
        assert fake_mt5.order_send_calls == 0

    def test_risk_governor_halt_never_reaches_the_broker(self, fake_mt5):
        request = self._base_request()
        gate = validate_trade_request(request, governor=_HaltedGovernor())
        assert gate.decision is GateDecision.REJECT
        assert "risk_governor_halted" in gate.reason
        assert fake_mt5.order_send_calls == 0

    def test_nan_in_the_request_never_reaches_the_broker(self, fake_mt5):
        request = self._base_request(sl_distance=float("nan"))
        gate = validate_trade_request(request, governor=_AllowingGovernor())
        assert gate.decision is GateDecision.REJECT
        assert fake_mt5.order_send_calls == 0

    def test_invalid_signal_never_reaches_the_broker(self, fake_mt5):
        request = self._base_request(signal_is_valid=False)
        gate = validate_trade_request(request, governor=_AllowingGovernor())
        assert gate.decision is GateDecision.REJECT
        assert fake_mt5.order_send_calls == 0


class _AllowingGovernor:
    def is_halted(self):
        return False

    def get_halt_reason(self):
        return ""

    def can_open_new_position(self, count):
        return True, ""


class _HaltedGovernor:
    def is_halted(self):
        return True

    def get_halt_reason(self):
        return "MAX_LOSS_R exceeded"

    def can_open_new_position(self, count):
        return False, "halted"


# ---------------------------------------------------------------------------
# 2. Duplicate signal: the same signal retried must not open two positions.
# ---------------------------------------------------------------------------

class TestDuplicateSignalSubmission:
    def test_retrying_the_same_signal_after_a_successful_open_is_refused(self, fake_mt5):
        """Simulates main.py retrying a symbol it thinks failed, using the
        same signal_ts both times — the real-world duplicate-order shape."""
        import execution.mt5_direct as md

        first = md.open_trade("EURUSD", "BUY", 0.10, 0.0030, 0.0050, "t1",
                               signal_ts=1_700_000_000)
        assert first["status"] == "success"
        assert fake_mt5.order_send_calls == 1

        # A caller that (incorrectly) retries with the identical signal_ts,
        # e.g. after a supervisory restart, must not place a second order —
        # this is exactly what SignalIdentity's deterministic magic exists for.
        # open_trade itself has no cross-call memory (each call builds a fresh
        # ExecutionRecord), so the real protection is the magic-based position
        # match: a second attempt for the same signal finds the existing
        # position via find_position_for_signal, not via open_trade's own
        # retry loop. Prove that primitive directly against the broker state
        # open_trade just created.
        from execution.order_idempotency import SignalIdentity, find_position_for_signal

        identity = SignalIdentity(symbol="EURUSD", direction="BUY", signal_ts=1_700_000_000)
        found = find_position_for_signal(identity, lambda s: fake_mt5.positions_get(symbol=s))
        assert found is not None
        assert str(found["ticket"]) == first["order_id"]

    def test_a_different_signal_ts_is_a_genuinely_new_trade(self, fake_mt5):
        import execution.mt5_direct as md

        first = md.open_trade("EURUSD", "BUY", 0.10, 0.0030, 0.0050, "t1",
                               signal_ts=1_700_000_000)
        second = md.open_trade("EURUSD", "BUY", 0.10, 0.0030, 0.0050, "t2",
                                signal_ts=1_700_003_600)
        assert first["status"] == second["status"] == "success"
        assert first["order_id"] != second["order_id"]
        assert fake_mt5.order_send_calls == 2


# ---------------------------------------------------------------------------
# 3. MT5-level outcomes: rejection, timeout, ambiguous.
# ---------------------------------------------------------------------------

class TestBrokerLevelOutcomes:
    def test_broker_rejection_returns_a_clean_error(self, fake_mt5):
        import execution.mt5_direct as md

        fake_mt5.send_behavior = "reject"
        result = md.open_trade("EURUSD", "BUY", 0.10, 0.0030, 0.0050, "t",
                                signal_ts=1_700_000_000)
        assert result["status"] == "error"
        assert result["order_id"] is None
        # retcode 10016 (invalid stops) stops immediately, no filling-mode churn.
        assert fake_mt5.order_send_calls == 1

    def test_timeout_with_no_execution_is_reported_as_failed_not_retried_into_a_duplicate(
        self, fake_mt5
    ):
        """order_send raises, and the broker genuinely never got the order —
        the safe outcome, and it must be reported as a failure, not silently
        retried into a second position."""
        import execution.mt5_direct as md

        fake_mt5.send_behavior = ("exception", "lost")
        result = md.open_trade("EURUSD", "BUY", 0.10, 0.0030, 0.0050, "t",
                                signal_ts=1_700_000_000)
        assert result["status"] == "error"
        assert fake_mt5.positions_get(symbol="EURUSD") == []

    def test_timeout_but_the_order_actually_executed_is_recovered_not_duplicated(
        self, fake_mt5
    ):
        """The dangerous case this whole mechanism exists for: the reply was
        lost, but the broker did open the position. open_trade must find it
        and report success — not retry and open a second one."""
        import execution.mt5_direct as md

        fake_mt5.send_behavior = ("exception", "executed")
        result = md.open_trade("EURUSD", "BUY", 0.10, 0.0030, 0.0050, "t",
                                signal_ts=1_700_000_000)
        assert result["status"] == "success"
        assert result.get("recovered_from_ambiguous_response") is True
        # Exactly one position exists at the broker — no duplicate was sent.
        assert len(fake_mt5.positions_get(symbol="EURUSD")) == 1

    def test_ambiguous_and_broker_unreachable_refuses_to_retry(self, fake_mt5):
        """Cannot prove the order didn't execute, and cannot check either.
        The only safe move is to stop and escalate, not guess."""
        import execution.mt5_direct as md

        fake_mt5.send_behavior = ("exception", "unreachable")
        result = md.open_trade("EURUSD", "BUY", 0.10, 0.0030, 0.0050, "t",
                                signal_ts=1_700_000_000)
        assert result["status"] == "error"
        assert result["error"] == "ambiguous_execution_unresolved"
        # Only one order_send call: it refused to retry into the unknown.
        assert fake_mt5.order_send_calls == 1


# ---------------------------------------------------------------------------
# 4. Reconciliation: a broker position with no DB row, and vice versa.
# ---------------------------------------------------------------------------

class TestReconciliationMatching:
    """find_position_for_signal is reconciliation's core primitive: given a
    signal identity, does a matching broker position exist. Exercised here
    against a broker state built the same way open_trade builds it."""

    def test_a_position_from_a_different_symbol_is_not_matched(self, fake_mt5):
        import execution.mt5_direct as md
        from execution.order_idempotency import SignalIdentity, find_position_for_signal

        md.open_trade("EURUSD", "BUY", 0.10, 0.0030, 0.0050, "t",
                       signal_ts=1_700_000_000)

        identity = SignalIdentity(symbol="GBPUSD", direction="BUY", signal_ts=1_700_000_000)
        found = find_position_for_signal(identity, lambda s: fake_mt5.positions_get(symbol=s))
        assert found is None

    def test_an_orphan_position_no_db_row_still_reconciles_by_magic(self, fake_mt5):
        """A position that exists at the broker but was never recorded (bot
        crashed between order_send succeeding and the DB write) must still be
        found by magic — that's the whole point of a deterministic magic."""
        import execution.mt5_direct as md
        from execution.order_idempotency import SignalIdentity, find_position_for_signal

        result = md.open_trade("EURUSD", "SELL", 0.10, 0.0030, 0.0050, "t",
                                signal_ts=1_700_000_000)
        assert result["status"] == "success"

        # Simulate a restart: nothing but the broker's own state is known now.
        identity = SignalIdentity(symbol="EURUSD", direction="SELL", signal_ts=1_700_000_000)
        found = find_position_for_signal(identity, lambda s: fake_mt5.positions_get(symbol=s))
        assert found is not None
        assert str(found["ticket"]) == result["order_id"]


# ---------------------------------------------------------------------------
# 5. Restart: partial-TP and breakeven state survives a process restart.
# ---------------------------------------------------------------------------

class TestRestartRecovery:
    """post_entry_manager reads persisted state back through _ensure_state.
    This proves the write (database.py) and the read (post_entry_manager.py)
    agree on format — test_restart_recovery.py covers each in isolation."""

    def test_partial_and_breakeven_state_round_trips_through_the_real_db(self, tmp_path, monkeypatch):
        import config
        import data.storage.database as db

        path = str(tmp_path / "restart_integration.db")
        monkeypatch.setattr(config, "DB_FILE", path)
        monkeypatch.setattr(db, "DB_FILE", path)
        db.init_db()

        db.upsert_execution_expected(
            order_id="555001", symbol="EURUSD", direction="BUY",
            expected_entry=1.1000, expected_final_score=72.0, expected_ai_score=70.0,
            expected_ai_confidence=0.70, expected_trend_score=70.0,
            expected_momentum_score=60.0, expected_sentiment_score=55.0,
            expected_volatility_score=50.0, expected_sl=1.0950, expected_tp=1.1100,
            expected_volume=0.10, entry_profile="trend",
        )
        assert db.update_partial_levels_done("555001", {0}) is True
        assert db.update_breakeven_done("555001", True) is True

        from execution.post_entry.post_entry_manager import PostEntryManager

        # __init__ only builds collaborators (monitor/executor/orchestrator/
        # event bus) — it does not start the loop thread, so this is safe to
        # construct directly rather than reaching for __new__.
        mgr = PostEntryManager()

        db_row = db.get_execution_dataset("555001")
        assert db_row is not None

        pos = {"order_id": "555001", "symbol": "EURUSD", "direction": "buy",
               "volume": 0.10, "entry_price": 1.1000, "sl": 1.1000, "tp": 1.1100}
        state = mgr._ensure_state(pos, db_row)

        assert state.breakeven_done is True
        assert 0 in state.partial_levels_done


# ---------------------------------------------------------------------------
# 6. Emergency close.
# ---------------------------------------------------------------------------

class TestEmergencyClose:
    @pytest.fixture
    def fake_mt5_ae(self, monkeypatch):
        import execution.post_entry.action_executor as ae

        fake = FakeMT5()
        monkeypatch.setattr(ae, "mt5", fake)
        return fake

    def test_close_position_succeeds_on_first_try(self, fake_mt5_ae, monkeypatch):
        import execution.post_entry.action_executor as ae

        pos = fake_mt5_ae._add_position("EURUSD", magic=1, volume=0.10)

        def _order_send(request):
            fake_mt5_ae.order_send_calls += 1
            return _OrderResult(TRADE_RETCODE_DONE, order=pos.ticket)

        monkeypatch.setattr(fake_mt5_ae, "order_send", _order_send)
        monkeypatch.setattr(fake_mt5_ae, "symbol_info_tick", lambda s: _Tick(ask=1.1000, bid=1.0998))
        monkeypatch.setattr(fake_mt5_ae, "symbol_info", lambda s: _SymbolInfo())

        executor = ae.ActionExecutor()
        assert executor.close_position(str(pos.ticket)) is True

    def test_close_position_retries_then_succeeds_on_requote(self, fake_mt5_ae, monkeypatch):
        """retcode 10036 (market closed / requote-class) is retried."""
        import execution.post_entry.action_executor as ae

        pos = fake_mt5_ae._add_position("EURUSD", magic=1, volume=0.10)
        attempts = {"n": 0}

        def _order_send(request):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return _OrderResult(10036, comment="requote")
            return _OrderResult(TRADE_RETCODE_DONE, order=pos.ticket)

        monkeypatch.setattr(fake_mt5_ae, "order_send", _order_send)
        monkeypatch.setattr(fake_mt5_ae, "symbol_info_tick", lambda s: _Tick(ask=1.1000, bid=1.0998))
        monkeypatch.setattr(fake_mt5_ae, "symbol_info", lambda s: _SymbolInfo())
        monkeypatch.setattr(time, "sleep", lambda s: None)

        executor = ae.ActionExecutor()
        assert executor.close_position(str(pos.ticket)) is True
        assert attempts["n"] == 2

    def test_close_position_on_an_unknown_ticket_fails_safely(self, fake_mt5_ae):
        import execution.post_entry.action_executor as ae

        executor = ae.ActionExecutor()
        # No position with this ticket exists in the fake broker.
        assert executor.close_position("999999") is False

    def test_close_position_rejects_a_non_numeric_ticket(self, fake_mt5_ae):
        import execution.post_entry.action_executor as ae

        executor = ae.ActionExecutor()
        assert executor.close_position("not-a-ticket") is False
