"""C-02 regression: one logical signal must never open two positions.

The defect: ``open_trade`` looped over filling modes and treated both an
``order_send`` exception and a ``None`` result as failure, moving straight to
the next mode. Both actually mean *outcome unknown*. With 198 ``No IPC
connection`` errors in the live logs, the "order executed but reply was lost"
path is not hypothetical.

These tests cover the ten scenarios required by the remediation brief.
"""

from __future__ import annotations

import threading

import pytest

from execution.order_idempotency import (
    ExecutionRecord,
    ExecutionState,
    SignalIdentity,
    find_position_for_signal,
    may_send_another_order,
    resolve_unknown_outcome,
)


def identity(symbol="XAUUSD", direction="BUY", ts=1_786_118_400):
    return SignalIdentity(symbol=symbol, direction=direction, signal_ts=ts)


def position(magic=0, symbol="XAUUSD", ptype=0, opened=1_786_118_400, ticket=999):
    return {"magic": magic, "symbol": symbol, "type": ptype, "time": opened, "ticket": ticket}


class TestSignalIdentity:
    def test_magic_is_deterministic_across_retries(self):
        """A random id per attempt would make reconciliation impossible."""
        assert identity().magic == identity().magic

    def test_different_signals_get_different_magics(self):
        assert identity(direction="BUY").magic != identity(direction="SELL").magic
        assert identity(symbol="EURUSD").magic != identity(symbol="XAUUSD").magic
        assert identity(ts=1).magic != identity(ts=2).magic

    def test_magic_fits_in_mt5_int32(self):
        for sym in ("EURUSD", "XAUUSD", "GBPUSD"):
            for d in ("BUY", "SELL"):
                assert 0 <= identity(sym, d).magic < 2_147_483_647


class TestPositionMatching:
    def test_matches_by_magic(self):
        ident = identity()
        found = find_position_for_signal(
            ident, lambda s: [position(magic=ident.magic)], now=ident.signal_ts + 1
        )
        assert found is not None

    def test_matches_recent_same_side_position_without_magic(self):
        ident = identity()
        found = find_position_for_signal(
            ident, lambda s: [position(magic=0, opened=ident.signal_ts)],
            now=ident.signal_ts + 5,
        )
        assert found is not None

    def test_ignores_opposite_direction(self):
        ident = identity(direction="BUY")
        found = find_position_for_signal(
            ident, lambda s: [position(magic=0, ptype=1, opened=ident.signal_ts)],
            now=ident.signal_ts + 5,
        )
        assert found is None

    def test_ignores_old_position(self):
        ident = identity()
        found = find_position_for_signal(
            ident, lambda s: [position(magic=0, opened=ident.signal_ts - 9999)],
            now=ident.signal_ts,
        )
        assert found is None

    def test_no_positions_means_no_match(self):
        assert find_position_for_signal(identity(), lambda s: [], now=0) is None

    def test_lookup_failure_propagates(self):
        """Must raise, not return None — None would read as 'not executed'."""
        def boom(_symbol):
            raise RuntimeError("IPC down")

        with pytest.raises(RuntimeError):
            find_position_for_signal(identity(), boom)


class TestAmbiguousOutcomeResolution:
    def test_4_timeout_after_broker_execution_is_confirmed_executed(self):
        """The dangerous case: order landed, reply lost."""
        record = ExecutionRecord(identity=identity())
        outcome = resolve_unknown_outcome(
            record,
            lambda s: [position(magic=record.identity.magic, ticket=555)],
            now=record.identity.signal_ts + 1,
        )
        assert outcome is ExecutionState.CONFIRMED_EXECUTED
        assert record.order_id == "555"
        assert record.already_executed

    def test_3_timeout_before_broker_execution_is_confirmed_not_executed(self):
        record = ExecutionRecord(identity=identity())
        outcome = resolve_unknown_outcome(record, lambda s: [], now=0)
        assert outcome is ExecutionState.CONFIRMED_NOT_EXECUTED
        assert record.may_retry

    def test_broker_unreachable_stays_unknown(self):
        """Cannot prove non-execution -> must not retry."""
        def boom(_symbol):
            raise RuntimeError("IPC down")

        record = ExecutionRecord(identity=identity())
        outcome = resolve_unknown_outcome(record, boom)
        assert outcome is ExecutionState.UNKNOWN
        assert not record.may_retry


class TestRetryGate:
    def test_8_retry_blocked_after_confirmed_execution(self):
        record = ExecutionRecord(identity=identity())
        record.transition(ExecutionState.CONFIRMED_EXECUTED, "test")
        allowed, reason = may_send_another_order(record)
        assert allowed is False
        assert "already executed" in reason

    def test_retry_blocked_while_outcome_unknown(self):
        record = ExecutionRecord(identity=identity())
        record.transition(ExecutionState.UNKNOWN, "test")
        allowed, reason = may_send_another_order(record)
        assert allowed is False
        assert "unknown" in reason.lower()

    def test_7_retry_allowed_after_confirmed_non_execution(self):
        record = ExecutionRecord(identity=identity())
        record.transition(ExecutionState.CONFIRMED_NOT_EXECUTED, "test")
        assert may_send_another_order(record)[0] is True

    def test_2_retry_allowed_after_plain_rejection(self):
        record = ExecutionRecord(identity=identity())
        record.transition(ExecutionState.REJECTED, "invalid stops")
        assert may_send_another_order(record)[0] is True

    def test_1_fresh_signal_may_submit(self):
        assert may_send_another_order(ExecutionRecord(identity=identity()))[0] is True

    def test_in_flight_submission_blocks_a_second(self):
        record = ExecutionRecord(identity=identity())
        record.transition(ExecutionState.SUBMITTING, "attempt 1")
        assert may_send_another_order(record)[0] is False


class TestEndToEndDuplicatePrevention:
    """PROPERTY 2: one logical signal -> at most one executed position."""

    def _simulate_filling_loop(self, order_send, positions_provider, modes=("FOK", "IOC", "RETURN")):
        """Mirror of open_trade's loop, exercising the same guard sequence."""
        record = ExecutionRecord(identity=identity())
        sends = []
        for mode in modes:
            allowed, _ = may_send_another_order(record)
            if not allowed:
                break
            record.transition(ExecutionState.SUBMITTING, mode)
            sends.append(mode)
            result = order_send(mode)
            if result is None:
                record.transition(ExecutionState.UNKNOWN, "None reply")
                outcome = resolve_unknown_outcome(record, positions_provider)
                if outcome is ExecutionState.CONFIRMED_EXECUTED:
                    return sends, "success_recovered"
                if outcome is ExecutionState.UNKNOWN:
                    return sends, "aborted_unresolved"
                continue
            if result == "done":
                record.transition(ExecutionState.EXECUTED, mode)
                return sends, "success"
            record.transition(ExecutionState.REJECTED, mode)
        return sends, "failed"

    def test_5_response_none_but_position_exists_sends_only_once(self):
        ident = identity()
        sends, status = self._simulate_filling_loop(
            order_send=lambda mode: None,
            positions_provider=lambda s: [position(magic=ident.magic, ticket=777)],
        )
        assert len(sends) == 1, f"sent {len(sends)} orders — duplicate risk"
        assert status == "success_recovered"

    def test_6_response_none_and_no_position_may_try_next_mode(self):
        sends, status = self._simulate_filling_loop(
            order_send=lambda mode: None,
            positions_provider=lambda s: [],
        )
        assert len(sends) == 3
        assert status == "failed"

    def test_broker_unreachable_aborts_after_one_send(self):
        def boom(_symbol):
            raise RuntimeError("IPC down")

        sends, status = self._simulate_filling_loop(
            order_send=lambda mode: None, positions_provider=boom
        )
        assert len(sends) == 1
        assert status == "aborted_unresolved"

    def test_normal_success_sends_once(self):
        sends, status = self._simulate_filling_loop(
            order_send=lambda mode: "done", positions_provider=lambda s: []
        )
        assert sends == ["FOK"]
        assert status == "success"

    def test_9_same_signal_twice_yields_same_identity(self):
        """A repeated call must reconcile to the first position, not duplicate."""
        first, second = identity(), identity()
        assert first.magic == second.magic
        found = find_position_for_signal(
            second, lambda s: [position(magic=first.magic)], now=second.signal_ts + 1
        )
        assert found is not None

    def test_10_concurrent_submissions_produce_one_execution(self):
        """Two threads racing the same signal."""
        record = ExecutionRecord(identity=identity())
        lock = threading.Lock()
        executed = []

        def worker():
            with lock:
                allowed, _ = may_send_another_order(record)
                if not allowed:
                    return
                record.transition(ExecutionState.SUBMITTING, "racing")
            executed.append(1)
            with lock:
                record.transition(ExecutionState.EXECUTED, "done")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(executed) == 1, f"{len(executed)} concurrent executions"
