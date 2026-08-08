"""M-04: exception handlers on the risk-gating path must fail closed.

Two defects in ``risk.risk_engine.can_trade`` used to let an error in a
*safety* check become a trade *approval*:

1. **The MT5 duplicate-position check** (step 1) explicitly said "allow
   trading if MT5 query fails" and caught every exception. It also treated
   ``positions_get()`` returning ``None`` the same as "zero positions found" —
   the MT5 API documents ``None`` as ambiguous between that and "an error
   occurred." Either path meant that exactly when MT5 was least reliable, the
   one check that exists to stop a second position opening on a symbol that
   already had one was silently skipped.

2. **The correlation-protection module** (step 7b) wrapped its own import and
   a database read in a bare ``except Exception: pass``, which turned a broken
   import or a locked/corrupt database into ``return True, "OK"`` — risk-check
   approval.

Both now reject the trade when the check itself cannot be trusted. A check
that cannot be verified is not a check that passed.
"""

from __future__ import annotations

import sys
import types

import pytest


def _reload_risk_engine():
    """risk_engine imports MT5 lazily inside can_trade, so no module reload
    is needed between tests — but importing fresh keeps this file order-safe
    if it ever grows a module-level import."""
    import risk.risk_engine as re_mod

    return re_mod


class _FakePosition:
    pass


class _FakeMT5:
    """Stands in for the MetaTrader5 module, keyed to specific test scenarios."""

    def __init__(self, *, positions=(), raise_on_call=None, none_result=False):
        self._positions = positions
        self._raise = raise_on_call
        self._none_result = none_result
        self.calls = 0

    def positions_get(self, symbol=None):
        self.calls += 1
        if self._raise:
            raise self._raise
        if self._none_result:
            return None
        return list(self._positions)

    def last_error(self):
        return (1, "connection lost")


@pytest.fixture
def base_kwargs(monkeypatch):
    """Get every check after step 1 (duplicate) to pass, so failures at step 1
    are what surfaces as the return value."""
    import risk.risk_engine as re_mod

    monkeypatch.setattr(re_mod, "get_daily_stats", lambda: {"total_pnl": 0, "consecutive_losses": 0})
    monkeypatch.setattr(re_mod, "get_total_open_trades", lambda: 0)
    monkeypatch.setattr(re_mod, "get_open_trades", lambda: [])
    monkeypatch.setattr(re_mod, "check_drawdown", lambda pnl, equity: {"action": "none", "reason": ""})
    monkeypatch.setattr(re_mod, "MAX_OPEN_TRADES", 5)
    monkeypatch.setattr(re_mod, "STOP_AFTER_LOSSES", 999)
    return dict(symbol="EURUSD", direction="BUY", final_score=80.0,
                ai_confidence=0.9, equity=5000.0)


def _patch_mt5_session(monkeypatch, fake_mt5, *, session_up=True):
    import data.market.mt5_session as session

    fake_module = types.ModuleType("data.market.mt5_session")
    monkeypatch.setattr(session, "mt5", fake_mt5)
    monkeypatch.setattr(session, "ensure_session", lambda *a, **k: session_up)

    from contextlib import contextmanager

    @contextmanager
    def _fake_lock():
        yield

    monkeypatch.setattr(session, "mt5_call", _fake_lock)


class TestDuplicatePositionCheckFailsClosed:
    def test_no_open_position_allows_the_trade(self, monkeypatch, base_kwargs):
        re_mod = _reload_risk_engine()
        fake = _FakeMT5(positions=())
        _patch_mt5_session(monkeypatch, fake)

        passed, reason = re_mod.can_trade(**base_kwargs)
        assert passed is True

    def test_an_existing_position_blocks_the_trade(self, monkeypatch, base_kwargs):
        re_mod = _reload_risk_engine()
        fake = _FakeMT5(positions=[_FakePosition()])
        _patch_mt5_session(monkeypatch, fake)

        passed, reason = re_mod.can_trade(**base_kwargs)
        assert passed is False
        assert "Duplicate" in reason

    def test_an_exception_now_rejects_instead_of_approving(self, monkeypatch, base_kwargs):
        """The exact defect: this used to log 'allowing trade' and proceed."""
        re_mod = _reload_risk_engine()
        fake = _FakeMT5(raise_on_call=ConnectionError("IPC failure"))
        _patch_mt5_session(monkeypatch, fake)

        passed, reason = re_mod.can_trade(**base_kwargs)
        assert passed is False
        assert "duplicate" in reason.lower() or "failed" in reason.lower()

    def test_none_result_is_treated_as_unverifiable_not_as_zero_positions(
        self, monkeypatch, base_kwargs
    ):
        """positions_get() returning None is documented by MT5 as ambiguous
        between 'no results' and 'an error occurred' — it must not silently
        mean 'no open position'."""
        re_mod = _reload_risk_engine()
        fake = _FakeMT5(none_result=True)
        _patch_mt5_session(monkeypatch, fake)

        passed, reason = re_mod.can_trade(**base_kwargs)
        assert passed is False
        assert "unavailable" in reason.lower() or "none" in reason.lower()

    def test_session_down_rejects_before_calling_positions_get(self, monkeypatch, base_kwargs):
        re_mod = _reload_risk_engine()
        fake = _FakeMT5(positions=())
        _patch_mt5_session(monkeypatch, fake, session_up=False)

        passed, reason = re_mod.can_trade(**base_kwargs)
        assert passed is False
        assert fake.calls == 0, "positions_get() called despite a down session"

    def test_the_call_is_routed_through_the_shared_session_lock(self, monkeypatch, base_kwargs):
        """H-01: no live MT5 read may bypass data.market.mt5_session."""
        import data.market.mt5_session as session

        re_mod = _reload_risk_engine()
        fake = _FakeMT5(positions=())
        monkeypatch.setattr(session, "mt5", fake)
        monkeypatch.setattr(session, "ensure_session", lambda *a, **k: True)

        lock_entered = {"count": 0}
        from contextlib import contextmanager

        @contextmanager
        def _tracked_lock():
            lock_entered["count"] += 1
            yield

        monkeypatch.setattr(session, "mt5_call", _tracked_lock)

        re_mod.can_trade(**base_kwargs)
        assert lock_entered["count"] == 1


class TestCorrelationProtectionFailsClosed:
    def test_normal_operation_allows_the_trade(self, monkeypatch, base_kwargs):
        re_mod = _reload_risk_engine()
        fake = _FakeMT5(positions=())
        _patch_mt5_session(monkeypatch, fake)

        passed, reason = re_mod.can_trade(**base_kwargs)
        assert passed is True

    def test_a_broken_import_rejects_instead_of_approving(self, monkeypatch, base_kwargs):
        """The exact defect: `except Exception: pass` fell through to
        `return True, "OK"` when the module couldn't even be imported."""
        re_mod = _reload_risk_engine()
        fake = _FakeMT5(positions=())
        _patch_mt5_session(monkeypatch, fake)

        # Force the deferred import inside can_trade to fail.
        blocked = types.ModuleType("execution.risk_management.correlation_protection")
        monkeypatch.setitem(sys.modules, "execution.risk_management.correlation_protection", None)

        passed, reason = re_mod.can_trade(**base_kwargs)
        assert passed is False
        assert "CorrelationProtection" in reason

    def test_a_db_read_failure_rejects_instead_of_approving(self, monkeypatch, base_kwargs):
        """get_open_trades() is a database call; a locked/corrupt DB must not
        silently read as 'no correlated position exists'.

        can_trade reads open trades twice: once in the primary correlation
        filter (step 7, unguarded — an exception there already propagates
        correctly) and once in the correlation-protection module (step 7b,
        the one this test targets). Succeed the first call and fail the
        second so the fix under test is what's actually exercised.
        """
        re_mod = _reload_risk_engine()
        fake = _FakeMT5(positions=())
        _patch_mt5_session(monkeypatch, fake)

        calls = {"n": 0}

        def _flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                return []
            raise RuntimeError("database is locked")

        monkeypatch.setattr(re_mod, "get_open_trades", _flaky)

        passed, reason = re_mod.can_trade(**base_kwargs)
        assert passed is False
        assert "CorrelationProtection" in reason
        assert calls["n"] == 2

    def test_an_actual_correlated_position_still_blocks(self, monkeypatch, base_kwargs):
        """The fail-closed fix must not have broken the check it wraps."""
        re_mod = _reload_risk_engine()
        fake = _FakeMT5(positions=())
        _patch_mt5_session(monkeypatch, fake)

        monkeypatch.setattr(
            re_mod, "get_open_trades",
            lambda: [{"symbol": "GBPUSD", "direction": "BUY"}],
        )
        monkeypatch.setattr(re_mod, "CORRELATED_PAIRS", [("EURUSD", "GBPUSD")], raising=False)

        # correlation_protection reads CORRELATED_PAIRS from config directly,
        # so patch it there too.
        import config as cfg

        monkeypatch.setattr(cfg, "CORRELATED_PAIRS", [("EURUSD", "GBPUSD")])
        monkeypatch.setattr(cfg, "CORRELATION_PROTECTION_ENABLED", True)

        kwargs = dict(base_kwargs)
        kwargs["direction"] = "BUY"
        passed, reason = re_mod.can_trade(**kwargs)
        assert passed is False
        assert "CorrelationProtection" in reason
