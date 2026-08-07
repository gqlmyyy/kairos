"""Tests for the shared MT5 session.

These verify the three properties the module exists to guarantee:
  - the expensive login happens once, not per call
  - concurrent access is serialised (the IPC contention fix)
  - a dropped session is detected and rebuilt
"""

from __future__ import annotations

import threading

import pytest

from data.market import mt5_session


class FakeAccount:
    def __init__(self, balance=1000.0, equity=1000.0, margin=0.0):
        self.balance = balance
        self.equity = equity
        self.margin = margin
        self.margin_free = equity - margin
        self.login = 12345
        self.server = "Test-Server"


class FakeMT5:
    """Minimal stand-in that counts the calls we care about."""

    def __init__(self, *, healthy=True):
        self.initialize_calls = 0
        self.login_calls = 0
        self.account_info_calls = 0
        self._healthy = healthy
        self._terminal_up = False

    def terminal_info(self):
        return object() if self._terminal_up else None

    def initialize(self, path=None):
        self.initialize_calls += 1
        self._terminal_up = True
        return True

    def login(self, login, password=None, server=None):
        self.login_calls += 1
        return self._healthy

    def account_info(self):
        self.account_info_calls += 1
        return FakeAccount() if self._healthy else None

    def last_error(self):
        return (0, "ok")

    def shutdown(self):
        self._terminal_up = False

    def symbol_info(self, symbol):
        class Info:
            visible = True
        return Info()

    def symbol_select(self, symbol, enable):
        return True

    def drop(self):
        """Simulate the terminal dropping the session."""
        self._healthy = False


@pytest.fixture
def fake_mt5(monkeypatch):
    fake = FakeMT5()
    monkeypatch.setattr(mt5_session, "mt5", fake)
    monkeypatch.setattr(mt5_session, "MT5_AVAILABLE", True)
    mt5_session._reset_state_for_tests()
    yield fake
    mt5_session._reset_state_for_tests()


class TestSessionEstablishment:
    def test_first_call_initialises_and_logs_in(self, fake_mt5):
        assert mt5_session.ensure_session() is True
        assert fake_mt5.initialize_calls == 1
        assert fake_mt5.login_calls == 1

    def test_repeated_calls_do_not_re_login(self, fake_mt5):
        """The whole point: login is a broker round trip, account_info is not."""
        for _ in range(20):
            assert mt5_session.ensure_session() is True
        assert fake_mt5.login_calls == 1
        assert fake_mt5.initialize_calls == 1
        # Subsequent calls verify cheaply instead.
        assert fake_mt5.account_info_calls >= 19

    def test_force_relogin_authenticates_again(self, fake_mt5):
        mt5_session.ensure_session()
        mt5_session.ensure_session(force_relogin=True)
        assert fake_mt5.login_calls == 2

    # --- edge cases ---
    def test_unavailable_library_fails_cleanly(self, monkeypatch):
        monkeypatch.setattr(mt5_session, "mt5", None)
        monkeypatch.setattr(mt5_session, "MT5_AVAILABLE", False)
        mt5_session._reset_state_for_tests()
        assert mt5_session.ensure_session() is False
        assert mt5_session.is_healthy() is False

    def test_failed_login_reports_failure(self, monkeypatch):
        fake = FakeMT5(healthy=False)
        monkeypatch.setattr(mt5_session, "mt5", fake)
        monkeypatch.setattr(mt5_session, "MT5_AVAILABLE", True)
        mt5_session._reset_state_for_tests()
        assert mt5_session.ensure_session() is False


class TestRecovery:
    def test_dropped_session_is_detected(self, fake_mt5):
        assert mt5_session.ensure_session() is True
        fake_mt5.drop()
        assert mt5_session.is_healthy() is False

    def test_dropped_session_is_rebuilt_on_next_ensure(self, fake_mt5):
        mt5_session.ensure_session()
        logins_before = fake_mt5.login_calls
        fake_mt5.drop()
        mt5_session.ensure_session()  # detects staleness, tries to rebuild
        assert fake_mt5.login_calls > logins_before


class TestConcurrency:
    def test_lock_serialises_access(self, fake_mt5):
        """Two threads must not interleave inside mt5_call()."""
        mt5_session.ensure_session()
        overlaps = []
        inside = threading.Event()

        def worker(index):
            with mt5_session.mt5_call():
                if inside.is_set():
                    overlaps.append(index)
                inside.set()
                # Give the other thread a chance to interleave if unlocked.
                for _ in range(1000):
                    pass
                inside.clear()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert overlaps == [], "threads interleaved inside the MT5 lock"

    def test_lock_is_reentrant(self, fake_mt5):
        """A locked helper may call another locked helper without deadlocking."""
        mt5_session.ensure_session()
        with mt5_session.mt5_call():
            with mt5_session.mt5_call():
                assert mt5_session.is_healthy() is True


class TestAccountAccess:
    def test_get_account_info_returns_account(self, fake_mt5):
        account = mt5_session.get_account_info()
        assert account is not None
        assert account.equity == 1000.0

    def test_get_account_info_none_when_unhealthy(self, monkeypatch):
        fake = FakeMT5(healthy=False)
        monkeypatch.setattr(mt5_session, "mt5", fake)
        monkeypatch.setattr(mt5_session, "MT5_AVAILABLE", True)
        mt5_session._reset_state_for_tests()
        assert mt5_session.get_account_info() is None
