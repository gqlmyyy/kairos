"""H-01 guard: MT5 session ownership must not be bypassed.

Two rules, enforced statically over the source tree:

1. **Only the session module may create or destroy the session.**
   ``mt5.initialize()``, ``mt5.login()`` and ``mt5.shutdown()`` anywhere else
   fight the owner. ``reconciliation.py`` used to call ``mt5.shutdown()``
   followed by ``mt5.initialize()`` while three other threads were mid-call,
   manufacturing the IPC failures it was trying to repair.

2. **Hot concurrent paths must take the lock.** The post-entry loop runs every
   5 seconds beside the main cycle, reconciliation and the watchdog.

The remaining unwrapped call sites in ``reconciliation.py`` and
``mt5_direct.py`` are recorded as a known allowance below rather than silently
passing — the count is pinned so it cannot grow.
"""

from __future__ import annotations

import ast
import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {"__pycache__", ".git", "models_backup", "logs", "models",
             "backups", ".pytest_cache", "tests", "scripts"}

# Only this module may own the session lifecycle.
SESSION_OWNER = os.path.join("data", "market", "mt5_session.py")

LIFECYCLE_CALLS = ("initialize", "login", "shutdown")

# Files still holding direct calls, with the count at the time of remediation.
# This is a ratchet: the numbers may go down, never up.
KNOWN_DIRECT_CALL_BUDGET = {
    os.path.join("execution", "reconciliation.py"): 25,
    os.path.join("execution", "mt5_direct.py"): 20,
    os.path.join("execution", "post_entry", "action_executor.py"): 19,
    os.path.join("main.py"): 5,
    os.path.join("risk", "symbol_info.py"): 2,
    os.path.join("risk", "risk_engine.py"): 2,
    os.path.join("data", "market", "candle_boundary.py"): 6,
    os.path.join("data", "market", "mt5_client.py"): 10,
    os.path.join("execution", "post_entry", "post_entry_manager.py"): 1,
    os.path.join("execution", "post_entry", "trade_monitor.py"): 1,
}


def _python_files():
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(root, name)
                yield os.path.relpath(path, REPO_ROOT), path


def _read(path):
    with open(path, encoding="utf-8-sig") as fh:
        return fh.read()


def _direct_mt5_calls(source: str):
    """Every `mt5.<something>(` occurrence, excluding comments/docstrings."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "mt5"
        ):
            found.append((node.func.attr, getattr(node, "lineno", 0)))
    return found


class TestSessionLifecycleOwnership:
    def test_only_the_session_module_creates_or_destroys_the_session(self):
        offenders = []
        for rel, path in _python_files():
            if rel == SESSION_OWNER:
                continue
            for attr, line in _direct_mt5_calls(_read(path)):
                if attr in LIFECYCLE_CALLS:
                    offenders.append(f"{rel}:{line} mt5.{attr}()")

        assert not offenders, (
            "session lifecycle called outside data/market/mt5_session.py — "
            "these compete with the shared session:\n  " + "\n  ".join(offenders)
        )

    def test_reconciliation_no_longer_shuts_the_session_down(self):
        """The specific call that killed other threads' connections."""
        source = _read(os.path.join(REPO_ROOT, "execution", "reconciliation.py"))
        calls = [a for a, _ in _direct_mt5_calls(source)]
        assert "shutdown" not in calls
        assert "initialize" not in calls

    def test_action_executor_no_longer_relogs_in(self):
        source = _read(
            os.path.join(REPO_ROOT, "execution", "post_entry", "action_executor.py")
        )
        calls = [a for a, _ in _direct_mt5_calls(source)]
        assert "login" not in calls
        assert "initialize" not in calls

    def test_candle_boundary_delegates_initialisation(self):
        source = _read(os.path.join(REPO_ROOT, "data", "market", "candle_boundary.py"))
        calls = [a for a, _ in _direct_mt5_calls(source)]
        assert "initialize" not in calls


class TestHotLoopLocking:
    """The 5-second loop must not race the other threads."""

    @pytest.mark.parametrize(
        "relpath,function",
        [
            (os.path.join("execution", "post_entry", "trade_monitor.py"), "get_open_positions"),
            (os.path.join("data", "market", "mt5_client.py"), "get_candles"),
        ],
    )
    def test_hot_path_takes_the_lock(self, relpath, function):
        source = _read(os.path.join(REPO_ROOT, relpath))
        tree = ast.parse(source)
        target = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == function),
            None,
        )
        assert target is not None, f"{function} not found in {relpath}"
        body = ast.get_source_segment(source, target) or ""
        assert "mt5_call()" in body, f"{relpath}:{function} does not take the session lock"


class TestDirectCallRatchet:
    """Bypasses may shrink but must never grow."""

    def test_no_new_files_gain_direct_calls(self):
        unexpected = []
        for rel, path in _python_files():
            if rel == SESSION_OWNER:
                continue
            calls = _direct_mt5_calls(_read(path))
            if calls and rel not in KNOWN_DIRECT_CALL_BUDGET:
                unexpected.append(f"{rel} ({len(calls)} calls)")

        assert not unexpected, (
            "new files making direct MT5 calls — route them through "
            "data.market.mt5_session.mt5_call():\n  " + "\n  ".join(unexpected)
        )

    def test_known_files_do_not_exceed_their_budget(self):
        grew = []
        for rel, budget in KNOWN_DIRECT_CALL_BUDGET.items():
            path = os.path.join(REPO_ROOT, rel)
            if not os.path.exists(path):
                continue
            count = len(_direct_mt5_calls(_read(path)))
            if count > budget:
                grew.append(f"{rel}: {count} > budget {budget}")

        assert not grew, "direct MT5 call count increased:\n  " + "\n  ".join(grew)


class TestDatabaseConcurrency:
    """M-02: bounded wait instead of an instant lock error."""

    def test_busy_timeout_is_configured(self):
        import data.storage.database as db

        assert db.DB_BUSY_TIMEOUT_SEC > 0

    def test_connection_applies_busy_timeout(self, tmp_path, monkeypatch):
        import config
        import data.storage.database as db

        path = str(tmp_path / "t.db")
        monkeypatch.setattr(config, "DB_FILE", path)
        monkeypatch.setattr(db, "DB_FILE", path)

        conn = db.get_conn()
        try:
            timeout_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert timeout_ms >= 1000
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        finally:
            conn.close()

    def test_concurrent_writers_all_succeed(self, tmp_path, monkeypatch):
        import threading

        import config
        import data.storage.database as db

        path = str(tmp_path / "concurrent.db")
        monkeypatch.setattr(config, "DB_FILE", path)
        monkeypatch.setattr(db, "DB_FILE", path)
        db.init_db()

        errors = []

        def writer(index):
            try:
                for i in range(10):
                    db.save_decision(
                        f"SYM{index}", "BUY",
                        {"final": 50, "ai": 50, "trend": 50, "momentum": 50,
                         "sentiment": 50, "volatility": 50},
                        0.7, 0.7, True, "TRENDING", "test", "DECIDED",
                    )
            except Exception as exc:
                errors.append(f"writer {index}: {type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "concurrent writes failed:\n  " + "\n  ".join(errors)

        conn = db.get_conn()
        try:
            count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        finally:
            conn.close()
        assert count == 50, f"expected 50 rows, found {count}"
