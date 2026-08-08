"""M-01: one name, one value, and the value actually reaches the code.

Two separate defects are pinned here.

**The duplicate-definition defect.** ``ML_EXIT_ENABLED``, ``ATR_SL_BASE_MULTIPLIER``,
``ATR_TP_BASE_MULTIPLIER`` and ``MAX_SL_PIPS`` were defined in *both* ``config.py``
and ``trade_management/tm_config.py``. Nothing read the ``config.py`` copies —
every layer resolves them through ``tm_config`` — so the two could disagree
indefinitely and editing the ``config.py`` one silently did nothing. An operator
turning a knob and seeing no effect is worse than a crash: it looks like the
setting has no power, when in fact they edited the wrong file.

Deleting a duplicate is only half a fix. The other half is proving the surviving
definition is load-bearing, so these tests change the authoritative value and
assert the *runtime output* moves. A constant nobody reads would pass a
"it exists" check and fail these.

**The empty-environment-variable defect** (KNOWN_ISSUES #5). ``os.getenv(name,
default)`` returns the default only when the variable is absent. A present-but-
empty ``MT5_LOGIN=`` — the normal result of copying ``.env.example`` — yielded
``int("")`` and a ``ValueError`` at import time, from a module every entry point
imports, before logging existed, with a traceback that did not name the
variable. It took the entire bot down.
"""

from __future__ import annotations

import importlib

import pytest

import config
from trade_management import tm_config as C
from trade_management.layer1_initial_protection import compute_initial_protection
from trade_management.layer6_trade_profile import resolve_settings


# Names that used to exist in both files. tm_config owns them now.
FORMERLY_DUPLICATED = [
    "ML_EXIT_ENABLED",
    "ATR_SL_BASE_MULTIPLIER",
    "ATR_TP_BASE_MULTIPLIER",
    "MAX_SL_PIPS",
    "ATR_SL_MULTIPLIER",
    "ATR_TP_MULTIPLIER",
]


class TestNoShadowDefinitions:
    @pytest.mark.parametrize("name", FORMERLY_DUPLICATED)
    def test_config_no_longer_defines_trade_management_settings(self, name):
        assert not hasattr(config, name), (
            f"config.{name} is back. It is a shadow copy: every consumer reads "
            f"trade_management.tm_config, so this one can drift out of sync "
            f"while looking authoritative."
        )

    @pytest.mark.parametrize(
        "name",
        ["ML_EXIT_ENABLED", "ATR_SL_BASE_MULTIPLIER",
         "ATR_TP_BASE_MULTIPLIER", "MAX_SL_PIPS"],
    )
    def test_tm_config_owns_them(self, name):
        assert hasattr(C, name), f"tm_config.{name} missing — nothing defines it now"

    def test_no_module_imports_them_from_config(self):
        """A leftover `from config import MAX_SL_PIPS` would be an ImportError."""
        import ast
        import os

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        skip = {"__pycache__", ".git", "logs", "models", "models_backup",
                "backups", ".pytest_cache"}
        offenders = []
        for root, dirs, files in os.walk(repo):
            dirs[:] = [d for d in dirs if d not in skip]
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(root, name)
                try:
                    with open(path, encoding="utf-8-sig") as fh:
                        tree = ast.parse(fh.read())
                except (SyntaxError, UnicodeDecodeError):
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module == "config":
                        for alias in node.names:
                            if alias.name in FORMERLY_DUPLICATED:
                                offenders.append(
                                    f"{os.path.relpath(path, repo)}:{node.lineno} "
                                    f"imports {alias.name}"
                                )
        assert not offenders, (
            "these now raise ImportError — read them from "
            "trade_management.tm_config instead:\n  " + "\n  ".join(offenders)
        )


class TestTheSurvivingValueIsLoadBearing:
    """Change the config, prove the runtime output changes."""

    def test_sl_multiplier_changes_the_stop_distance(self, monkeypatch):
        atr = 0.0020

        monkeypatch.setattr(C, "ATR_SL_BASE_MULTIPLIER", 1.0)
        narrow = compute_initial_protection("EURUSD", atr, regime="normal")

        monkeypatch.setattr(C, "ATR_SL_BASE_MULTIPLIER", 2.0)
        wide = compute_initial_protection("EURUSD", atr, regime="normal")

        assert wide.sl_multiplier == pytest.approx(2.0 * narrow.sl_multiplier)
        # Not capped at this ATR, so the distance must move with it.
        assert not narrow.capped and not wide.capped
        assert wide.sl_distance > narrow.sl_distance

    def test_tp_multiplier_changes_the_target_distance(self, monkeypatch):
        atr = 0.0020

        monkeypatch.setattr(C, "ATR_TP_BASE_MULTIPLIER", 2.0)
        near = compute_initial_protection("EURUSD", atr, regime="normal")

        monkeypatch.setattr(C, "ATR_TP_BASE_MULTIPLIER", 4.0)
        far = compute_initial_protection("EURUSD", atr, regime="normal")

        assert far.tp_distance == pytest.approx(2.0 * near.tp_distance)

    def test_max_sl_pips_actually_caps(self, monkeypatch):
        """A huge ATR must be clipped by the pip ceiling, not sail past it."""
        huge_atr = 1.0  # 10,000 pips on EURUSD

        monkeypatch.setattr(C, "MAX_SL_PIPS", 50)
        tight = compute_initial_protection("EURUSD", huge_atr, regime="normal")

        monkeypatch.setattr(C, "MAX_SL_PIPS", 200)
        loose = compute_initial_protection("EURUSD", huge_atr, regime="normal")

        assert tight.capped and loose.capped
        assert loose.sl_distance > tight.sl_distance, (
            "MAX_SL_PIPS did not reach the ceiling calculation"
        )

    def test_ml_exit_enabled_reaches_the_resolved_settings(self, monkeypatch):
        monkeypatch.setattr(C, "ML_EXIT_ENABLED", False)
        assert resolve_settings("trend")["ML_EXIT_ENABLED"] is False

        monkeypatch.setattr(C, "ML_EXIT_ENABLED", True)
        assert resolve_settings("trend")["ML_EXIT_ENABLED"] is True

    def test_profile_override_beats_the_module_default(self, monkeypatch):
        """Layer 6 must win over the module constant — that is its whole job."""
        monkeypatch.setattr(C, "EXIT_SCORE_THRESHOLD", 0.75)
        # "trend" overrides it to 0.80, "mean_reversion" to 0.65.
        assert resolve_settings("trend")["EXIT_SCORE_THRESHOLD"] == 0.80
        assert resolve_settings("mean_reversion")["EXIT_SCORE_THRESHOLD"] == 0.65

    def test_env_override_reaches_the_constant(self, monkeypatch):
        """TM_* environment variables are the documented tuning mechanism."""
        monkeypatch.setenv("TM_ATR_SL_BASE_MULTIPLIER", "3.25")
        reloaded = importlib.reload(C)
        try:
            assert reloaded.ATR_SL_BASE_MULTIPLIER == pytest.approx(3.25)
        finally:
            monkeypatch.delenv("TM_ATR_SL_BASE_MULTIPLIER", raising=False)
            importlib.reload(C)

    def test_reload_restored_the_default(self):
        """Guard against the previous test leaking a value into the suite."""
        assert C.ATR_SL_BASE_MULTIPLIER == pytest.approx(1.5)


class TestEmptyEnvironmentVariables:
    """KNOWN_ISSUES #5: present-but-empty must behave as absent, not crash."""

    @pytest.mark.parametrize("raw", ["", "   ", "\t"])
    def test_empty_int_falls_back_to_default(self, monkeypatch, raw):
        monkeypatch.setenv("KAIROS_TEST_INT", raw)
        assert config._env_int("KAIROS_TEST_INT", 42) == 42

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_empty_float_falls_back_to_default(self, monkeypatch, raw):
        monkeypatch.setenv("KAIROS_TEST_FLOAT", raw)
        assert config._env_float("KAIROS_TEST_FLOAT", 1.5) == pytest.approx(1.5)

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_empty_str_falls_back_to_default(self, monkeypatch, raw):
        monkeypatch.setenv("KAIROS_TEST_STR", raw)
        assert config._env_str("KAIROS_TEST_STR", "fallback") == "fallback"

    def test_absent_falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("KAIROS_TEST_ABSENT", raising=False)
        assert config._env_int("KAIROS_TEST_ABSENT", 7) == 7
        assert config._env_str("KAIROS_TEST_ABSENT", "x") == "x"

    def test_a_real_value_is_used(self, monkeypatch):
        monkeypatch.setenv("KAIROS_TEST_INT", "1234")
        assert config._env_int("KAIROS_TEST_INT", 42) == 1234

    def test_surrounding_whitespace_is_tolerated(self, monkeypatch):
        """`.env` files routinely carry a trailing space; it must not break login."""
        monkeypatch.setenv("KAIROS_TEST_INT", "  1234  ")
        monkeypatch.setenv("KAIROS_TEST_STR", "  server-name  ")
        assert config._env_int("KAIROS_TEST_INT", 0) == 1234
        assert config._env_str("KAIROS_TEST_STR", "") == "server-name"

    def test_malformed_number_names_the_variable(self, monkeypatch):
        """The old failure was a bare ValueError that said nothing useful."""
        monkeypatch.setenv("KAIROS_TEST_INT", "not-a-number")
        with pytest.raises(ValueError, match="KAIROS_TEST_INT"):
            config._env_int("KAIROS_TEST_INT", 42)

    def test_config_imports_with_a_blank_env(self, monkeypatch):
        """The exact scenario that took the bot down."""
        for name in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER",
                     "LOG_RETENTION_DAYS", "BROKER_UTC_OFFSET_HOURS"):
            monkeypatch.setenv(name, "")
        reloaded = importlib.reload(config)
        try:
            assert reloaded.MT5_LOGIN == 0
            assert reloaded.LOG_RETENTION_DAYS == 14
            assert reloaded.BROKER_UTC_OFFSET_HOURS == pytest.approx(3.0)
        finally:
            for name in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER",
                         "LOG_RETENTION_DAYS", "BROKER_UTC_OFFSET_HOURS"):
                monkeypatch.delenv(name, raising=False)
            importlib.reload(config)


class TestIncompleteCredentialsFailClosed:
    """No hardcoded fallback account, and no login attempt with a partial set."""

    SECRET_NAMES = {
        "MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "DEEPSEEK_API_KEY",
        "FINNHUB_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
    }

    def test_no_hardcoded_credentials_in_source(self):
        """Secrets belong in .env; a default here gets committed to git.

        config.py previously shipped a live DeepSeek key, a Telegram bot token
        and an MT5 login/password as literal fallbacks.
        """
        import ast
        import os

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo, "config.py"), encoding="utf-8-sig") as fh:
            tree = ast.parse(fh.read())

        bad = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if not node.func.id.startswith("_env_") or len(node.args) < 2:
                continue
            name_node, default_node = node.args[0], node.args[1]
            if not (isinstance(name_node, ast.Constant)
                    and name_node.value in self.SECRET_NAMES):
                continue
            default = getattr(default_node, "value", "<non-literal>")
            if default not in ("", 0):
                bad.append(f"{name_node.value} defaults to {default!r}")

        assert not bad, (
            "hardcoded credential defaults in config.py:\n  " + "\n  ".join(bad)
        )

    def test_the_credential_scan_actually_inspects_something(self):
        """Sanity check: the AST walk is not vacuously passing."""
        import ast
        import os

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo, "config.py"), encoding="utf-8-sig") as fh:
            tree = ast.parse(fh.read())

        seen = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("_env_")
            and node.args
            and isinstance(node.args[0], ast.Constant)
        }
        assert self.SECRET_NAMES <= seen, (
            f"credential reads not found by the scan: {self.SECRET_NAMES - seen}"
        )

    @pytest.mark.parametrize(
        "login,password,server,expected",
        [
            (0, "pw", "srv", "MT5_LOGIN"),
            (123, "", "srv", "MT5_PASSWORD"),
            (123, "pw", "", "MT5_SERVER"),
        ],
    )
    def test_session_refuses_incomplete_credentials(
        self, monkeypatch, login, password, server, expected, caplog
    ):
        import data.market.mt5_session as session

        monkeypatch.setattr(session, "MT5_LOGIN", login)
        monkeypatch.setattr(session, "MT5_PASSWORD", password)
        monkeypatch.setattr(session, "MT5_SERVER", server)
        monkeypatch.setattr(session, "is_available", lambda: True)

        # If the guard fails to stop us, this makes the test fail loudly
        # rather than silently "passing" on a mocked login.
        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("login attempted with incomplete credentials")

        if session.mt5 is not None:
            monkeypatch.setattr(session.mt5, "login", _must_not_be_called)

        session._reset_state_for_tests()
        with caplog.at_level("ERROR"):
            assert session.ensure_session() is False
        assert expected in caplog.text

    def test_complete_credentials_get_past_the_guard(self, monkeypatch):
        """The guard must not block a properly configured deployment."""
        import data.market.mt5_session as session

        monkeypatch.setattr(session, "MT5_LOGIN", 123456)
        monkeypatch.setattr(session, "MT5_PASSWORD", "pw")
        monkeypatch.setattr(session, "MT5_SERVER", "srv")
        monkeypatch.setattr(session, "is_available", lambda: True)

        reached = {"initialize": False}

        class _FakeMT5:
            @staticmethod
            def terminal_info():
                reached["initialize"] = True
                return None

            @staticmethod
            def initialize(*args, **kwargs):
                return False

            @staticmethod
            def last_error():
                return (0, "test")

        monkeypatch.setattr(session, "mt5", _FakeMT5)
        session._reset_state_for_tests()

        session.ensure_session()
        assert reached["initialize"], "the credential guard blocked a valid config"
