"""Guard against config constants being removed while still imported.

This exists because of a real regression: the trade-management rewrite deleted
the "Equity Guard" config section as part of removing a dead subsystem, but that
section also defined ``MAX_CONSECUTIVE_LOSSES``, which ``risk/risk_governor.py``
imports. The governor's import then raised ImportError on every call, and since
``main.py`` wraps the lookup in a bare ``except``, risk halting was silently
disabled for hours of live trading with no log line to show for it.

A missing constant should fail here, loudly, at test time — not in production
inside an exception handler.
"""

from __future__ import annotations

import ast
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {"__pycache__", ".git", "models_backup", "logs", "models", "backups", ".pytest_cache"}


def _iter_python_files():
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(root, name)


def _config_imports():
    """Every ``from config import X`` across the repo, as (file, name) pairs."""
    found = []
    for path in _iter_python_files():
        try:
            with open(path, encoding="utf-8-sig") as fh:
                tree = ast.parse(fh.read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "config":
                for alias in node.names:
                    if alias.name != "*":
                        found.append((os.path.relpath(path, REPO_ROOT), alias.name))
    return found


class TestConfigIntegrity:
    def test_every_imported_constant_exists(self):
        import config

        missing = [
            (path, name) for path, name in _config_imports()
            if not hasattr(config, name)
        ]
        assert not missing, (
            "config constants imported but not defined:\n"
            + "\n".join(f"  {path}: {name}" for path, name in missing)
        )

    def test_risk_governor_imports_cleanly(self):
        """The exact failure that went unnoticed in production."""
        from risk.risk_governor import get_risk_governor

        governor = get_risk_governor()
        assert governor is not None
        assert isinstance(governor.is_halted(), bool)

    def test_sweep_actually_finds_imports(self):
        """Sanity check: the AST walk is not silently returning nothing."""
        assert len(_config_imports()) > 10

    @pytest.mark.parametrize(
        "name",
        [
            "MAX_CONSECUTIVE_LOSSES",
            "RISK_GOVERNOR_MAX_LOSS_R",
            "RISK_GOVERNOR_PERSIST",
            "MAX_OPEN_TRADES",
            "SYMBOLS",
            "DB_FILE",
            "MT5_LOGIN",
        ],
    )
    def test_critical_constants_present(self, name):
        import config

        assert hasattr(config, name), f"config.{name} is missing"

    def test_quantdinger_settings_are_gone(self):
        """QuantDinger was removed; its settings must not linger."""
        import config

        leftovers = [n for n in dir(config) if "QUANTDINGER" in n.upper()]
        assert not leftovers, f"QuantDinger config leftovers: {leftovers}"
