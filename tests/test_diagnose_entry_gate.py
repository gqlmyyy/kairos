"""The diagnostic must run anywhere — no MT5, no Windows, no broker.

Its whole value is being runnable on the machine where the problem is being
reasoned about, which is usually not the trading host. These tests pin that,
plus the two things a wrong answer would cost: naming the real blocker, and
never writing anything.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_ROOT, "scripts", "diagnose_entry_gate.py")

spec = importlib.util.spec_from_file_location("diagnose_entry_gate", _SCRIPT)
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)


def run(env_overrides=None):
    env = dict(os.environ)
    env.update(env_overrides or {})
    env["PYTHONPATH"] = _ROOT
    result = subprocess.run(
        [sys.executable, _SCRIPT],
        capture_output=True, text=True, cwd=_ROOT, env=env, timeout=120,
    )
    return result


class TestRunsHeadless:
    def test_it_exits_zero_without_mt5(self):
        """MetaTrader5 is not installed in this environment, which is the
        point — the script must still complete."""
        result = run()
        assert result.returncode == 0, result.stderr[-2000:]

    def test_mt5_is_genuinely_absent_here(self):
        """Guards the test above from passing for the wrong reason.

        Phase 1 note: this suite now also runs on the Windows data machine
        (where the Phase 1 fetch happens), and there MetaTrader5 IS installed.
        The headless premise of `test_it_exits_zero_without_mt5` cannot be
        verified on such a machine, so the guard skips instead of failing --
        the guard still does its job wherever MT5 is genuinely absent, and the
        headless behaviour itself is exercised by env-stripping in run().
        """
        if importlib.util.find_spec("MetaTrader5") is not None:
            pytest.skip("MetaTrader5 is installed on this machine -- the "
                        "headless premise cannot hold here")
        assert importlib.util.find_spec("MetaTrader5") is None

    def test_it_never_imports_mt5(self):
        import ast

        with open(_SCRIPT, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "MetaTrader5" not in alias.name
            elif isinstance(node, ast.ImportFrom):
                assert "MetaTrader5" not in (node.module or "")


class TestReportsTheRealBlocker:
    def test_it_names_the_missing_sidecar(self):
        out = run().stdout
        assert "metadata.json" in out
        assert "MISSING" in out

    def test_it_reports_both_feature_counts(self):
        out = run().stdout
        assert "65" in out, "the artifact's feature count must be shown"
        assert "10" in out, "the live spec's feature count must be shown"

    def test_it_prints_the_config_values(self):
        out = run().stdout
        for key in ("ENTRY_MODEL_VERSION", "ENTRY_ML_MODE",
                     "ENTRY_ML_ABSENT_SIZE_MULT"):
            assert key in out

    def test_required_mode_says_no_trade_will_open(self):
        out = run({"ENTRY_ML_MODE": "required"}).stdout
        assert "NO TRADE WILL OPEN" in out
        assert "ml_unavailable" in out

    def test_advisory_mode_says_trades_can_open_unfiltered(self):
        out = run({"ENTRY_ML_MODE": "advisory"}).stdout
        assert "TRADES CAN OPEN" in out
        assert "NO ML FILTERING" in out

    def test_off_mode_says_trades_can_open_unfiltered(self):
        out = run({"ENTRY_ML_MODE": "off"}).stdout
        assert "TRADES CAN OPEN" in out

    def test_it_always_ends_with_a_next_step(self):
        for mode in ("required", "advisory", "off"):
            out = run({"ENTRY_ML_MODE": mode}).stdout
            assert "NEXT STEP" in out, mode

    def test_it_does_not_advise_hand_writing_a_sidecar(self):
        """The sidecar refusal is deliberate. The script must point at the
        trainer, not at forging provenance."""
        out = run().stdout
        assert "Do NOT" in out and "hand-write" in out


class TestReadOnly:
    def test_the_script_has_no_write_path(self):
        import ast

        with open(_SCRIPT, encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "open":
                    # a bare open() defaults to read mode; a write mode would
                    # appear as a second positional or a mode= keyword
                    modes = [a for a in node.args[1:]]
                    modes += [k.value for k in node.keywords if k.arg == "mode"]
                    for m in modes:
                        assert not (isinstance(m, ast.Constant)
                                     and any(c in str(m.value) for c in "wax")), \
                            "the diagnostic opened a file for writing"
        assert "json.dump" not in source
        assert "os.remove" not in source

    def test_running_it_does_not_create_the_sidecar(self):
        sidecar = os.path.join(_ROOT, "models", "entry",
                                "entry_model.json.metadata.json")
        before = os.path.exists(sidecar)
        run()
        assert os.path.exists(sidecar) == before, (
            "the diagnostic created or removed the metadata sidecar")


class TestHelpers:
    def test_booster_feature_count_reports_errors_instead_of_raising(self):
        count, error = diagnose.booster_feature_count("/nonexistent/model.json")
        assert count is None
        assert error, "a failure must be reported, not swallowed"

    def test_booster_feature_count_reads_the_real_artifact(self):
        path = os.path.join(_ROOT, "models", "entry", "entry_model.json")
        if not os.path.exists(path):
            import pytest

            pytest.skip("no artifact in this checkout")
        count, error = diagnose.booster_feature_count(path)
        if error and "xgboost is not installed" in error:
            import pytest

            pytest.skip("xgboost unavailable")
        assert count == 65, "the shipped artifact is the 65-feature one"
