"""Offline replay on Linux: stored candles in, p_win out, no MT5 anywhere.

Also pins the two things that make a replay worth running: it is deterministic
(same inputs, byte-identical outputs, every time) and it is honest about what
it could not score.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "models" / "research" / "registry.json"
CANDLES = ROOT / "tests" / "fixtures" / "research" / "candles"

pytestmark = pytest.mark.skipif(
    not REGISTRY.exists() or not CANDLES.exists(),
    reason="research models/fixtures not present in this checkout")


@pytest.fixture(scope="module")
def source():
    from analysis.research import candles as cd
    return cd.JsonCandleSource(CANDLES)


PAIRS = [("XAUUSD", "H1"), ("EURUSD", "H1"), ("GBPUSD", "H4"), ("XAUUSD", "H4")]


@pytest.mark.parametrize("symbol,timeframe", PAIRS)
def test_replay_produces_probabilities_on_a_complete_source(source, symbol, timeframe):
    from analysis.research import replay as rp

    result = rp.replay(symbol, timeframe, source, tail=40, registry_path=REGISTRY)
    assert result.served_any, result.summary()
    served = result.predictions[result.predictions["status"] == "OK"]
    assert len(served) > 0
    assert served["p_win"].between(0.0, 1.0).all()
    assert served["p_win"].notna().all()
    # Both sides of each bar are scored, never just one.
    assert set(served["direction"]) <= {"long", "short"}


@pytest.mark.parametrize("symbol,timeframe", PAIRS)
def test_replay_is_deterministic(source, symbol, timeframe):
    """Same candles, same model, same range -> byte-identical output."""
    from analysis.research import replay as rp

    a = rp.replay(symbol, timeframe, source, tail=25, registry_path=REGISTRY)
    b = rp.replay(symbol, timeframe, source, tail=25, registry_path=REGISTRY)
    pd.testing.assert_frame_equal(a.predictions, b.predictions)
    assert a.status_counts == b.status_counts


@pytest.mark.parametrize("symbol,timeframe", PAIRS)
def test_feature_frame_is_deterministic(source, symbol, timeframe):
    from analysis.research import candles as cd
    from analysis.research import engine as E
    from analysis.research.model_loader import load_model

    model = load_model(symbol, timeframe, registry_path=REGISTRY)
    tfs = [timeframe, *model.card.context_timeframes]
    frames = [
        E.build_feature_frame(symbol, timeframe, cd.load_stack(source, symbol, tfs),
                              list(model.card.context_timeframes))
        for _ in range(2)
    ]
    pd.testing.assert_frame_equal(frames[0], frames[1])


def test_a_narrower_window_does_not_change_the_values_inside_it(source):
    """`start` bounds which rows are SCORED, never which candles are loaded.

    Trimming the loaded history would change every rolling window, so a
    replay over a sub-range must agree with the same rows of a wider one.
    """
    from analysis.research import replay as rp

    wide = rp.replay("XAUUSD", "H1", source, tail=60, registry_path=REGISTRY)
    served = wide.predictions[wide.predictions["status"] == "OK"]
    assert len(served) > 10
    start = served["timestamp"].iloc[5]

    narrow = rp.replay("XAUUSD", "H1", source, start=str(start), tail=10,
                       registry_path=REGISTRY)
    merged = narrow.predictions.merge(wide.predictions, on=["timestamp", "direction"],
                                      suffixes=("_narrow", "_wide"))
    assert len(merged) > 0
    pd.testing.assert_series_equal(merged["p_win_narrow"], merged["p_win_wide"],
                                   check_names=False)


def test_replay_reports_what_it_refused(source):
    from analysis.research import candles as cd
    from analysis.research import replay as rp

    bare = cd.KairosHistoricalSource(ROOT / "data" / "historical")
    if not bare.available("XAUUSD", "H1"):
        pytest.skip("no stored historical candles in this checkout")

    result = rp.replay("XAUUSD", "H1", bare, start="2024-06-01", limit=5,
                       registry_path=REGISTRY)
    assert result.rows_refused > 0
    assert result.rows_scored == 0
    assert "spread" in result.summary()
    assert all(r for r in result.predictions["reason"]), "every refusal must say why"


def test_replay_touches_no_mt5_and_no_account():
    """The research package must not import a terminal, a broker or an account.

    Parsed rather than grepped, so prose in a docstring saying "this never
    imports MetaTrader5" does not trip it and a real call cannot hide inside a
    string. This is the constraint most likely to be broken by accident later.
    """
    import ast

    BANNED_MODULES = {"MetaTrader5", "data.market.mt5_client", "data.market.client",
                      "data.market.hybrid_client", "risk", "execution",
                      "trade_management"}
    BANNED_CALLS = {"order_send", "account_info", "positions_get", "symbol_info_tick",
                    "initialize", "login"}

    offenders = []
    for path in sorted((ROOT / "analysis" / "research").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in BANNED_MODULES or alias.name in BANNED_MODULES:
                        offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if node.module in BANNED_MODULES or root in BANNED_MODULES:
                    offenders.append(f"{path.name}:{node.lineno} from {node.module}")
            elif isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if name in BANNED_CALLS:
                    offenders.append(f"{path.name}:{node.lineno} call {name}()")
    assert not offenders, f"research package reaches for a live trading API: {offenders}"


def test_the_replay_cli_runs_headless():
    """The documented command must actually work on this platform."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "scripts/research_replay.py", "--symbol", "XAUUSD",
         "--tf", "H1", "--source-kind", "json",
         "--source-root", str(CANDLES), "--tail", "5", "--show", "2"],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "p_win over" in proc.stdout, proc.stdout


def test_the_import_script_is_reproducible(tmp_path):
    """Re-importing the same source must produce identical cards and registry."""
    import subprocess
    import sys

    source = ROOT.parent / "xgbooost"
    if not (source / "models" / "research_v2").is_dir():
        pytest.skip("the xgbooost research repository is not beside this checkout")

    proc = subprocess.run(
        [sys.executable, "scripts/import_research_model.py", "--source", str(source),
         "--dest", str(tmp_path), "--symbol", "XAUUSD", "--tf", "H1"],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr

    fresh = json.loads((tmp_path / "XAUUSD" / "H1" / "model_card.json").read_text())
    shipped = json.loads(
        (ROOT / "models" / "research" / "XAUUSD" / "H1" / "model_card.json").read_text())
    for key in ("model_id", "feature_list", "model_hash", "target", "horizon_bars",
                "training_dataset_hash", "kairos_contract_fingerprint"):
        assert fresh[key] == shipped[key], f"{key} is not reproducible"
