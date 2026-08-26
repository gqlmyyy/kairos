"""Golden parity: KAIROS must reproduce the research repo's own numbers.

Each fixture in ``tests/fixtures/research/golden/`` was produced by the
xgbooost research repository running ITS engine and ITS shipped model over the
shared candle fixture in ``tests/fixtures/research/candles/``. This test feeds
KAIROS the identical file and requires the identical answer — every feature
value and the probability.

This matters more than "the code runs". A re-implementation that is subtly
different (Wilder vs simple smoothing, ddof=0 vs ddof=1, a timezone, a
zero-denominator convention) produces plausible vectors that mean something
other than what the model was trained on, and no amount of downstream testing
catches it.

What the fixtures are NOT
-------------------------
The fixture ``spread`` column is synthetic and the M15 stack is a random walk
(KAIROS stores no M15 candles). These files prove two implementations agree.
They are not evidence about markets, and the research repo's own verdict on
every shipped model — NO_SIGNAL or DROP — is carried in each golden file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "fixtures" / "research" / "golden"
CANDLES = ROOT / "tests" / "fixtures" / "research" / "candles"
REGISTRY = ROOT / "models" / "research" / "registry.json"

#: Both sides run the same library versions, so agreement is exact in practice.
#: A tolerance is still stated rather than asserting equality, because a future
#: BLAS or libm change could move the last bits without anything being wrong.
FEATURE_TOLERANCE = 1e-12
PROBABILITY_TOLERANCE = 1e-12

pytestmark = pytest.mark.skipif(
    not REGISTRY.exists() or not GOLDEN.exists(),
    reason="research models/golden fixtures not present in this checkout")


def _golden_files():
    return sorted(GOLDEN.glob("golden_*.json")) if GOLDEN.exists() else []


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source():
    from analysis.research import candles as cd
    return cd.JsonCandleSource(CANDLES)


def _recompute(golden: dict, source) -> pd.DataFrame:
    from analysis.research import candles as cd
    from analysis.research import engine as E

    tfs = [golden["entry_timeframe"], *golden["context_timeframes"]]
    stack = cd.load_stack(source, golden["fixture_symbol"], tfs)
    frame = E.build_feature_frame(golden["fixture_symbol"], golden["entry_timeframe"],
                                  stack, golden["context_timeframes"])
    return frame.set_index("timestamp")


@pytest.mark.parametrize("path", _golden_files(), ids=lambda p: p.stem)
def test_feature_values_match_the_research_engine(path, source):
    golden = _load(path)
    frame = _recompute(golden, source)
    assert golden["samples"], f"{path.name} carries no samples"

    worst_name, worst_delta = None, 0.0
    for sample in golden["samples"]:
        row = frame.loc[pd.Timestamp(sample["timestamp"])]
        for name, expected in sample["features"].items():
            if name == "entry_direction":
                continue
            delta = abs(float(row[name]) - float(expected))
            if delta > worst_delta:
                worst_name, worst_delta = name, delta
    assert worst_delta <= FEATURE_TOLERANCE, (
        f"{path.name}: {worst_name} differs from the research engine by "
        f"{worst_delta:.3e} (tolerance {FEATURE_TOLERANCE:.0e})")


@pytest.mark.parametrize("path", _golden_files(), ids=lambda p: p.stem)
def test_p_win_matches_the_research_model(path, source):
    from analysis.research import inference as inf
    from analysis.research.model_loader import load_model

    golden = _load(path)
    model = load_model(golden["symbol"], golden["entry_timeframe"])
    frame = _recompute(golden, source)

    checked = 0
    for sample in golden["samples"]:
        ts = pd.Timestamp(sample["timestamp"])
        pred = inf.predict_row(model, frame.loc[ts].to_dict(),
                               entry_direction=sample["direction"], timestamp=ts)
        assert pred.status == "OK", f"{path.name} @ {ts}: {pred.status} — {pred.reason}"
        assert abs(pred.p_win - sample["p_win"]) <= PROBABILITY_TOLERANCE, (
            f"{path.name} @ {ts} {sample['direction']}: p_win {pred.p_win} vs "
            f"golden {sample['p_win']}")
        assert abs(pred.raw_probability - sample["raw_probability"]) <= PROBABILITY_TOLERANCE
        checked += 1
    assert checked == len(golden["samples"])


@pytest.mark.parametrize("path", _golden_files(), ids=lambda p: p.stem)
def test_feature_order_matches_the_shipped_model(path):
    from analysis.research.model_loader import load_model

    golden = _load(path)
    model = load_model(golden["symbol"], golden["entry_timeframe"])
    assert list(model.feature_names) == golden["feature_order"]


@pytest.mark.parametrize("path", _golden_files(), ids=lambda p: p.stem)
def test_both_directions_are_scored_and_differ_where_the_model_uses_direction(path):
    """A model carrying `entry_direction` must not return one number for both sides.

    The legacy 65-vs-10 defect showed up exactly here: BUY and SELL received
    identical probabilities because `direction` never reached the model.
    """
    golden = _load(path)
    by_ts = {}
    for s in golden["samples"]:
        by_ts.setdefault(s["timestamp"], {})[s["direction"]] = s["p_win"]
    pairs = [v for v in by_ts.values() if len(v) == 2]
    assert pairs, f"{path.name} has no long/short pair"

    if "entry_direction" in golden["feature_order"]:
        assert any(v["long"] != v["short"] for v in pairs), (
            f"{path.name} carries entry_direction but every long/short pair is "
            f"identical — the direction is not reaching the model")
    else:
        # A `_nodir` feature set genuinely has no direction input, so identical
        # probabilities are correct rather than a bug. Pinned so the two cases
        # can never be confused.
        assert all(v["long"] == v["short"] for v in pairs), (
            f"{path.name} has no entry_direction feature, so long and short must "
            f"score identically")


def test_golden_fixtures_carry_their_provenance_and_verdict():
    """A golden file must state what it is, so it cannot be read as a market result."""
    for path in _golden_files():
        g = _load(path)
        assert g["fixture_ohlc"] in ("REAL", "SYNTHETIC")
        assert "not_a_market_claim" in g
        assert g["model_verdict"] in ("NO_SIGNAL", "DROP", "PASS", "REJECTED")


def test_every_shipped_model_has_a_golden_fixture():
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    covered = {(g["symbol"], g["entry_timeframe"])
               for g in (_load(p) for p in _golden_files())}
    for m in registry["models"]:
        assert (m["symbol"], m["timeframe"]) in covered, (
            f"{m['model_id']} has no golden parity fixture")
