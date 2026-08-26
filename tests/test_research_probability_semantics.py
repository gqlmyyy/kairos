"""What `p_win` means, and what the shipped models can actually emit.

Two different jobs in one file, both about not misreading a number:

1. `p_win` is P(TP before SL | direction). It is not P(price rises), not an
   expected return, not a confidence. The card declares it and the loader
   enforces it; these tests pin that the declaration is load-bearing.

2. The shipped calibrators are coarse isotonic step functions whose output
   NEVER reaches 0.5. A downstream gate written against the legacy path's
   0.60 threshold would therefore never fire — not because of a bug in the
   integration, but because these models have no edge to express. That is
   worth failing a test over if it silently changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analysis.research import model_card as mc

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "models" / "research" / "registry.json"

pytestmark = pytest.mark.skipif(not REGISTRY.exists(),
                                reason="research models not imported in this checkout")


def _models():
    raw = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return [(m["symbol"], m["timeframe"]) for m in raw["models"]]


IDS = [f"{s}-{t}" for s, t in _models()]


@pytest.mark.parametrize("symbol,timeframe", _models(), ids=IDS)
def test_the_card_states_the_target_and_its_barriers(symbol, timeframe):
    from analysis.research.model_loader import load_model

    card = load_model(symbol, timeframe, registry_path=REGISTRY).card
    assert card.probability_semantics == mc.PROBABILITY_SEMANTICS
    assert "TP" in card.target and "SL" in card.target
    assert card.tp_atr_multiple == 2.5 and card.sl_atr_multiple == 1.5
    assert card.horizon_bars > 0
    # The horizon is measured in bars of the model's OWN timeframe.
    assert card.horizon_bars == {"M15": 96, "M30": 48, "H1": 72, "H4": 60}[timeframe]


@pytest.mark.parametrize("symbol,timeframe", _models(), ids=IDS)
def test_p_win_is_labelled_p_win_all_the_way_through(symbol, timeframe):
    from analysis.research.inference import Prediction

    pred = Prediction(status="OK", p_win=0.4, symbol=symbol, timeframe=timeframe)
    assert pred.as_dict()["semantics"] == mc.PROBABILITY_SEMANTICS
    assert "p_win" in pred.as_dict()


@pytest.mark.parametrize("symbol,timeframe", _models(), ids=IDS)
def test_shipped_calibrators_cannot_emit_a_probability_of_one_half_or_more(
        symbol, timeframe):
    """A consumer must not assume a 0.60-style threshold is reachable.

    Every shipped model is a NO_SIGNAL/DROP artifact and its isotonic
    calibrator, fitted on out-of-fold predictions, tops out below 0.5 across
    its whole input domain. Anything gated on `p_win >= 0.5` would never fire.
    If a future artifact changes that, this test should fail and be read
    before anything is wired to it.
    """
    from analysis.research.model_loader import load_model

    model = load_model(symbol, timeframe, registry_path=REGISTRY)
    assert model.calibrator is not None, "expected a calibrated artifact"
    grid = np.linspace(0.0, 1.0, 1001)
    out = np.asarray(model.calibrator.predict(grid), dtype=float)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.max() <= 0.5 + 1e-9, (
        f"{symbol}/{timeframe}: calibrated p_win now reaches {out.max():.4f}; "
        f"re-read the model's verdict before relying on it")


@pytest.mark.parametrize("symbol,timeframe", _models(), ids=IDS)
def test_the_decision_threshold_comes_from_research_not_from_kairos(symbol, timeframe):
    """The threshold is the research repo's out-of-fold choice, carried verbatim.

    KAIROS must not invent one. The legacy path's hard-coded 0.60 belongs to a
    different model with different probability semantics and is not applicable
    here.
    """
    from analysis.research.model_loader import load_model

    card = load_model(symbol, timeframe, registry_path=REGISTRY).card
    assert card.decision_threshold is not None
    assert 0.0 < card.decision_threshold < 0.5, (
        "a threshold at or above 0.5 could never be crossed by these calibrators")

    manifest = json.loads(
        (ROOT / "models" / "research" / symbol / timeframe /
         mc.RESEARCH_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert card.decision_threshold == manifest["threshold"]["threshold"]


@pytest.mark.parametrize("symbol,timeframe", _models(), ids=IDS)
def test_the_research_verdict_travels_with_the_model(symbol, timeframe):
    """A NO_SIGNAL model must never be able to present as anything else."""
    from analysis.research.model_loader import load_model

    model = load_model(symbol, timeframe, registry_path=REGISTRY)
    manifest = json.loads(
        (model.directory / mc.RESEARCH_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert model.card.research_verdict == manifest["verdict"]
    assert model.card.research_verdict in ("NO_SIGNAL", "DROP")


def test_kairos_never_treats_p_win_as_a_direction_probability():
    """`p_win` for a short is P(short's TP first), not P(price falls).

    Both sides of the same bar are scored independently and both are
    probabilities of success — they do not, and must not, sum to 1.
    """
    import glob

    pairs = 0
    for path in sorted(glob.glob(str(ROOT / "tests/fixtures/research/golden/*.json"))):
        golden = json.loads(Path(path).read_text(encoding="utf-8"))
        by_ts = {}
        for s in golden["samples"]:
            by_ts.setdefault(s["timestamp"], {})[s["direction"]] = s["p_win"]
        for v in by_ts.values():
            if len(v) == 2:
                pairs += 1
                assert abs((v["long"] + v["short"]) - 1.0) > 1e-6, (
                    "long and short p_win summed to 1 — that would mean the model "
                    "is being read as a direction probability")
    assert pairs > 0
