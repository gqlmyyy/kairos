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

#: Every registered model, both generations. These properties must hold for
#: all of them, so the suite parametrises over the registry rather than
#: pinning one generation.

pytestmark = pytest.mark.skipif(not REGISTRY.exists(),
                                reason="research models not imported in this checkout")


def _models():
    raw = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return [(m["symbol"], m["timeframe"], m["version"], m["path"])
            for m in sorted(raw["models"], key=lambda x: x["model_id"])]


IDS = [f"{v}-{s}-{t}" for s, t, v, _ in _models()]

#: The legacy entry path hard-codes `p_win >= 0.60`. No shipped research model
#: can reach it; measured ceilings are 0.393-0.501.
LEGACY_THRESHOLD = 0.60


@pytest.mark.parametrize("symbol,timeframe,version,path", _models(), ids=IDS)
def test_the_card_states_its_target_and_matching_semantics(symbol, timeframe,
                                                           version, path):
    """The declared semantics must match the declared kind, and the geometry
    must be present exactly when the kind has one."""
    from analysis.research.model_loader import load_model

    card = load_model(symbol, timeframe, registry_path=REGISTRY, version=version).card
    assert card.target_kind in (mc.KIND_BARRIER, mc.KIND_RETURN)
    assert card.probability_semantics == mc.PROBABILITY_SEMANTICS_BY_KIND[card.target_kind]
    assert card.horizon_bars > 0
    if card.target_kind == mc.KIND_BARRIER:
        assert "TP" in card.target and "SL" in card.target
        assert card.tp_atr_multiple and card.sl_atr_multiple
        assert "SL distance" in card.risk_unit
    else:
        assert card.return_threshold_atr is not None
        assert card.tp_atr_multiple is None and card.sl_atr_multiple is None
        assert "1 ATR" in card.risk_unit


@pytest.mark.parametrize("symbol,timeframe,version,path", _models(), ids=IDS)
def test_p_win_is_labelled_p_win_all_the_way_through(symbol, timeframe,
                                                     version, path):
    from analysis.research.inference import Prediction

    pred = Prediction(status="OK", p_win=0.4, symbol=symbol, timeframe=timeframe)
    assert "p_win" in pred.as_dict()
    assert pred.as_dict()["semantics"]


@pytest.mark.parametrize("symbol,timeframe,version,path", _models(), ids=IDS)
def test_no_shipped_calibrator_can_reach_the_legacy_threshold(
        symbol, timeframe, version, path):
    """A consumer must not assume the legacy 0.60 gate is reachable.

    Every shipped model was rejected or only conditionally accepted by the
    research gates, and its isotonic calibrator -- fitted on out-of-fold
    predictions -- tops out around the label's own base rate: 0.393-0.501
    across the eighteen artifacts. Anything gated on `p_win >= 0.60` would
    never fire on any of them, which is a property of these models rather
    than a defect in the integration.

    The exact ceiling tracks the target: an asymmetric 2.5:1.5 barrier has a
    base rate near 0.375 and tops out at 0.5, while a symmetric 1.5:1.5 label
    has a base rate near 0.5 and can edge just past it. So the assertion is
    against the legacy threshold, not against 0.5.

    If a future artifact changes this, the test should fail and be read before
    anything is wired to it.
    """
    from analysis.research.model_loader import load_model

    model = load_model(symbol, timeframe, registry_path=REGISTRY, version=version)
    assert model.calibrator is not None, "expected a calibrated artifact"
    grid = np.linspace(0.0, 1.0, 1001)
    out = np.asarray(model.calibrator.predict(grid), dtype=float)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.max() < LEGACY_THRESHOLD, (
        f"{version} {symbol}/{timeframe}: calibrated p_win now reaches "
        f"{out.max():.4f}, at or above the legacy {LEGACY_THRESHOLD} gate. "
        f"Re-read the model's research verdict before relying on it.")


@pytest.mark.parametrize("symbol,timeframe,version,path", _models(), ids=IDS)
def test_the_decision_threshold_comes_from_research_not_from_kairos(
        symbol, timeframe, version, path):
    """The threshold is the research repo's out-of-fold choice, carried verbatim.

    KAIROS must not invent one. The legacy path's hard-coded 0.60 belongs to a
    different model with different probability semantics and is not applicable
    here.
    """
    from analysis.research.model_loader import load_model

    card = load_model(symbol, timeframe, registry_path=REGISTRY, version=version).card
    manifest = json.loads((ROOT / path / mc.RESEARCH_MANIFEST_FILENAME)
                          .read_text(encoding="utf-8"))
    # research_v2 records it under `threshold.threshold`; research_v3 selected
    # its threshold on development and records it at the top level. Whichever
    # the artifact carries, KAIROS must copy it rather than invent one.
    research_threshold = (manifest.get("threshold", {}) or {}).get("threshold")
    if research_threshold is None:
        research_threshold = manifest.get("decision_threshold")
    assert card.decision_threshold == research_threshold, (
        "KAIROS must not invent a threshold; the legacy path's hard-coded 0.60 "
        "belongs to a different model with different probability semantics")
    if card.decision_threshold is not None:
        assert 0.0 < card.decision_threshold < 1.0


@pytest.mark.parametrize("symbol,timeframe,version,path", _models(), ids=IDS)
def test_the_research_verdict_travels_with_the_model(symbol, timeframe,
                                                     version, path):
    """A NO_SIGNAL model must never be able to present as anything else."""
    from analysis.research.model_loader import load_model

    model = load_model(symbol, timeframe, registry_path=REGISTRY, version=version)
    manifest = json.loads(
        (model.directory / mc.RESEARCH_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    recorded = manifest.get("verdict") or manifest.get("final_classification")
    assert model.card.research_verdict == recorded
    assert model.card.research_verdict in (
        "NO_SIGNAL", "DROP", "RESEARCH", "CANDIDATE", "VALIDATED",
        "PRODUCTION_ELIGIBLE")


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
