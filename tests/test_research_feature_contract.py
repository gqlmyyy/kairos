"""The canonical feature contract: schema, ordering, completeness, scale-freedom."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.research import contract as C

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "tests" / "fixtures" / "research" / "golden"
REGISTRY = ROOT / "models" / "research" / "registry.json"

REQUIRED_CONTRACT_FIELDS = (
    "name", "source", "formula", "timeframe", "lookback", "minimum_history",
    "dtype", "unit", "normalization", "availability", "missing_policy",
    "stationarity", "requires",
)


def _model_dirs():
    if not REGISTRY.exists():
        return []
    raw = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return [Path(m["path"]) for m in raw["models"]]


def _cards():
    out = []
    for d in _model_dirs():
        card = d / "model_card.json"
        if card.exists():
            out.append(json.loads(card.read_text(encoding="utf-8")))
    return out


pytestmark = pytest.mark.skipif(not REGISTRY.exists(),
                                reason="research models not imported in this checkout")


def test_every_library_feature_declares_the_full_contract():
    """A partially-described feature is how provenance questions go unanswered."""
    for name, spec in C.FEATURE_LIBRARY.items():
        d = spec.as_dict()
        missing = [f for f in REQUIRED_CONTRACT_FIELDS if f not in d]
        assert not missing, f"{name} is missing contract fields {missing}"
        assert d["formula"].strip(), f"{name} has an empty formula"
        assert d["minimum_history"] >= d["lookback"], (
            f"{name}: minimum_history {d['minimum_history']} < lookback {d['lookback']}")
        assert d["availability"] == C.AVAILABILITY_CLOSE_TIME


def test_every_library_feature_is_scale_free():
    """The library is the scale-free vocabulary; nothing else may live in it."""
    offenders = [(n, s.stationarity) for n, s in C.FEATURE_LIBRARY.items()
                 if s.stationarity != C.SCALE_FREE]
    assert not offenders, f"non-scale-free features in the library: {offenders}"


@pytest.mark.parametrize("name", sorted(C.EXCLUDED_NON_STATIONARY))
def test_price_scale_features_are_refused_by_name(name):
    """`atr`, `macd_line`, `ema_20` … must not resolve into a research contract.

    This is the mechanical form of the research finding: a price-unit column
    cannot transfer across price regimes, so it must be impossible to request
    rather than merely discouraged.
    """
    if name in C.FEATURE_LIBRARY:
        pytest.skip(f"{name} is a legitimate library feature")
    with pytest.raises(C.ContractError):
        C.resolve(name, "H1")


def test_unknown_feature_is_an_error_not_a_default():
    with pytest.raises(C.ContractError, match="not covered"):
        C.resolve("totally_made_up_feature", "H1")


@pytest.mark.parametrize("card", _cards(), ids=lambda c: c["model_id"])
def test_every_shipped_model_resolves_and_is_scale_free(card):
    contract = C.build_contract(card["symbol"], card["timeframe"], card["feature_list"])
    C.assert_scale_free(contract)
    assert contract.feature_count == len(card["feature_list"])


@pytest.mark.parametrize("card", _cards(), ids=lambda c: c["model_id"])
def test_contract_preserves_the_models_feature_order_exactly(card):
    """Order is part of the contract: the same names permuted is a different input."""
    contract = C.build_contract(card["symbol"], card["timeframe"], card["feature_list"])
    assert list(contract.feature_names) == list(card["feature_list"])
    assert [s.name for s in contract.specs] == list(card["feature_list"])


@pytest.mark.parametrize("card", _cards(), ids=lambda c: c["model_id"])
def test_context_prefixed_features_keep_their_own_timeframe(card):
    """An `H4_rsi` on an H1 row is still an H4 measurement.

    Reporting it as H1 would make the causality claim unverifiable, because
    the warm-up and availability checks both key off the spec's timeframe.
    """
    contract = C.build_contract(card["symbol"], card["timeframe"], card["feature_list"])
    for spec in contract.specs:
        base, ctx = C.strip_context(spec.name)
        if ctx is not None:
            assert spec.timeframe == ctx, f"{spec.name} claims timeframe {spec.timeframe}"


def test_fingerprint_changes_when_a_formula_changes():
    """Renaming nothing but changing arithmetic must still move the fingerprint."""
    names = ["rsi", "adx", "atr_pct"]
    a = C.build_contract("XAUUSD", "H1", names)
    tweaked = tuple(
        s if s.name != "rsi" else C.FeatureSpec(**{**s.as_dict(), "formula": "different"})
        for s in a.specs
    )
    b = C.CanonicalContract(a.symbol, a.entry_timeframe, a.context_timeframes,
                            a.feature_names, tweaked)
    assert C.contract_fingerprint(a) != C.contract_fingerprint(b)


def test_duplicate_feature_names_are_refused():
    with pytest.raises(C.ContractError, match="duplicates"):
        C.build_contract("XAUUSD", "H1", ["rsi", "adx", "rsi"])


def test_no_shipped_model_requests_m30():
    """The contract adds no timeframe a model did not ask for."""
    for card in _cards():
        assert "M30" not in card["context_timeframes"], card["model_id"]
        assert not any(f.startswith("M30_") for f in card["feature_list"]), card["model_id"]


def test_legacy_and_research_vocabularies_do_not_overlap_in_meaning():
    """Same word, different arithmetic — the two contracts must stay apart.

    `trend_score` is a normalised regression slope here and a bucket from
    {40, 65, 70, 75, 85} in the legacy spec; `market_regime` is binary here
    and one of four encoded states there. Sharing a name is not sharing a
    definition, and a test is the only thing that keeps someone from mapping
    one onto the other.
    """
    from analysis.models import entry_feature_spec as legacy

    shared = set(legacy.FEATURE_NAMES) & set(C.FEATURE_LIBRARY)
    assert shared, "expected some shared names — that is exactly the hazard"
    for name in shared:
        research_formula = C.FEATURE_LIBRARY[name].formula
        legacy_calc = legacy.FEATURE_CONTRACT[name]["calculation"]
        assert research_formula != legacy_calc, (
            f"{name}: the two contracts must not be assumed compatible by name")
