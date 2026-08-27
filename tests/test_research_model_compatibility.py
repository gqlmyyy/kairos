"""The loader must refuse anything it cannot fully verify.

Every check in the chain gets a test that BREAKS it and asserts the load fails
with MODEL_NOT_COMPATIBLE. A validation nobody has seen fail is a validation
nobody knows works — and this repository has already shipped a model that
loaded fine and served numbers unrelated to the trade.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from analysis.research import model_card as mc
from analysis.research import model_registry as reg
from analysis.research.model_loader import ModelNotCompatible, load_model

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "models" / "research" / "registry.json"

#: The generation these path-level tests operate on. Both are registered, so a
#: test that wants "a model" has to name one.
VERSION = "research_v2"

pytestmark = pytest.mark.skipif(not REGISTRY.exists(),
                                reason="research models not imported in this checkout")


@pytest.fixture
def sandbox(tmp_path):
    """A private copy of one model, so a test can corrupt it safely."""
    src = ROOT / "models" / "research" / VERSION / "XAUUSD" / "H1"
    dest = tmp_path / "models" / "research" / "XAUUSD" / "H1"
    shutil.copytree(src, dest)
    entry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    row = next(m for m in entry["models"] if m["model_id"] == f"{VERSION}__XAUUSD__H1")
    row["path"] = str(dest)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"schema": entry["schema"], "models": [row]}),
                             encoding="utf-8")
    return dest, registry_path


def _card(dest: Path) -> dict:
    return json.loads((dest / mc.CARD_FILENAME).read_text(encoding="utf-8"))


def _write_card(dest: Path, card: dict) -> None:
    (dest / mc.CARD_FILENAME).write_text(json.dumps(card), encoding="utf-8")


def test_a_clean_model_loads(sandbox):
    dest, registry_path = sandbox
    model = load_model("XAUUSD", "H1", registry_path=registry_path)
    assert model.symbol == "XAUUSD" and model.timeframe == "H1"
    assert len(model.feature_names) == model.card.feature_count


def test_symbol_mismatch_is_refused(sandbox):
    """An EURUSD path must never be served an XAUUSD model."""
    _, registry_path = sandbox
    with pytest.raises(ModelNotCompatible, match="no model registered for EURUSD"):
        load_model("EURUSD", "H1", registry_path=registry_path)


def test_a_model_card_claiming_the_wrong_symbol_is_refused(sandbox):
    dest, registry_path = sandbox
    card = _card(dest)
    card["symbol"] = "EURUSD"
    _write_card(dest, card)
    with pytest.raises(ModelNotCompatible, match="symbol mismatch"):
        load_model("XAUUSD", "H1", registry_path=registry_path)


def test_timeframe_mismatch_is_refused(sandbox):
    _, registry_path = sandbox
    with pytest.raises(ModelNotCompatible, match="no model registered for XAUUSD/H4"):
        load_model("XAUUSD", "H4", registry_path=registry_path)


def test_a_model_card_claiming_the_wrong_timeframe_is_refused(sandbox):
    dest, registry_path = sandbox
    card = _card(dest)
    card["timeframe"] = "M15"
    _write_card(dest, card)
    with pytest.raises(ModelNotCompatible, match="timeframe mismatch"):
        load_model("XAUUSD", "H1", registry_path=registry_path)


def test_model_hash_mismatch_is_refused(sandbox):
    """The bytes on disk must be the bytes the card describes."""
    dest, registry_path = sandbox
    with (dest / mc.MODEL_FILENAME).open("ab") as fh:
        fh.write(b"\x00")
    with pytest.raises(ModelNotCompatible, match="model hash mismatch"):
        load_model("XAUUSD", "H1", registry_path=registry_path)


def test_a_card_edited_to_match_a_swapped_artifact_still_fails_on_the_registry(sandbox):
    """Editing the card to launder a changed artifact does not get past the registry."""
    dest, registry_path = sandbox
    with (dest / mc.MODEL_FILENAME).open("ab") as fh:
        fh.write(b"\x00")
    card = _card(dest)
    card["model_hash"] = mc.sha256_file(dest / mc.MODEL_FILENAME)
    _write_card(dest, card)
    with pytest.raises(ModelNotCompatible, match="registry hash"):
        load_model("XAUUSD", "H1", registry_path=registry_path)


def test_missing_card_is_refused(sandbox):
    dest, registry_path = sandbox
    (dest / mc.CARD_FILENAME).unlink()
    with pytest.raises(ModelNotCompatible, match="no model card"):
        load_model("XAUUSD", "H1", registry_path=registry_path)


@pytest.mark.parametrize("field", ["target", "horizon_bars", "training_dataset_hash",
                                   "probability_semantics", "feature_list", "model_hash"])
def test_every_required_card_field_is_required(sandbox, field):
    dest, registry_path = sandbox
    card = _card(dest)
    del card[field]
    _write_card(dest, card)
    with pytest.raises(ModelNotCompatible, match="missing required field"):
        load_model("XAUUSD", "H1", registry_path=registry_path)


def test_feature_schema_version_mismatch_is_refused(sandbox):
    """Same names, different arithmetic, is still a different model input."""
    dest, registry_path = sandbox
    card = _card(dest)
    card["feature_schema_version"] = "research-0.9.0"
    _write_card(dest, card)
    with pytest.raises(ModelNotCompatible, match="feature schema mismatch"):
        load_model("XAUUSD", "H1", registry_path=registry_path)


def test_feature_order_mismatch_is_refused(sandbox):
    """The same names permuted is a different input, accepted silently downstream."""
    dest, registry_path = sandbox
    card = _card(dest)
    features = list(card["feature_list"])
    features[1], features[2] = features[2], features[1]
    card["feature_list"] = features
    _write_card(dest, card)
    with pytest.raises(ModelNotCompatible, match="feature ORDER mismatch"):
        load_model("XAUUSD", "H1", registry_path=registry_path)


def test_a_price_scale_feature_in_the_list_is_refused(sandbox):
    """A PRICE_UNIT column must not reach a research model, even via the card."""
    dest, registry_path = sandbox
    card = _card(dest)
    card["feature_list"] = list(card["feature_list"]) + ["atr"]
    _write_card(dest, card)
    with pytest.raises(ModelNotCompatible, match="PRICE_UNIT|excluded"):
        load_model("XAUUSD", "H1", registry_path=registry_path)


def test_foreign_probability_semantics_are_refused(sandbox):
    """A model predicting something else must not be served through this path."""
    dest, registry_path = sandbox
    card = _card(dest)
    card["probability_semantics"] = "P(price goes up)"
    _write_card(dest, card)
    with pytest.raises(ModelNotCompatible, match="probability_semantics"):
        load_model("XAUUSD", "H1", registry_path=registry_path)


def test_a_retired_model_is_not_served(sandbox):
    dest, registry_path = sandbox
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    raw["models"][0]["status"] = reg.RETIRED
    registry_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ModelNotCompatible, match="no model registered"):
        load_model("XAUUSD", "H1", registry_path=registry_path)


def test_a_missing_artifact_is_refused(sandbox):
    dest, registry_path = sandbox
    (dest / mc.MODEL_FILENAME).unlink()
    with pytest.raises(ModelNotCompatible, match="does not exist"):
        load_model("XAUUSD", "H1", registry_path=registry_path)


def test_two_models_for_one_pair_is_an_error_not_a_silent_choice(sandbox):
    """An implicit pick between two models is a second source of truth."""
    dest, registry_path = sandbox
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    clone = dict(raw["models"][0])
    clone["model_id"] = clone["model_id"] + "__clone"
    raw["models"].append(clone)
    registry_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ModelNotCompatible, match="Pass version=|retire the rest"):
        load_model("XAUUSD", "H1", registry_path=registry_path)


# --- registry-wide invariants ------------------------------------------------

def _registry():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_no_shipped_model_is_marked_production_eligible():
    """Offline integration does not promote anything.

    Every shipped model carries a NO_SIGNAL or DROP verdict from the research
    repo. VALIDATED is a recorded human decision about a specific artifact
    hash, never a side effect of a successful import.
    """
    for m in _registry()["models"]:
        assert m["status"] != reg.PRODUCTION_ELIGIBLE, f"{m['model_id']} is {m['status']}"


def test_the_registry_has_no_production_status():
    assert reg.PRODUCTION_ELIGIBLE in reg.GATED_STATUSES
    assert all(m["status"] in reg.STATUSES for m in _registry()["models"])


def test_every_registered_artifact_hash_matches_the_file_on_disk():
    for m in _registry()["models"]:
        path = ROOT / m["path"] / mc.MODEL_FILENAME
        assert path.exists(), f"{m['model_id']}: {path} missing"
        assert mc.sha256_file(path) == m["model_hash"], f"{m['model_id']} hash drifted"


def test_research_models_never_write_over_legacy_artifacts():
    """The two stores must not overlap — one vocabulary each."""
    for m in _registry()["models"]:
        p = Path(m["path"]).as_posix()
        assert p.startswith("models/research/"), p
        assert m["version"] in p, "artifacts must be version-scoped"
        assert "models/entry" not in p


def test_every_model_carries_its_research_verdict():
    """A NO_SIGNAL model must not be able to look like a validated one."""
    known = {"NO_SIGNAL", "DROP", "RESEARCH", "CANDIDATE", "VALIDATED",
             "PRODUCTION_ELIGIBLE"}
    for m in _registry()["models"]:
        assert m["research_verdict"] in known, m["model_id"]
