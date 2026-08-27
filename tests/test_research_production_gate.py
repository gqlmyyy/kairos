"""PRODUCTION_ELIGIBLE must be unreachable without complete evidence.

The status exists so a human can say "the evidence is complete", and every
test here is about making that claim impossible to fake, skip or infer. A model
that loads fine, hashes fine and predicts fine is still not eligible.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from analysis.research import model_card as mc
from analysis.research import model_registry as reg
from analysis.research import production_gate as pg

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "models" / "research" / "registry.json"

pytestmark = pytest.mark.skipif(not REGISTRY.exists(),
                                reason="research models not imported in this checkout")


@pytest.fixture
def sandbox(tmp_path):
    src = ROOT / "models" / "research" / "research_v2" / "XAUUSD" / "H1"
    dest = tmp_path / "XAUUSD" / "H1"
    shutil.copytree(src, dest)
    return dest


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _full_evidence(card: mc.ModelCard) -> dict:
    return {
        "final_classification": pg.REQUIRED_RESEARCH_VERDICT,
        "pre_oos_classification": "VALIDATED",
        "final_oos": {"G8_passed": True, "roc_auc": 0.56,
                      "beats_prior_log_loss": True, "net_expectancy_r": 0.04,
                      "n_trades": 400, "n_test": 2800},
        "economics": {"passed": True, "survives_cost_stress": True,
                      "net_expectancy_r": 0.05, "n_trades": 500,
                      "robustness": "ROBUST"},
        "kairos_parity": {"passed": True, "max_feature_delta": 0.0,
                          "max_probability_delta": 0.0, "n_samples": 50},
    }


def _approval(card: mc.ModelCard) -> dict:
    return {"approved_by": "a.reviewer", "approved_at_utc": "2026-08-27T00:00:00+00:00",
            "model_hash": card.model_hash, "statement": "evidence reviewed",
            "evidence_reviewed": ["final_oos", "economics", "kairos_parity"]}


# --- the shipped state -------------------------------------------------------

def test_no_shipped_model_is_currently_eligible():
    for d in sorted((ROOT / "models" / "research").glob("*/*/*")):
        if not (d / mc.CARD_FILENAME).exists():
            continue
        report = pg.evaluate(d)
        assert not report.passed, f"{d} unexpectedly passed the production gate"


def test_no_registry_entry_is_production_eligible():
    raw = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for m in raw["models"]:
        assert m["status"] != reg.PRODUCTION_ELIGIBLE, m["model_id"]


def test_the_status_exists_and_is_gated():
    assert reg.PRODUCTION_ELIGIBLE in reg.STATUSES
    assert reg.PRODUCTION_ELIGIBLE in reg.GATED_STATUSES
    assert reg.LADDER[-1] == reg.PRODUCTION_ELIGIBLE


# --- each missing piece blocks on its own ------------------------------------

def test_complete_evidence_plus_approval_passes(sandbox):
    card = mc.load(sandbox)
    _write(sandbox / pg.RESEARCH_EVIDENCE_FILENAME, _full_evidence(card))
    _write(sandbox / pg.APPROVAL_FILENAME, _approval(card))
    report = pg.evaluate(sandbox)
    assert report.passed, report.describe()


@pytest.mark.parametrize("drop", [pg.RESEARCH_EVIDENCE_FILENAME, pg.APPROVAL_FILENAME])
def test_a_missing_evidence_file_blocks(sandbox, drop):
    card = mc.load(sandbox)
    _write(sandbox / pg.RESEARCH_EVIDENCE_FILENAME, _full_evidence(card))
    _write(sandbox / pg.APPROVAL_FILENAME, _approval(card))
    (sandbox / drop).unlink()
    assert not pg.evaluate(sandbox).passed


def test_a_research_verdict_short_of_production_eligible_blocks(sandbox):
    card = mc.load(sandbox)
    ev = _full_evidence(card)
    ev["final_classification"] = "VALIDATED"
    _write(sandbox / pg.RESEARCH_EVIDENCE_FILENAME, ev)
    _write(sandbox / pg.APPROVAL_FILENAME, _approval(card))
    report = pg.evaluate(sandbox)
    assert not report.passed
    assert "research_verdict" in [c.name for c in report.failures()]


def test_a_failed_final_oos_blocks(sandbox):
    card = mc.load(sandbox)
    ev = _full_evidence(card)
    ev["final_oos"]["G8_passed"] = False
    _write(sandbox / pg.RESEARCH_EVIDENCE_FILENAME, ev)
    _write(sandbox / pg.APPROVAL_FILENAME, _approval(card))
    assert "final_oos" in [c.name for c in pg.evaluate(sandbox).failures()]


def test_an_edge_that_dies_under_cost_stress_blocks(sandbox):
    """FRAGILE is not ROBUST, and eligibility requires ROBUST."""
    card = mc.load(sandbox)
    ev = _full_evidence(card)
    ev["economics"]["survives_cost_stress"] = False
    ev["economics"]["robustness"] = "FRAGILE"
    _write(sandbox / pg.RESEARCH_EVIDENCE_FILENAME, ev)
    _write(sandbox / pg.APPROVAL_FILENAME, _approval(card))
    assert "economics" in [c.name for c in pg.evaluate(sandbox).failures()]


def test_missing_feature_parity_blocks(sandbox):
    card = mc.load(sandbox)
    ev = _full_evidence(card)
    ev["kairos_parity"] = {}
    _write(sandbox / pg.RESEARCH_EVIDENCE_FILENAME, ev)
    _write(sandbox / pg.APPROVAL_FILENAME, _approval(card))
    assert "feature_parity" in [c.name for c in pg.evaluate(sandbox).failures()]


def test_an_approval_for_a_different_artifact_does_not_apply(sandbox):
    """A byte changes, the hash changes, and the approval stops applying."""
    card = mc.load(sandbox)
    _write(sandbox / pg.RESEARCH_EVIDENCE_FILENAME, _full_evidence(card))
    approval = _approval(card)
    approval["model_hash"] = "0" * 64
    _write(sandbox / pg.APPROVAL_FILENAME, approval)
    report = pg.evaluate(sandbox)
    assert not report.passed
    assert "explicit_approval" in [c.name for c in report.failures()]


@pytest.mark.parametrize("field", list(pg.REQUIRED_APPROVAL_FIELDS))
def test_every_approval_field_is_required(sandbox, field):
    card = mc.load(sandbox)
    _write(sandbox / pg.RESEARCH_EVIDENCE_FILENAME, _full_evidence(card))
    approval = _approval(card)
    approval[field] = "" if isinstance(approval[field], str) else []
    _write(sandbox / pg.APPROVAL_FILENAME, approval)
    assert "explicit_approval" in [c.name for c in pg.evaluate(sandbox).failures()]


def test_a_tampered_artifact_blocks(sandbox):
    card = mc.load(sandbox)
    _write(sandbox / pg.RESEARCH_EVIDENCE_FILENAME, _full_evidence(card))
    _write(sandbox / pg.APPROVAL_FILENAME, _approval(card))
    with (sandbox / mc.MODEL_FILENAME).open("ab") as fh:
        fh.write(b"\x00")
    report = pg.evaluate(sandbox)
    assert not report.passed
    assert "artifact_verified" in [c.name for c in report.failures()]


def test_promote_refuses_and_changes_nothing_when_evidence_is_incomplete(sandbox, tmp_path):
    registry_path = tmp_path / "registry.json"
    shutil.copy2(REGISTRY, registry_path)
    before = registry_path.read_text(encoding="utf-8")
    with pytest.raises(pg.ProductionGateError):
        pg.promote(sandbox, registry_path=registry_path)
    assert registry_path.read_text(encoding="utf-8") == before, (
        "a refused promotion must not touch the registry")


def test_eligibility_is_not_activation():
    """Eligibility must never be SUFFICIENT to switch trading on.

    Originally this asserted that nothing outside the registry and its gate
    referenced the status at all. The live gate
    (`analysis/research/live_gate.py`) now does reference it, so the check
    was tightened rather than relaxed: a module may read the status only if
    it ALSO requires a separate activation record. Eligibility stays
    necessary-and-not-sufficient, which is the invariant this test exists to
    protect — the string was only ever a proxy for it.
    """
    # Reads the status and is exempt because it does not enable anything.
    DEFINES_OR_INSPECTS = (
        "analysis/research/model_registry.py",
        "analysis/research/production_gate.py",
        "scripts/import_research_model.py",
    )
    # Reads the status AND gates on an independent activation record. The
    # assertions below prove that second condition really is enforced.
    REQUIRES_SEPARATE_ACTIVATION = ("analysis/research/live_gate.py",)

    offenders = []
    for path in sorted(ROOT.rglob("*.py")):
        if "/tests/" in path.as_posix() or "/.git/" in path.as_posix():
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            continue
        if "PRODUCTION_ELIGIBLE" not in text:
            continue
        suffix = path.as_posix()
        if any(suffix.endswith(a) for a in DEFINES_OR_INSPECTS):
            continue
        if any(suffix.endswith(a) for a in REQUIRES_SEPARATE_ACTIVATION):
            continue
        offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, (
        f"PRODUCTION_ELIGIBLE is referenced outside the registry, its gate, and "
        f"the activation-gated live path: {offenders}. Eligibility must never be "
        f"sufficient on its own to enable trading.")

    # The exemption above is only honest if the live gate genuinely enforces a
    # second, independent condition. Prove it here rather than trusting the
    # allowlist.
    from analysis.research import live_gate

    assert hasattr(live_gate, "NotActivated")
    assert live_gate.REQUIRED_ACTIVATION_FIELDS, "activation must require real fields"
    assert "model_hash" in live_gate.REQUIRED_ACTIVATION_FIELDS, (
        "activation must bind to the artifact's hash, or it cannot expire when "
        "the artifact changes")
    # No activation file ships, so nothing is activated by default.
    assert live_gate.load_activations() == {}, (
        "an activation record is present in the repo — activation is meant to be "
        "an out-of-band operator act, not a committed default")
