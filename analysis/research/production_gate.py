"""The only route to PRODUCTION_ELIGIBLE, and it is deliberately hard to walk.

What this gate is
-----------------
A checklist that must be satisfied by EVIDENCE ON DISK, plus a countersigned
human approval. Nothing here is a judgement call the code makes on its own, and
nothing an import or a successful model load can trigger.

    research verdict      the research repo classified this artifact
                          PRODUCTION_ELIGIBLE after its own gates
    final OOS             a recorded one-shot out-of-sample pass exists for
                          this exact model
    economics             the research economic gate passed and survived the
                          cost stress
    artifact verified     the bytes on disk hash to what the card declares
    contract verified     the feature list resolves, is scale-free, and its
                          fingerprint matches what was imported
    feature parity        KAIROS reproduces the research feature vectors and
                          probabilities within tolerance, recorded by a run of
                          the golden-parity suite
    explicit approval     a signed record naming a person, a date, the exact
                          model hash, and what they approved

What "eligible" does NOT mean
-----------------------------
It does not enable anything. KAIROS has no code path that reads this status and
starts trading; there is no auto-enable, and this module deliberately provides
none. ``PRODUCTION_ELIGIBLE`` is a statement that the evidence is complete —
turning a model on remains a separate, human, out-of-band act.

Why the approval record is a file and not a flag
------------------------------------------------
A boolean can be flipped by anyone at any time and carries no account of why.
An approval record names who, when, and against which artifact hash; if the
artifact changes by a single byte the hash stops matching and the approval no
longer applies to it. That is the property a flag cannot have.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis.research import contract as C
from analysis.research import model_card as mc
from analysis.research import model_registry as reg

APPROVAL_FILENAME = "approval.json"
RESEARCH_EVIDENCE_FILENAME = "research_evidence.json"

#: Fields an approval record must carry. None is optional and none is defaulted.
REQUIRED_APPROVAL_FIELDS = ("approved_by", "approved_at_utc", "model_hash",
                            "statement", "evidence_reviewed")

#: The research classification this gate requires. Anything less means the
#: research repository itself did not consider the evidence complete.
REQUIRED_RESEARCH_VERDICT = "PRODUCTION_ELIGIBLE"


class ProductionGateError(Exception):
    """The gate was asked to approve something the evidence does not support."""


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"check": self.name, "passed": self.passed, "detail": self.detail,
                "evidence": self.evidence}


@dataclass
class GateReport:
    model_id: str
    checks: List[Check]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]

    def as_dict(self) -> Dict[str, Any]:
        return {"model_id": self.model_id, "passed": self.passed,
                "checks": [c.as_dict() for c in self.checks],
                "failed": [c.name for c in self.failures()]}

    def describe(self) -> str:
        if self.passed:
            return f"{self.model_id}: every production-eligibility check passed"
        return (f"{self.model_id}: NOT eligible — failed "
                f"{[c.name for c in self.failures()]}")


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def evaluate(directory) -> GateReport:
    """Run every check against the evidence in one model's directory."""
    d = Path(directory)
    checks: List[Check] = []

    # --- the card itself -----------------------------------------------------
    try:
        card = mc.load(d)
    except mc.ModelCardError as exc:
        return GateReport(str(d), [Check("model_card", False, str(exc))])
    checks.append(Check("model_card", True, card.describe(),
                        {"model_id": card.model_id, "symbol": card.symbol,
                         "timeframe": card.timeframe}))

    # --- artifact bytes ------------------------------------------------------
    model_path = d / mc.MODEL_FILENAME
    if not model_path.exists():
        checks.append(Check("artifact_verified", False, f"{model_path} is missing"))
    else:
        actual = mc.sha256_file(model_path)
        checks.append(Check(
            "artifact_verified", actual == card.model_hash,
            f"artifact sha256 {actual[:16]} vs card {card.model_hash[:16]}",
            {"actual": actual, "declared": card.model_hash}))

    # --- the feature contract ------------------------------------------------
    try:
        contract = C.build_contract(card.symbol, card.timeframe, card.feature_list)
        C.assert_scale_free(contract)
        fp = C.contract_fingerprint(contract)
        recorded = (card.extra or {}).get("kairos_contract_fingerprint")
        checks.append(Check(
            "contract_verified", recorded is None or recorded == fp,
            (f"{contract.feature_count} scale-free features; fingerprint "
             f"{fp[:16]}" + ("" if recorded is None else f" vs recorded {recorded[:16]}")),
            {"fingerprint": fp, "recorded": recorded}))
    except C.ContractError as exc:
        checks.append(Check("contract_verified", False, str(exc)))

    # --- research evidence ---------------------------------------------------
    evidence = _load_json(d / RESEARCH_EVIDENCE_FILENAME)
    if evidence is None:
        checks.append(Check(
            "research_verdict", False,
            f"no {RESEARCH_EVIDENCE_FILENAME}; the research repository's own "
            f"classification and out-of-sample result are not recorded here"))
        checks.append(Check("final_oos", False, "no research evidence file"))
        checks.append(Check("economics", False, "no research evidence file"))
    else:
        verdict = evidence.get("final_classification")
        checks.append(Check(
            "research_verdict", verdict == REQUIRED_RESEARCH_VERDICT,
            f"research classification is {verdict!r}, required "
            f"{REQUIRED_RESEARCH_VERDICT!r}",
            {"classification": verdict,
             "pre_oos_classification": evidence.get("pre_oos_classification")}))

        oos = evidence.get("final_oos") or {}
        checks.append(Check(
            "final_oos", bool(oos.get("G8_passed")),
            (f"out-of-sample AUC {oos.get('roc_auc')}, beats prior log-loss="
             f"{oos.get('beats_prior_log_loss')}, net expectancy "
             f"{oos.get('net_expectancy_r')} R" if oos
             else "no final out-of-sample record"),
            oos))

        ec = evidence.get("economics") or {}
        checks.append(Check(
            "economics", bool(ec.get("passed")) and bool(ec.get("survives_cost_stress")),
            (f"validation net expectancy {ec.get('net_expectancy_r')} R over "
             f"{ec.get('n_trades')} trades; cost stress "
             f"{ec.get('robustness')}" if ec else "no economic record"),
            ec))

    # --- feature parity ------------------------------------------------------
    parity = (evidence or {}).get("kairos_parity") or {}
    checks.append(Check(
        "feature_parity", bool(parity.get("passed")),
        (f"max feature delta {parity.get('max_feature_delta')}, max p_win delta "
         f"{parity.get('max_probability_delta')} over {parity.get('n_samples')} "
         f"golden samples" if parity else
         "no recorded golden-parity run for this model"),
        parity))

    # --- explicit human approval --------------------------------------------
    approval = _load_json(d / APPROVAL_FILENAME)
    if approval is None:
        checks.append(Check(
            "explicit_approval", False,
            f"no {APPROVAL_FILENAME}. Eligibility is a human decision and must be "
            f"recorded as one; it is never inferred from passing checks."))
    else:
        missing = [f for f in REQUIRED_APPROVAL_FIELDS if not approval.get(f)]
        hash_ok = approval.get("model_hash") == card.model_hash
        checks.append(Check(
            "explicit_approval", not missing and hash_ok,
            (f"missing fields {missing}" if missing else
             f"approved by {approval.get('approved_by')} on "
             f"{approval.get('approved_at_utc')}" +
             ("" if hash_ok else
              " — but the approval names a different artifact hash, so it does "
              "not apply to the model on disk")),
            {"approved_by": approval.get("approved_by"),
             "approved_at_utc": approval.get("approved_at_utc"),
             "hash_matches": hash_ok}))

    return GateReport(card.model_id, checks)


def promote(directory, registry_path=reg.DEFAULT_REGISTRY_PATH) -> dict:
    """Set PRODUCTION_ELIGIBLE, or refuse and say exactly what is missing."""
    report = evaluate(directory)
    if not report.passed:
        raise ProductionGateError(
            report.describe() + ". Nothing was changed. " +
            "; ".join(f"{c.name}: {c.detail}" for c in report.failures()))

    registry = reg.load_registry(registry_path)
    entry = registry.get(report.model_id)
    updated = [
        reg.RegistryEntry(**{**vars(e),
                             "status": reg.PRODUCTION_ELIGIBLE} if e.model_id == entry.model_id
                          else vars(e))
        for e in registry.entries.values()
    ]
    reg.write_registry(updated, registry_path)
    return {"model_id": report.model_id, "status": reg.PRODUCTION_ELIGIBLE,
            "report": report.as_dict()}


def approval_template(card: mc.ModelCard) -> Dict[str, Any]:
    """A blank approval record. Deliberately not pre-filled.

    ``approved_by`` and ``statement`` are left empty because a template that
    fills them in is a template that gets committed unread.
    """
    return {
        "approved_by": "",
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_hash": card.model_hash,
        "statement": "",
        "evidence_reviewed": [],
        "note": ("Eligibility is not activation. Approving this record states that "
                 "the recorded evidence was reviewed; it does not enable trading, "
                 "and KAIROS has no code path that reads this file to do so."),
    }
