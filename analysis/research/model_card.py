"""The model artifact contract: what a research model must declare about itself.

``model.json`` alone is not evidence
------------------------------------
An XGBoost JSON records ``num_feature`` and, usually, an empty
``feature_names``. It says nothing about which symbol, which timeframe, which
label definition, or which dataset produced it. That is exactly how four
trainers in this repository wrote three different schemas to one filename with
nothing noticing (see ``analysis/models/entry_model_metadata.py``).

So a research model is servable only alongside a **model card** that declares
its full identity, and every declared field is checked before the model is
allowed to predict. The card is generated from the research manifest by
``scripts/import_research_model.py`` — never hand-written, never backfilled to
make a stubborn artifact load. Repairing provenance after the fact is
indistinguishable from fabricating it.

Target semantics
----------------
``target_kind`` and ``probability_semantics`` are carried explicitly because
the number a model emits is easy to misread, and because research_v3 produces
TWO different kinds of entry model:

``barrier``
    ``p_win = P(TP before SL | entry_direction)`` -- TP and SL at fixed ATR
    multiples measured at the entry bar, inside a bounded forward horizon.
    1R is the SL distance.

``return``
    ``p_win = P(move beyond the threshold | entry_direction)`` -- a
    hold-to-horizon trade with NO stop. 1R is defined as 1 ATR at entry.

These are different risk models and different probabilities. A barrier model's
0.55 and a return model's 0.55 do not mean the same thing and their
R-multiples are not comparable, so the kind travels with the artifact and the
loader refuses any card whose declared semantics do not match its kind.

Neither is P(price goes up), an expected return, or a confidence score. A
caller that treats them as such is wrong in a way no amount of calibration
fixes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CARD_FILENAME = "model_card.json"
MODEL_FILENAME = "model.joblib"
RESEARCH_MANIFEST_FILENAME = "research_manifest.json"

#: Bumped when the MEANING of a feature changes. Same name, different
#: arithmetic is the drift that is hardest to see and easiest to trade on, so
#: names and order agreeing is not sufficient — this must agree too.
CURRENT_FEATURE_SCHEMA_VERSION = "research-1.2.0"

KIND_BARRIER = "barrier"
KIND_RETURN = "return"

#: The probability each kind of research entry model emits. A card must
#: declare the string that matches its kind, exactly.
PROBABILITY_SEMANTICS_BY_KIND = {
    KIND_BARRIER: "p_win = P(TP before SL | entry_direction)",
    KIND_RETURN: "p_win = P(forward move beyond threshold | entry_direction)",
}

#: Back-compatible default for barrier models (every research_v2 artifact).
PROBABILITY_SEMANTICS = PROBABILITY_SEMANTICS_BY_KIND[KIND_BARRIER]

REQUIRED_FIELDS: Tuple[str, ...] = (
    "model_id", "symbol", "timeframe", "model_version", "feature_schema_version",
    "feature_list", "target", "target_kind", "horizon_bars",
    "training_dataset_hash", "feature_manifest_hash", "target_spec_hash", "model_hash",
    "probability_semantics", "research_verdict", "calibration", "context_timeframes",
    "entry_direction_encoding", "source_repo_commit", "environment",
)

#: Required only for a barrier target. A hold-to-horizon return model has no
#: stop, so demanding an SL multiple would force a card to declare a barrier
#: that does not exist.
REQUIRED_BARRIER_FIELDS: Tuple[str, ...] = ("tp_atr_multiple", "sl_atr_multiple")


class ModelCardError(Exception):
    """A model cannot prove what it is."""


@dataclass(frozen=True)
class ModelCard:
    """The declared identity of one research model artifact."""

    model_id: str
    symbol: str
    timeframe: str
    model_version: str
    feature_schema_version: str
    feature_list: Tuple[str, ...]
    target: str
    target_kind: str
    horizon_bars: int
    training_dataset_hash: str
    feature_manifest_hash: str
    target_spec_hash: str
    model_hash: str
    probability_semantics: str
    research_verdict: str
    calibration: str
    context_timeframes: Tuple[str, ...]
    entry_direction_encoding: Dict[str, float]
    source_repo_commit: str
    environment: Dict[str, str]
    tp_atr_multiple: Optional[float] = None
    sl_atr_multiple: Optional[float] = None
    return_threshold_atr: Optional[float] = None
    decision_threshold: Optional[float] = None
    extra: Dict[str, Any] = None

    @property
    def risk_unit(self) -> str:
        """What 1R means for this model. Never assumed by a caller."""
        if self.target_kind == KIND_BARRIER:
            return f"SL distance = {self.sl_atr_multiple} x ATR at entry"
        return "1 ATR at entry (hold-to-horizon trade, no stop)"

    @property
    def feature_count(self) -> int:
        return len(self.feature_list)

    def describe(self) -> str:
        return (f"{self.model_id} ({self.symbol}/{self.timeframe}, "
                f"schema={self.feature_schema_version}, {self.feature_count} features, "
                f"target={self.target_kind}, verdict={self.research_verdict}, "
                f"hash={self.model_hash[:12]})")

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["feature_list"] = list(self.feature_list)
        d["context_timeframes"] = list(self.context_timeframes)
        return d


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse(raw: Dict[str, Any]) -> ModelCard:
    """Validate a card dict into a ModelCard, or raise. No field is defaulted."""
    if not isinstance(raw, dict):
        raise ModelCardError(f"model card must be a JSON object, got {type(raw).__name__}")
    missing = [k for k in REQUIRED_FIELDS if k not in raw]
    if missing:
        raise ModelCardError(f"model card is missing required field(s): {missing}")

    features = raw["feature_list"]
    if not isinstance(features, list) or not features:
        raise ModelCardError(f"feature_list must be a non-empty list, got {features!r}")
    if not all(isinstance(f, str) and f.strip() for f in features):
        raise ModelCardError("feature_list must contain only non-empty strings")
    if len(set(features)) != len(features):
        dupes = sorted({f for f in features if features.count(f) > 1})
        raise ModelCardError(f"feature_list contains duplicates: {dupes}")

    horizon = raw["horizon_bars"]
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ModelCardError(f"horizon_bars must be a positive int, got {horizon!r}")

    kind = raw.get("target_kind")
    if kind not in (KIND_BARRIER, KIND_RETURN):
        raise ModelCardError(
            f"target_kind must be {KIND_BARRIER!r} or {KIND_RETURN!r}, got {kind!r}. "
            f"The kind decides what p_win means and what 1R is; it cannot be defaulted.")

    if kind == KIND_BARRIER:
        missing_geom = [k for k in REQUIRED_BARRIER_FIELDS if k not in raw]
        if missing_geom:
            raise ModelCardError(
                f"a barrier model must declare its geometry; missing {missing_geom}")
        for key in REQUIRED_BARRIER_FIELDS:
            v = raw[key]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
                raise ModelCardError(f"{key} must be a positive number, got {v!r}")
    else:
        thr = raw.get("return_threshold_atr")
        if not isinstance(thr, (int, float)) or isinstance(thr, bool) or thr < 0:
            raise ModelCardError(
                f"a return model must declare return_threshold_atr (>= 0), got {thr!r}")

    for key in ("model_id", "symbol", "timeframe", "model_version",
                "feature_schema_version", "target", "target_kind",
                "training_dataset_hash",
                "feature_manifest_hash", "target_spec_hash", "model_hash",
                "probability_semantics", "research_verdict", "calibration",
                "source_repo_commit"):
        v = raw[key]
        if not isinstance(v, str) or not v.strip():
            raise ModelCardError(f"{key} must be a non-empty string, got {v!r}")

    expected = PROBABILITY_SEMANTICS_BY_KIND[kind]
    if raw["probability_semantics"] != expected:
        raise ModelCardError(
            f"probability_semantics is {raw['probability_semantics']!r}, but a "
            f"{kind!r} model emits {expected!r}. Same number, different meaning is "
            f"the drift this field exists to stop.")

    threshold = raw.get("decision_threshold")
    if threshold is not None:
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise ModelCardError(f"decision_threshold must be numeric, got {threshold!r}")
        if not 0.0 < float(threshold) < 1.0:
            raise ModelCardError(f"decision_threshold must lie in (0,1), got {threshold!r}")

    known = (set(REQUIRED_FIELDS) | set(REQUIRED_BARRIER_FIELDS)
             | {"decision_threshold", "return_threshold_atr"})
    return ModelCard(
        model_id=raw["model_id"], symbol=raw["symbol"], timeframe=raw["timeframe"],
        model_version=raw["model_version"],
        feature_schema_version=raw["feature_schema_version"],
        feature_list=tuple(features), target=raw["target"], target_kind=kind,
        horizon_bars=horizon,
        tp_atr_multiple=(float(raw["tp_atr_multiple"])
                         if raw.get("tp_atr_multiple") is not None else None),
        sl_atr_multiple=(float(raw["sl_atr_multiple"])
                         if raw.get("sl_atr_multiple") is not None else None),
        return_threshold_atr=(float(raw["return_threshold_atr"])
                              if raw.get("return_threshold_atr") is not None else None),
        training_dataset_hash=raw["training_dataset_hash"],
        feature_manifest_hash=raw["feature_manifest_hash"],
        target_spec_hash=raw["target_spec_hash"], model_hash=raw["model_hash"],
        probability_semantics=raw["probability_semantics"],
        research_verdict=raw["research_verdict"], calibration=raw["calibration"],
        context_timeframes=tuple(raw["context_timeframes"]),
        entry_direction_encoding=dict(raw["entry_direction_encoding"]),
        source_repo_commit=raw["source_repo_commit"],
        environment=dict(raw["environment"]),
        decision_threshold=None if threshold is None else float(threshold),
        extra={k: v for k, v in raw.items() if k not in known},
    )


def load(directory) -> ModelCard:
    """Read and validate the card sitting beside a model artifact."""
    path = Path(directory) / CARD_FILENAME
    if not path.exists():
        raise ModelCardError(
            f"no model card at {path}. A model without declared provenance cannot be "
            f"served; re-import it with scripts/import_research_model.py.")
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ModelCardError(f"{path} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ModelCardError(f"{path} is unreadable: {exc}") from exc
    return parse(raw)


def write(directory, payload: Dict[str, Any]) -> Path:
    """Write a card, validating it first so an unloadable one is never emitted."""
    parse(payload)
    path = Path(directory) / CARD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path
