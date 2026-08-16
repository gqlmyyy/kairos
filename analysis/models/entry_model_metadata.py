"""What a model file must declare about itself before it may be trusted.

An XGBoost JSON artifact carries almost nothing about its own provenance. The
deployed one records `num_feature: 65` and an empty `feature_names` list — no
schema version, no label definition, no dataset, no training window. That is
how four different trainers came to write three different schemas to one
filename without anything noticing.

So the artifact alone is not sufficient evidence. A model is loadable only when
a sidecar `<model>.metadata.json` sits next to it and every declared field
agrees with both the booster and the live feature contract.

The rule is FAIL CLOSED. Absent metadata, unreadable metadata, a missing field,
a version the live path does not implement, or names that differ from the live
spec in *order* as well as in membership — every one of these blocks the model.
No defaults are invented. A trading system that cannot prove which schema its
model speaks must not trade.

Note what this deliberately does not do: it never edits, upgrades or backfills
a metadata file to make a model loadable. Repairing provenance after the fact
is indistinguishable from fabricating it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.logger import get_logger

logger = get_logger("entry_model_metadata")

METADATA_SUFFIX = ".metadata.json"

# Bumped when the *meaning* of a feature changes. A model built against an
# older definition must not be served by a newer live path even when the names
# and order still line up — same name, different arithmetic is the drift that
# is hardest to see and easiest to trade on.
CURRENT_FEATURE_SCHEMA_VERSION = "entry-2"
CURRENT_FEATURE_DEFINITION_VERSION = "live-parity-1"
CURRENT_LABEL_SCHEMA_VERSION = "barrier-1"

# Every field must be present. Splitting them only documents intent; the
# loader requires the union.
REQUIRED_IDENTITY = (
    "model_version",
    "training_pipeline_id",
    "feature_schema_version",
    "feature_definition_version",
    "label_schema_version",
)
REQUIRED_FEATURES = (
    "feature_names",
    "feature_order",
    "feature_count",
)
REQUIRED_DATA = (
    "dataset_fingerprint",
    "training_start",
    "training_end",
    "validation_start",
    "validation_end",
    "test_start",
    "test_end",
    "symbol_scope",
    "timeframe_scope",
)
REQUIRED_TARGET = (
    "target_definition",
    "target_horizon",
)
REQUIRED_BUILD = (
    "xgboost_version",
    "python_version",
    "git_commit",
)
REQUIRED_FIELDS: Tuple[str, ...] = (
    REQUIRED_IDENTITY + REQUIRED_FEATURES + REQUIRED_DATA
    + REQUIRED_TARGET + REQUIRED_BUILD
)


class MetadataError(Exception):
    """Raised when a model cannot prove what it is."""


@dataclass(frozen=True)
class ModelMetadata:
    """The declared identity of a model artifact."""

    model_version: str
    training_pipeline_id: str
    feature_schema_version: str
    feature_definition_version: str
    label_schema_version: str
    feature_names: Tuple[str, ...]
    feature_order: Tuple[int, ...]
    feature_count: int
    dataset_fingerprint: str
    training_start: str
    training_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    symbol_scope: Tuple[str, ...]
    timeframe_scope: Tuple[str, ...]
    target_definition: str
    target_horizon: int
    xgboost_version: str
    python_version: str
    git_commit: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"{self.model_version} (pipeline={self.training_pipeline_id}, "
            f"schema={self.feature_schema_version}/{self.feature_definition_version}, "
            f"label={self.label_schema_version}, {self.feature_count} features, "
            f"dataset={self.dataset_fingerprint[:12]})"
        )


def metadata_path_for(model_path: str) -> str:
    return model_path + METADATA_SUFFIX


def _require_str(raw: Dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MetadataError(f"field {key!r} must be a non-empty string, got {value!r}")
    return value


def _require_str_tuple(raw: Dict[str, Any], key: str) -> Tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise MetadataError(f"field {key!r} must be a non-empty list, got {value!r}")
    if not all(isinstance(v, str) and v.strip() for v in value):
        raise MetadataError(f"field {key!r} must contain only non-empty strings")
    return tuple(value)


def parse(raw: Dict[str, Any]) -> ModelMetadata:
    """Turn a metadata dict into a validated ModelMetadata, or raise."""
    if not isinstance(raw, dict):
        raise MetadataError(f"metadata must be a JSON object, got {type(raw).__name__}")

    missing = [k for k in REQUIRED_FIELDS if k not in raw]
    if missing:
        raise MetadataError(f"missing required field(s): {missing}")

    names = _require_str_tuple(raw, "feature_names")

    order = raw.get("feature_order")
    if not isinstance(order, list) or not all(isinstance(v, int) for v in order):
        raise MetadataError(f"feature_order must be a list of ints, got {order!r}")
    order_t = tuple(order)

    count = raw.get("feature_count")
    if not isinstance(count, int) or isinstance(count, bool):
        raise MetadataError(f"feature_count must be an int, got {count!r}")

    # The three feature fields must agree with each other before anything else
    # is compared against them.
    if len(names) != count:
        raise MetadataError(
            f"feature_count {count} disagrees with {len(names)} feature_names")
    if len(order_t) != count:
        raise MetadataError(
            f"feature_count {count} disagrees with {len(order_t)} feature_order entries")
    if sorted(order_t) != list(range(count)):
        raise MetadataError(
            f"feature_order must be a permutation of 0..{count - 1}, got {list(order_t)}")
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise MetadataError(f"feature_names contains duplicates: {dupes}")

    horizon = raw.get("target_horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise MetadataError(f"target_horizon must be a positive int, got {horizon!r}")

    known = set(REQUIRED_FIELDS)
    return ModelMetadata(
        model_version=_require_str(raw, "model_version"),
        training_pipeline_id=_require_str(raw, "training_pipeline_id"),
        feature_schema_version=_require_str(raw, "feature_schema_version"),
        feature_definition_version=_require_str(raw, "feature_definition_version"),
        label_schema_version=_require_str(raw, "label_schema_version"),
        feature_names=names,
        feature_order=order_t,
        feature_count=count,
        dataset_fingerprint=_require_str(raw, "dataset_fingerprint"),
        training_start=_require_str(raw, "training_start"),
        training_end=_require_str(raw, "training_end"),
        validation_start=_require_str(raw, "validation_start"),
        validation_end=_require_str(raw, "validation_end"),
        test_start=_require_str(raw, "test_start"),
        test_end=_require_str(raw, "test_end"),
        symbol_scope=_require_str_tuple(raw, "symbol_scope"),
        timeframe_scope=_require_str_tuple(raw, "timeframe_scope"),
        target_definition=_require_str(raw, "target_definition"),
        target_horizon=horizon,
        xgboost_version=_require_str(raw, "xgboost_version"),
        python_version=_require_str(raw, "python_version"),
        git_commit=_require_str(raw, "git_commit"),
        extra={k: v for k, v in raw.items() if k not in known},
    )


def load(model_path: str) -> ModelMetadata:
    """Read and validate the sidecar for a model. Raises MetadataError."""
    path = metadata_path_for(model_path)
    if not os.path.exists(path):
        raise MetadataError(
            f"no metadata sidecar at {path}. A model without declared provenance "
            f"cannot be served; regenerate it with the training pipeline.")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise MetadataError(f"metadata at {path} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise MetadataError(f"metadata at {path} is unreadable: {exc}") from exc
    return parse(raw)


def validate_against_booster(meta: ModelMetadata, booster: Any) -> Optional[str]:
    """Does the artifact agree with what the metadata claims about it?"""
    try:
        actual = int(booster.num_features())
    except Exception as exc:  # noqa: BLE001 - any failure here must block
        return f"cannot read num_features from booster: {exc}"

    if actual != meta.feature_count:
        return (f"metadata declares {meta.feature_count} features but the model "
                f"artifact expects {actual}")

    # When the artifact does carry names they are authoritative over the
    # sidecar, since they are what XGBoost itself will match on.
    embedded = tuple(getattr(booster, "feature_names", None) or ())
    if embedded and embedded != meta.feature_names:
        return (f"model artifact feature_names differ from metadata: "
                f"artifact={list(embedded[:4])}... metadata={list(meta.feature_names[:4])}...")
    return None


def validate_for_serving(
    meta: ModelMetadata,
    *,
    live_feature_names: Sequence[str],
    feature_schema_version: str = CURRENT_FEATURE_SCHEMA_VERSION,
    feature_definition_version: str = CURRENT_FEATURE_DEFINITION_VERSION,
    label_schema_version: str = CURRENT_LABEL_SCHEMA_VERSION,
) -> Optional[str]:
    """May the live path send its vector to this model?

    Order matters as much as membership: the same ten names permuted is a
    different model input, and XGBoost would accept it silently.
    """
    if meta.feature_schema_version != feature_schema_version:
        return (f"feature schema version mismatch: model was built for "
                f"{meta.feature_schema_version!r}, live path implements "
                f"{feature_schema_version!r}")

    if meta.feature_definition_version != feature_definition_version:
        return (f"feature definition version mismatch: model was built for "
                f"{meta.feature_definition_version!r}, live path implements "
                f"{feature_definition_version!r}")

    if meta.label_schema_version != label_schema_version:
        return (f"label schema version mismatch: model predicts "
                f"{meta.label_schema_version!r}, live path expects "
                f"{label_schema_version!r}")

    live = tuple(live_feature_names)
    if live != meta.feature_names:
        if set(live) == set(meta.feature_names):
            first = next(i for i, (a, b) in enumerate(zip(meta.feature_names, live))
                         if a != b)
            return (f"feature order mismatch at index {first}: model expects "
                    f"{meta.feature_names[first]!r}, live sends {live[first]!r} "
                    f"(same names, different order)")
        only_model = [n for n in meta.feature_names if n not in set(live)]
        only_live = [n for n in live if n not in set(meta.feature_names)]
        return (f"feature name mismatch: model-only={only_model[:5]}, "
                f"live-only={only_live[:5]}")

    return None


def build(
    *,
    model_version: str,
    training_pipeline_id: str,
    feature_names: Sequence[str],
    dataset_fingerprint: str,
    training_start: str,
    training_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
    symbol_scope: Sequence[str],
    timeframe_scope: Sequence[str],
    target_definition: str,
    target_horizon: int,
    git_commit: str,
    feature_schema_version: str = CURRENT_FEATURE_SCHEMA_VERSION,
    feature_definition_version: str = CURRENT_FEATURE_DEFINITION_VERSION,
    label_schema_version: str = CURRENT_LABEL_SCHEMA_VERSION,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a metadata dict for a freshly trained model.

    Build-environment fields are read from the running interpreter rather than
    accepted as arguments, so they cannot be misreported by a caller.
    """
    import platform
    import sys as _sys

    try:
        import xgboost as _xgb
        xgb_version = str(_xgb.__version__)
    except Exception:  # noqa: BLE001
        xgb_version = "unavailable"

    names = list(feature_names)
    payload: Dict[str, Any] = {
        "model_version": model_version,
        "training_pipeline_id": training_pipeline_id,
        "feature_schema_version": feature_schema_version,
        "feature_definition_version": feature_definition_version,
        "label_schema_version": label_schema_version,
        "feature_names": names,
        "feature_order": list(range(len(names))),
        "feature_count": len(names),
        "dataset_fingerprint": dataset_fingerprint,
        "training_start": training_start,
        "training_end": training_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "test_start": test_start,
        "test_end": test_end,
        "symbol_scope": list(symbol_scope),
        "timeframe_scope": list(timeframe_scope),
        "target_definition": target_definition,
        "target_horizon": int(target_horizon),
        "xgboost_version": xgb_version,
        "python_version": platform.python_version() or _sys.version.split()[0],
        "git_commit": git_commit,
    }
    if extra:
        payload.update(extra)
    parse(payload)  # refuse to emit something the loader would reject
    return payload


def write(model_path: str, payload: Dict[str, Any]) -> str:
    """Write a sidecar next to a model, validating it first."""
    parse(payload)
    path = metadata_path_for(model_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    logger.info("[ML_CONTRACT] wrote metadata sidecar %s", path)
    return path
