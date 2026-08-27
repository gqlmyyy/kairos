"""The one entry point from the live loop into the research models.

    main.py
      -> live_gate.predict_entry(symbol, timeframe, row, entry_direction)
      -> model_registry.resolve(..., statuses=(PRODUCTION_ELIGIBLE,), version="research_v2")
      -> model_loader.load_model()        card + artifact hash + contract verified
      -> inference.predict_row()          canonical feature vector, then score
      -> dict(status, available, p_win, reason, ...)

Two independent conditions, both required
-----------------------------------------
`production_gate` is explicit that eligibility is a statement about evidence,
not a switch: *"turning a model on remains a separate, human, out-of-band
act"*, and `tests/test_research_production_gate.py::test_eligibility_is_not_activation`
enforces it. So this module treats the registry status as **necessary and
not sufficient**:

1. **Eligible** — the registry entry is ``PRODUCTION_ELIGIBLE`` at generation
   ``research_v2``. This is `production_gate.promote()`'s output and cannot be
   reached without the evidence it demands.
2. **Activated** — `activation.json` names this exact ``model_id`` AND its
   exact ``model_hash``. This is the separate human act. Binding it to the
   hash is deliberate, and mirrors why the approval record is a file: if the
   artifact changes by one byte the activation stops applying to it, which a
   boolean flag could never express.

Neither alone serves a model. An eligible-but-not-activated model blocks; an
activated-but-not-eligible model blocks. That is strictly stronger than
gating on status alone, and it is why this module may reference the status at
all without turning it into an enable path.

Exactly one source
------------------
`VERSION` pins the generation to ``research_v2`` and `REQUIRED_STATUSES` pins
the status. Both are module constants rather than parameters with defaults,
because a caller that could pass ``statuses=SERVABLE_STATUSES`` would be able
to serve a RESEARCH-grade model into a live account by accident. There is no
argument that widens either one.

Consequences that are deliberate, not oversights:

* **No legacy path.** ``models/entry/entry_model.json`` is never consulted.
  Not as a fallback, not on error, not when the registry is empty.
* **No other generation.** ``research_v3`` is registered but is not
  ``research_v2``, so it is not served here. Promoting a generation is an
  explicit edit to `VERSION`, reviewed like any other change.
* **No status widening.** Every shipped model is currently ``RESEARCH`` (17)
  or ``CANDIDATE`` (1), and no ``activation.json`` exists, so *every* call
  through this module blocks today on both counts.

Feature availability is checked, never assumed
----------------------------------------------
The shipped cards ask for 83 canonical features built from multi-timeframe
OHLC frames (see `analysis/research/engine.py`). A caller that supplies a
partial row does NOT get a padded vector — `inference.build_feature_vector`
reports the missing columns and this returns MODEL_NOT_COMPATIBLE naming
them. Sending a short vector to a wide model is the exact defect that made
the legacy 65-vs-10 path produce confident numbers unrelated to the trade.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from analysis.research import inference as inf
from analysis.research import model_loader as loader
from analysis.research import model_registry as reg
from utils.logger import get_logger

logger = get_logger("research_live_gate")

# The single sanctioned generation and the single sanctioned status.
VERSION = "research_v2"
REQUIRED_STATUSES = (reg.PRODUCTION_ELIGIBLE,)

# The separate, human, out-of-band act. Lives beside the registry.
ACTIVATION_FILENAME = "activation.json"
REQUIRED_ACTIVATION_FIELDS = ("model_id", "model_hash", "activated_by", "activated_at_utc")

# Blocked-before-scoring statuses. Distinct from inference's own statuses so a
# reader can tell "never resolved a model" from "resolved one and the features
# were wrong".
STATUS_MODEL_MISSING = "ML_MODEL_MISSING"
STATUS_NOT_ACTIVATED = "MODEL_NOT_ACTIVATED"
STATUS_NOT_COMPATIBLE = inf.STATUS_NOT_COMPATIBLE

_cache: Dict[tuple, loader.LoadedModel] = {}
_lock = threading.RLock()


def reset_cache() -> None:
    """Drop cached models. For tests, and after an operator promotes one."""
    with _lock:
        _cache.clear()


def activation_path(registry_path=None) -> Path:
    base = Path(registry_path) if registry_path is not None else Path(reg.DEFAULT_REGISTRY_PATH)
    return base.parent / ACTIVATION_FILENAME


def load_activations(registry_path=None) -> Dict[str, str]:
    """model_id -> model_hash for every activated model.

    A malformed or absent file activates nothing. It never raises: an
    unreadable activation file must block trading, not crash the loop, and
    "blocks everything" is already the safe direction.
    """
    path = activation_path(registry_path)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("[RESEARCH_GATE] activation file %s is unreadable (%s) — "
                     "nothing is activated", path, exc)
        return {}

    records = raw.get("activated") if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        logger.error("[RESEARCH_GATE] activation file %s has no 'activated' list — "
                     "nothing is activated", path)
        return {}

    out: Dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        missing = [f for f in REQUIRED_ACTIVATION_FIELDS
                   if not str(record.get(f) or "").strip()]
        if missing:
            logger.error("[RESEARCH_GATE] activation record %r is missing %s — ignored",
                         record.get("model_id", "?"), missing)
            continue
        out[str(record["model_id"])] = str(record["model_hash"]).lower()
    return out


def _blocked(status: str, reason: str, symbol: str, timeframe: str) -> Dict[str, Any]:
    logger.error("[RESEARCH_GATE] %s — entry BLOCKED. symbol=%s tf=%s reason=%s",
                 status, symbol, timeframe, reason)
    return {
        "status": status,
        "available": False,
        "p_win": None,
        "reason": reason,
        "model_id": "",
        "symbol": symbol,
        "timeframe": timeframe,
    }


class NotActivated(Exception):
    """Eligible, loaded, verified — but no human has activated it."""


def load_production_model(symbol: str, timeframe: str, *, registry_path=None):
    """Resolve, verify, and confirm activation. Both conditions or nothing.

    Raises `model_loader.ModelNotCompatible` for resolution/verification
    failures (unregistered, wrong status, missing artifact, hash mismatch,
    contract disagreement) and `NotActivated` when the model is eligible but
    has no matching activation record.
    """
    key = (symbol.upper(), timeframe.upper(), str(registry_path))
    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached

    kwargs: Dict[str, Any] = {"statuses": REQUIRED_STATUSES, "version": VERSION}
    if registry_path is not None:
        kwargs["registry_path"] = registry_path

    # Condition 1: eligible. Raises if not.
    model = loader.load_model(symbol, timeframe, **kwargs)

    # Condition 2: activated, and activated for THESE bytes.
    activations = load_activations(registry_path)
    model_id = model.card.model_id
    declared_hash = str(getattr(model.entry, "model_hash", "") or "").lower()
    activated_hash = activations.get(model_id)

    if activated_hash is None:
        raise NotActivated(
            f"{model_id} is {reg.PRODUCTION_ELIGIBLE} but not activated: no record "
            f"in {activation_path(registry_path)}. Eligibility is evidence, not "
            f"activation — turning a model on is a separate, deliberate act.")
    if declared_hash and activated_hash != declared_hash:
        raise NotActivated(
            f"{model_id} activation names hash {activated_hash[:16]}… but the "
            f"registered artifact is {declared_hash[:16]}… — the model changed "
            f"since it was activated, so the activation no longer applies to it.")

    logger.info("[VERIFY] RESEARCH MODEL LOADED %s activated=True", model.describe())
    with _lock:
        _cache[key] = model
    return model


def predict_entry(
    *,
    symbol: str,
    timeframe: str,
    row: Mapping[str, Any],
    entry_direction: Any,
    unavailable_columns: Sequence[str] = (),
    timestamp: Optional[Any] = None,
    registry_path=None,
) -> Dict[str, Any]:
    """Score one entry, or explain precisely why it cannot be scored.

    `row` must carry the canonical feature columns the model's card asks for.
    Anything short of that blocks — see the module docstring.
    """
    try:
        model = load_production_model(symbol, timeframe, registry_path=registry_path)
    except NotActivated as exc:
        return _blocked(STATUS_NOT_ACTIVATED, str(exc), symbol, timeframe)
    except loader.ModelNotCompatible as exc:
        # "no model registered ... with status in [...]" is an empty slot at
        # this status; anything else means a model exists but is unusable.
        message = str(exc)
        status = (STATUS_MODEL_MISSING if "no model registered" in message
                  else STATUS_NOT_COMPATIBLE)
        return _blocked(status, message, symbol, timeframe)
    except reg.RegistryError as exc:
        return _blocked(STATUS_MODEL_MISSING, str(exc), symbol, timeframe)

    prediction = inf.predict_row(
        model, row,
        entry_direction=entry_direction,
        unavailable_columns=unavailable_columns,
        timestamp=timestamp,
    )
    result = prediction.as_dict()

    if prediction.available:
        logger.info("[VERIFY] RESEARCH PREDICTION symbol=%s tf=%s model_id=%s "
                    "p_win=%.3f available=True",
                    model.symbol, model.timeframe, model.card.model_id, prediction.p_win)
    else:
        logger.error("[RESEARCH_GATE] %s — entry BLOCKED. symbol=%s tf=%s model_id=%s "
                     "reason=%s", prediction.status, model.symbol, model.timeframe,
                     model.card.model_id, prediction.reason)
    return result
