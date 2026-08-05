"""Layer 2b support - ML exit probability provider.

Recomputes the exit probability every candle by rebuilding the same features
used at entry and passing them through the exit model.

Two things this module is careful about:

1. **Qualification.** A single probability reading is never treated as a strong
   signal. It only carries full weight once it has declined for N consecutive
   candles, or has dropped more than X% from its value at entry. Otherwise the
   contribution is damped. That rule lives here, next to the data it needs.

2. **Gating.** Production keeps the model off (``ML_EXIT_ENABLED=False``).
   Shadow mode computes and logs the probability but returns
   ``influences_decision=False`` so the scorer ignores it — this is how the
   model earns trust before it is allowed to matter.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

from utils.logger import get_logger

from . import tm_config as C

logger = get_logger("tm.exit_probability")

_model_cache: Dict[str, Any] = {"loaded": False, "model": None, "kind": None}


@dataclass
class ProbabilityHistory:
    """Per-trade probability track, appended once per closed candle."""

    entry_probability: Optional[float] = None
    readings: Deque[float] = field(default_factory=lambda: deque(maxlen=32))

    def record(self, value: float) -> None:
        if self.entry_probability is None:
            self.entry_probability = float(value)
        self.readings.append(float(value))

    @property
    def latest(self) -> Optional[float]:
        return self.readings[-1] if self.readings else None

    def consecutive_declines(self) -> int:
        """How many candles in a row the probability has fallen."""
        count = 0
        for newer, older in zip(list(self.readings)[::-1], list(self.readings)[-2::-1]):
            if newer < older:
                count += 1
            else:
                break
        return count

    def drop_from_entry(self) -> float:
        """Fractional drop since entry; 0.0 when unknown or improved."""
        if self.entry_probability in (None, 0) or self.latest is None:
            return 0.0
        drop = (self.entry_probability - self.latest) / abs(self.entry_probability)
        return max(0.0, drop)


@dataclass(frozen=True)
class ProbabilityAssessment:
    """What Layer 2b needs to know about the model's current opinion."""

    available: bool
    probability: Optional[float]
    qualified: bool
    weight_multiplier: float
    influences_decision: bool
    reason: str
    consecutive_declines: int = 0
    drop_from_entry: float = 0.0


def _load_model():
    """Load the exit model via the sklearn wrapper so predict_proba is available.

    The model is trained with XGBClassifier but persisted in XGBoost's native
    JSON format. Loading it back through XGBClassifier restores the sklearn API
    (predict_proba) over exactly the same booster.
    """
    if _model_cache["loaded"]:
        return _model_cache["model"]

    _model_cache["loaded"] = True
    path = C.ML_EXIT_MODEL_PATH
    try:
        import os

        if not os.path.exists(path):
            logger.warning("[TM_L2_PROB] exit model not found at %s", path)
            _model_cache["model"] = None
            return None

        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(path)
        _model_cache["model"] = model
        _model_cache["kind"] = "sklearn"
        logger.info("[TM_L2_PROB] exit model loaded (sklearn wrapper) from %s", path)
        return model
    except Exception as exc:
        logger.error("[TM_L2_PROB] failed to load exit model: %s", exc)
        _model_cache["model"] = None
        return None


def _build_features(features: Dict[str, Any]):
    """Order the feature dict into the vector the model expects."""
    from analysis.models.feature_schema import FEATURE_ORDER, build_feature_vector

    vector = build_feature_vector(features)
    return [vector], FEATURE_ORDER


def compute_exit_probability(
    features: Dict[str, Any],
    history: ProbabilityHistory,
    settings: Optional[dict] = None,
) -> ProbabilityAssessment:
    """Score the current candle and judge whether the reading is actionable."""
    settings = settings or {}

    enabled = bool(settings.get("ML_EXIT_ENABLED", C.ML_EXIT_ENABLED))
    shadow = bool(settings.get("ML_EXIT_SHADOW_MODE", C.ML_EXIT_SHADOW_MODE))

    if not enabled and not shadow:
        return ProbabilityAssessment(
            available=False, probability=None, qualified=False,
            weight_multiplier=0.0, influences_decision=False,
            reason="ml_exit_disabled",
        )

    model = _load_model()
    if model is None:
        return ProbabilityAssessment(
            available=False, probability=None, qualified=False,
            weight_multiplier=0.0, influences_decision=False,
            reason="model_unavailable",
        )

    try:
        rows, _order = _build_features(features)
        # predict_proba returns [[p_class0, p_class1]]; class 1 is "bad exit".
        probability = float(model.predict_proba(rows)[0][1])
        probability = max(0.0, min(1.0, probability))
    except Exception as exc:
        logger.error("[TM_L2_PROB] prediction failed: %s", exc)
        return ProbabilityAssessment(
            available=False, probability=None, qualified=False,
            weight_multiplier=0.0, influences_decision=False,
            reason=f"prediction_error:{type(exc).__name__}",
        )

    history.record(probability)

    min_declines = int(settings.get("PROB_MIN_CONSECUTIVE_DECLINES", C.PROB_MIN_CONSECUTIVE_DECLINES))
    drop_threshold = float(settings.get("PROB_DROP_FROM_ENTRY_PCT", C.PROB_DROP_FROM_ENTRY_PCT))
    damping = float(settings.get("PROB_UNQUALIFIED_DAMPING", C.PROB_UNQUALIFIED_DAMPING))

    declines = history.consecutive_declines()
    drop = history.drop_from_entry()
    qualified = declines >= min_declines or drop > drop_threshold

    if shadow and not enabled:
        logger.info(
            "[TM_L2_PROB][SHADOW] p=%.4f declines=%d drop=%.1f%% qualified=%s "
            "(no effect on decisions)",
            probability, declines, drop * 100.0, qualified,
        )
        return ProbabilityAssessment(
            available=True, probability=probability, qualified=qualified,
            weight_multiplier=0.0, influences_decision=False,
            reason="shadow_mode",
            consecutive_declines=declines, drop_from_entry=drop,
        )

    return ProbabilityAssessment(
        available=True,
        probability=probability,
        qualified=qualified,
        weight_multiplier=1.0 if qualified else damping,
        influences_decision=True,
        reason="qualified" if qualified else "unqualified_damped",
        consecutive_declines=declines,
        drop_from_entry=drop,
    )


def reset_model_cache() -> None:
    """Test hook: force the next call to reload the model."""
    _model_cache.update({"loaded": False, "model": None, "kind": None})
