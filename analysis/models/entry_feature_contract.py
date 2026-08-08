"""Authoritative feature contract for the entry model.

Why this module exists
----------------------
The deployed artifact ``models/entry/entry_model.json`` expects **65** features
(the entry_v2 schema: H4/H1 indicators, lags, deltas, interactions, time
encodings). Live inference in ``xgboost_v2_inference.predict_with_v2`` supplies
**10** scalars in a completely different order.

XGBoost does not reject that. It treats the 55 absent columns as ``missing``
and follows each tree's default branch, returning a plausible-looking number.
Measured consequences on the deployed artifact:

* changing ``macd``, ``trend_score``, ``momentum_score``, ``volatility_score``
  or ``market_regime`` does not move ``p_win`` at all
* changing ``direction`` does not move it either — BUY and SELL receive the
  identical probability

So every entry decision has been gated on a number with no relationship to the
trade being evaluated.

What this module does
---------------------
It is the single place that answers "may this feature vector be fed to this
model?", and it answers conservatively. Validation happens **before** predict,
never after. On any mismatch the caller receives ``ML_GATE_INVALID`` and must
reject the trade.

Deliberately NOT done here (per remediation constraints): zero padding,
positional guessing, truncation, or synthesising the missing 55 features. A
model whose contract cannot be satisfied must block trading, not be coaxed into
producing output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from utils.logger import get_logger

logger = get_logger("entry_feature_contract")


# --- gate statuses -----------------------------------------------------------
STATUS_OK = "OK"
STATUS_INVALID = "ML_GATE_INVALID"
STATUS_MODEL_MISSING = "ML_MODEL_MISSING"
STATUS_PREDICTION_ERROR = "ML_PREDICTION_ERROR"

# Statuses that permit a trade to proceed. Everything else blocks.
PASSING_STATUSES = frozenset({STATUS_OK})


@dataclass(frozen=True)
class FeatureContract:
    """What a model requires of its input vector."""

    model_version: str
    feature_count: int
    feature_names: tuple = ()
    preprocessing_version: str = "v1"
    timeframes: tuple = ()

    def describe(self) -> str:
        return (
            f"{self.model_version}: {self.feature_count} features"
            + (f", named={list(self.feature_names[:5])}..." if self.feature_names else ", unnamed")
        )


@dataclass(frozen=True)
class GateResult:
    """Outcome of the ML entry gate.

    ``p_win`` is None whenever ``status != OK``. Callers must branch on
    ``allowed`` (or ``status``), never on ``p_win`` alone — a numeric default
    would be indistinguishable from a real low probability.
    """

    status: str
    p_win: Optional[float] = None
    reason: str = ""
    contract: Optional[FeatureContract] = None
    meta: dict = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        """True only when the model produced a trustworthy probability."""
        return self.status in PASSING_STATUSES and self.p_win is not None

    @property
    def available(self) -> bool:
        """Backwards-compatible alias used by main.py's decision gate."""
        return self.allowed


def contract_from_booster(booster: Any, model_version: str) -> FeatureContract:
    """Read the contract the loaded artifact actually enforces."""
    try:
        count = int(booster.num_features())
    except Exception as exc:
        logger.error("[ML_CONTRACT] cannot read num_features: %s", exc)
        count = -1

    names = getattr(booster, "feature_names", None) or ()
    return FeatureContract(
        model_version=model_version,
        feature_count=count,
        feature_names=tuple(names),
    )


def validate_features(
    features: Sequence[Any],
    contract: FeatureContract,
    supplied_names: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Check a vector against a contract.

    Returns None when the vector is acceptable, otherwise a human-readable
    reason. The caller must not predict when a reason is returned.
    """
    if contract.feature_count < 0:
        return "model feature count could not be determined"

    supplied = len(features)
    if supplied != contract.feature_count:
        return (
            f"feature count mismatch: model expects {contract.feature_count}, "
            f"got {supplied}"
        )

    # Name/order check, only possible when the artifact carries names.
    if contract.feature_names and supplied_names is not None:
        if tuple(supplied_names) != tuple(contract.feature_names):
            expected = list(contract.feature_names)
            got = list(supplied_names)
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(expected, got)) if a != b),
                min(len(expected), len(got)),
            )
            return (
                f"feature name/order mismatch at index {first_diff}: "
                f"expected {expected[first_diff:first_diff + 1]}, "
                f"got {got[first_diff:first_diff + 1]}"
            )

    # Numeric sanity. A NaN or Inf reaching the model produces an unusable
    # probability that would still look like a number downstream.
    for index, value in enumerate(features):
        name = (
            contract.feature_names[index]
            if index < len(contract.feature_names)
            else (supplied_names[index] if supplied_names and index < len(supplied_names) else f"#{index}")
        )
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return f"feature {name} is not numeric: {value!r}"
        if math.isnan(numeric):
            return f"feature {name} is NaN"
        if math.isinf(numeric):
            return f"feature {name} is infinite"

    return None


def validate_probability(value: Any) -> Optional[str]:
    """Reject a model output that cannot be used as a probability."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return f"prediction is not numeric: {value!r}"
    if math.isnan(numeric):
        return "prediction is NaN"
    if math.isinf(numeric):
        return "prediction is infinite"
    if not (0.0 <= numeric <= 1.0):
        return f"prediction {numeric} outside [0, 1]"
    return None


def invalid(reason: str, contract: Optional[FeatureContract] = None) -> GateResult:
    """Build a blocking result and log it loudly — this stops a trade."""
    logger.error("[ML_GATE] %s — entry BLOCKED. reason=%s", STATUS_INVALID, reason)
    return GateResult(status=STATUS_INVALID, p_win=None, reason=reason, contract=contract)


def model_missing(reason: str) -> GateResult:
    logger.error("[ML_GATE] %s — entry BLOCKED. reason=%s", STATUS_MODEL_MISSING, reason)
    return GateResult(status=STATUS_MODEL_MISSING, p_win=None, reason=reason)


def prediction_error(reason: str, contract: Optional[FeatureContract] = None) -> GateResult:
    logger.error("[ML_GATE] %s — entry BLOCKED. reason=%s", STATUS_PREDICTION_ERROR, reason)
    return GateResult(
        status=STATUS_PREDICTION_ERROR, p_win=None, reason=reason, contract=contract
    )


def ok(p_win: float, contract: FeatureContract, meta: Optional[dict] = None) -> GateResult:
    return GateResult(
        status=STATUS_OK, p_win=float(p_win), reason="", contract=contract, meta=meta or {}
    )
