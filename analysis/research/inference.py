"""Contract-driven inference. The model's own feature list builds the vector.

What replaced what
------------------
``analysis/models/xgboost_v2_inference.predict_with_v2`` takes nine positional
scalars and assembles a ten-slot list. That signature IS the bug: it can only
ever describe one schema, so when the artifact wanted a different one the call
still type-checked and still returned a number. Widening the signature would
just move the same failure to a bigger number.

Here the direction is reversed:

    model card -> canonical contract -> named feature frame -> vector

A :class:`FeatureVector` is built BY NAME from the model's own ordered feature
list. There is no argument list to keep in sync, so there is nothing to drift.
Adding a feature to a model changes nothing in this module.

What is never done
------------------
No zero-padding, no truncation, no positional guessing, no "the names are
close enough". A feature that is not VALID makes the PREDICTION invalid — see
``availability`` for why substituting a plausible number is worse than
refusing.

``p_win`` semantics
-------------------
The returned probability is ``P(TP before SL | entry_direction)`` and is
labelled ``p_win`` everywhere. It is not P(price rises), not an expected
return, not a confidence. The model card carries the same statement and the
loader refuses any artifact declaring different semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.research import availability as av
from analysis.research import contract as C
from analysis.research.model_loader import LoadedModel, ModelNotCompatible

STATUS_OK = "OK"
STATUS_FEATURE_UNAVAILABLE = "FEATURE_UNAVAILABLE"
STATUS_FEATURE_MISSING = "FEATURE_MISSING"
STATUS_FEATURE_INVALID = "FEATURE_INVALID"
STATUS_PREDICTION_ERROR = "PREDICTION_ERROR"
STATUS_NOT_COMPATIBLE = "MODEL_NOT_COMPATIBLE"

#: The only status on which a caller may read `p_win`.
PASSING_STATUSES = frozenset({STATUS_OK})

LONG = 1.0
SHORT = -1.0
ENTRY_DIRECTION_ENCODING: Dict[str, float] = {"BUY": LONG, "long": LONG, "LONG": LONG,
                                              "SELL": SHORT, "short": SHORT, "SHORT": SHORT}


class FeatureVectorError(Exception):
    """A vector cannot be assembled against this contract."""


def encode_entry_direction(direction: Any) -> float:
    """Map a trade side onto the meta feature. No default.

    A default here would silently score the wrong side of the trade, and
    ``target=1`` means something different for each side, so an unrecognised
    direction is an error rather than a fallback to BUY.
    """
    if isinstance(direction, (int, float)) and not isinstance(direction, bool):
        v = float(direction)
        if v in (LONG, SHORT):
            return v
        raise FeatureVectorError(
            f"numeric entry_direction must be {LONG} (long) or {SHORT} (short), got {v}")
    try:
        return ENTRY_DIRECTION_ENCODING[str(direction).strip()]
    except KeyError:
        raise FeatureVectorError(
            f"unrecognised entry direction {direction!r}; known: "
            f"{sorted(set(ENTRY_DIRECTION_ENCODING))}") from None


@dataclass(frozen=True)
class FeatureVector:
    """One row of model input, in the model's own feature order."""

    names: Tuple[str, ...]
    state: av.VectorState
    timestamp: Optional[pd.Timestamp] = None
    entry_direction: Optional[float] = None

    @property
    def usable(self) -> bool:
        return self.state.usable

    def as_array(self) -> np.ndarray:
        if not self.usable:
            raise FeatureVectorError(
                f"refusing to materialise an unusable vector: {self.state.summary()}")
        return np.asarray([s.value for s in self.state.states], dtype=float).reshape(1, -1)

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {s.name: s.value for s in self.state.states}

    def describe(self) -> str:
        return f"{len(self.names)} features @ {self.timestamp}: {self.state.summary()}"


@dataclass(frozen=True)
class Prediction:
    """The outcome of one scoring attempt. Branch on `status`, never on `p_win`."""

    status: str
    p_win: Optional[float]
    reason: str = ""
    model_id: str = ""
    symbol: str = ""
    timeframe: str = ""
    timestamp: Optional[pd.Timestamp] = None
    entry_direction: Optional[float] = None
    raw_probability: Optional[float] = None
    threshold: Optional[float] = None
    semantics: str = "p_win = P(TP before SL | entry_direction)"

    @property
    def available(self) -> bool:
        return self.status in PASSING_STATUSES

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status, "available": self.available, "p_win": self.p_win,
            "raw_probability": self.raw_probability, "reason": self.reason,
            "model_id": self.model_id, "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": None if self.timestamp is None else str(self.timestamp),
            "entry_direction": self.entry_direction, "threshold": self.threshold,
            "semantics": self.semantics,
        }


def feature_warmed_up(model: LoadedModel, row: Mapping[str, Any]) -> Dict[str, bool]:
    """Per feature: has its OWN timeframe produced enough bars to define it?

    This is what separates MISSING from INVALID. A NaN 40 bars into a
    50-bar window is the schedule working; the same NaN 500 bars in means a
    zero denominator or a corrupt bar. Treating them alike would either hide
    real data corruption or reject a perfectly normal warm-up — and treating
    either as "use a default" is how a fabricated value reaches a model.

    A row with no bar counter for a feature's timeframe is treated as NOT
    warmed up: unknown position is not evidence of completeness.
    """
    from analysis.research import engine as _engine

    out: Dict[str, bool] = {}
    for spec in model.contract.specs:
        if spec.source == C.SOURCE_META:
            out[spec.name] = True
            continue
        column = _engine.bar_index_column(spec.timeframe, model.timeframe)
        seen = row.get(column)
        try:
            bars = -1 if seen is None else int(seen)
        except (TypeError, ValueError):
            bars = -1
        if bars < 0 or (isinstance(seen, float) and pd.isna(seen)):
            out[spec.name] = False
            continue
        # `bar_index` is 0-based, so `minimum_history` bars have preceded this
        # row once bar_index >= minimum_history.
        out[spec.name] = bars >= spec.minimum_history
    return out


def build_feature_vector(
    model: LoadedModel,
    row: Mapping[str, Any],
    *,
    entry_direction: Any = None,
    unavailable_columns: Sequence[str] = (),
    timestamp: Optional[pd.Timestamp] = None,
    warmed_up: Optional[bool] = None,
) -> FeatureVector:
    """Assemble the vector this model asks for, by name, in its order.

    ``unavailable_columns`` names RAW candle columns the source cannot supply
    (e.g. ``spread``). Every feature that reads one is marked UNAVAILABLE
    rather than being computed from a substitute.

    ``warmed_up`` is normally derived per feature from the row's own bar
    counters; pass a bool only to override that for a caller that already
    knows (the golden fixtures, which are built past every warm-up window).
    """
    names = model.feature_names
    unavailable = {
        s.name for s in model.contract.specs
        if any(col in set(unavailable_columns) for col in s.requires)
    }
    warm = ({n: bool(warmed_up) for n in names} if warmed_up is not None
            else feature_warmed_up(model, row))

    direction_value: Optional[float] = None
    values: List[Any] = []
    for name in names:
        if name == C.ENTRY_DIRECTION.name:
            if entry_direction is None:
                raise FeatureVectorError(
                    f"{model.card.model_id} requires `entry_direction`: the model scores "
                    f"P(TP before SL | direction), so a candidate with no side has no "
                    f"defined probability")
            direction_value = encode_entry_direction(entry_direction)
            values.append(direction_value)
            continue
        if name in unavailable:
            values.append(None)
            continue
        values.append(row.get(name))

    states = tuple(
        av.classify(n, v, available=n not in unavailable, warmed_up=warm.get(n, True))
        for n, v in zip(names, values)
    )
    return FeatureVector(names=tuple(names), state=av.VectorState(states),
                         timestamp=timestamp, entry_direction=direction_value)


def _status_for(state: av.VectorState) -> str:
    """The most structural failure wins: UNAVAILABLE > INVALID > MISSING.

    A source that cannot produce a column at all is a worse problem than a
    corrupt bar, which is worse than an unfinished warm-up window, and the
    status a caller sees should name the one worth acting on.
    """
    kinds = {s.state for s in state.offenders()}
    if av.UNAVAILABLE in kinds:
        return STATUS_FEATURE_UNAVAILABLE
    if av.INVALID in kinds:
        return STATUS_FEATURE_INVALID
    return STATUS_FEATURE_MISSING


def predict(model: LoadedModel, vector: FeatureVector) -> Prediction:
    """Score one vector. Returns a Prediction; never raises for bad input."""
    base = dict(model_id=model.card.model_id, symbol=model.symbol,
                timeframe=model.timeframe, timestamp=vector.timestamp,
                entry_direction=vector.entry_direction,
                threshold=model.card.decision_threshold)

    if tuple(vector.names) != model.feature_names:
        return Prediction(status=STATUS_NOT_COMPATIBLE, p_win=None,
                          reason="vector was built against a different feature list",
                          **base)
    if not vector.usable:
        return Prediction(status=_status_for(vector.state), p_win=None,
                          reason=vector.state.summary(), **base)

    try:
        # A named DataFrame, not a bare array: sklearn and XGBoost both match
        # on names when the estimator carries them, so passing names removes
        # any chance of a silent positional reinterpretation.
        frame = pd.DataFrame(vector.as_array(), columns=list(model.feature_names))
        raw = float(model.estimator.predict_proba(frame)[0, 1])
    except Exception as exc:  # noqa: BLE001 - report, never substitute
        return Prediction(status=STATUS_PREDICTION_ERROR, p_win=None,
                          reason=f"{type(exc).__name__}: {exc}", **base)

    p_win = raw
    if model.calibrator is not None:
        try:
            p_win = float(np.asarray(model.calibrator.predict([raw])).ravel()[0])
        except Exception as exc:  # noqa: BLE001
            return Prediction(status=STATUS_PREDICTION_ERROR, p_win=None,
                              reason=f"calibrator failed: {type(exc).__name__}: {exc}",
                              raw_probability=raw, **base)

    bad = _validate_probability(p_win)
    if bad is not None:
        return Prediction(status=STATUS_PREDICTION_ERROR, p_win=None, reason=bad,
                          raw_probability=raw, **base)

    return Prediction(status=STATUS_OK, p_win=p_win, raw_probability=raw, **base)


def _validate_probability(p: Any) -> Optional[str]:
    try:
        v = float(p)
    except (TypeError, ValueError):
        return f"probability is not numeric: {p!r}"
    if np.isnan(v):
        return "probability is NaN"
    if np.isinf(v):
        return "probability is infinite"
    if not 0.0 <= v <= 1.0:
        return f"probability {v} is outside [0, 1]"
    return None


def predict_row(
    model: LoadedModel,
    row: Mapping[str, Any],
    *,
    entry_direction: Any,
    unavailable_columns: Sequence[str] = (),
    timestamp: Optional[pd.Timestamp] = None,
    warmed_up: Optional[bool] = None,
) -> Prediction:
    """Build the vector and score it in one call."""
    try:
        vector = build_feature_vector(
            model, row, entry_direction=entry_direction,
            unavailable_columns=unavailable_columns, timestamp=timestamp,
            warmed_up=warmed_up)
    except FeatureVectorError as exc:
        return Prediction(status=STATUS_NOT_COMPATIBLE, p_win=None, reason=str(exc),
                          model_id=model.card.model_id, symbol=model.symbol,
                          timeframe=model.timeframe, timestamp=timestamp)
    return predict(model, vector)
