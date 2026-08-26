"""Four states, kept apart on purpose: VALID, MISSING, INVALID, UNAVAILABLE.

Why this module exists
----------------------
The legacy entry path collapses every kind of absence into a number. Its
``_num`` helper is ``float(value or default)``, which means a genuine ``0.0``
becomes the default — a real, measured zero and "we never measured this" are
the same number by the time the model sees them. Its ``MISSING_DEFAULTS``
then substitute 50.0 for an unmeasured score, which is indistinguishable
from a real reading of 50.0.

A model cannot tell a substituted value from an observed one. So the
substitution does not degrade the prediction, it invalidates it — quietly,
with a plausible-looking probability attached. This module makes that
impossible by refusing to reduce the four cases to one:

``VALID``
    A real, finite, observed value. **Zero is valid.** This is the state a
    legitimate ``0.0`` lands in, and keeping it out of MISSING is the single
    most important thing here.

``MISSING``
    The feature is computable in principle, but not at this row — warm-up is
    incomplete, or the formula's denominator was zero. NaN, not a number.

``INVALID``
    A value was produced and is not usable: NaN where warm-up is complete,
    or an infinity. Something is wrong with the input, not with the schedule.

``UNAVAILABLE``
    The data source cannot produce this feature AT ALL — the column it needs
    does not exist in the feed. Structural, not per-row. KAIROS's stored
    historical candles carry no ``spread``, so ``spread_relative`` is
    UNAVAILABLE against them, and every research model requires it.

Any state other than VALID on a required feature makes the PREDICTION
invalid. It does not make the feature 0.0.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

VALID = "VALID"
MISSING = "MISSING"
INVALID = "INVALID"
UNAVAILABLE = "UNAVAILABLE"

#: The only state a required feature may be in for a prediction to be served.
SERVABLE_STATES = frozenset({VALID})


@dataclass(frozen=True)
class FeatureState:
    """One feature's value and why it is (or is not) usable."""

    name: str
    state: str
    value: Optional[float]
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.state in SERVABLE_STATES


def classify(name: str, value, *, available: bool = True,
             warmed_up: bool = True) -> FeatureState:
    """Put one produced value into exactly one of the four states.

    ``available=False`` means the SOURCE cannot produce this column; it
    outranks everything else, because there is no value to judge.
    """
    if not available:
        return FeatureState(name, UNAVAILABLE, None,
                            "the candle source does not carry the column this feature reads")
    if value is None:
        return FeatureState(name, MISSING, None, "no value produced")
    # Strictly a real number, never a string that happens to parse. The engine
    # always emits float64; a str arriving here means something upstream is
    # wrong, and `float("50")` would hide that behind a plausible value.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return FeatureState(name, INVALID, None,
                            f"not a real number: {type(value).__name__} {value!r}")
    v = float(value)
    if math.isnan(v):
        return (FeatureState(name, MISSING, None, "not yet defined (warm-up incomplete)")
                if not warmed_up else
                FeatureState(name, INVALID, None,
                             "NaN after warm-up — a zero denominator or a corrupt bar"))
    if math.isinf(v):
        return FeatureState(name, INVALID, None, "infinite")
    # Reached deliberately by v == 0.0. A zero is a measurement.
    return FeatureState(name, VALID, v)


@dataclass(frozen=True)
class VectorState:
    """The availability verdict for one complete feature vector."""

    states: Tuple[FeatureState, ...]

    @property
    def usable(self) -> bool:
        return all(s.usable for s in self.states)

    def values(self) -> List[Optional[float]]:
        return [s.value for s in self.states]

    def offenders(self) -> Tuple[FeatureState, ...]:
        return tuple(s for s in self.states if not s.usable)

    def by_state(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for s in self.states:
            out.setdefault(s.state, []).append(s.name)
        return out

    def summary(self, limit: int = 4) -> str:
        bad = self.offenders()
        if not bad:
            return f"all {len(self.states)} features VALID"
        head = "; ".join(f"{s.name}={s.state} ({s.reason})" for s in bad[:limit])
        more = f" and {len(bad) - limit} more" if len(bad) > limit else ""
        return f"{len(bad)}/{len(self.states)} features unusable: {head}{more}"


def build_vector_state(names: Sequence[str], values: Sequence,
                       unavailable: Sequence[str] = (),
                       warmed_up: bool = True) -> VectorState:
    """Classify a whole vector, preserving the contract's feature order."""
    if len(names) != len(values):
        raise ValueError(f"{len(names)} names vs {len(values)} values")
    unavail = set(unavailable)
    return VectorState(tuple(
        classify(n, v, available=n not in unavail, warmed_up=warmed_up)
        for n, v in zip(names, values)
    ))
