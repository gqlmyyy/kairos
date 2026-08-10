"""Single source of truth for the entry model's feature vector.

Both the live inference path (`analysis/models/xgboost_v2_inference.py`) and the
training pipeline (`scripts/train_entry_model.py`) build their vector by calling
:func:`build_feature_vector` in this module. Nothing else may define the order,
the encodings, or the missing-value policy — that duplication is exactly what
produced the 65-vs-10 mismatch this module exists to prevent.

If you change anything here, both paths change together, and
`tests/test_entry_feature_parity.py` proves they still agree.

The ten features, in wire order
-------------------------------

===  ===================  =========  =========================================
#    name                 type       how it is produced at inference time
===  ===================  =========  =========================================
0    rsi                  float      H1 snapshot `rsi` (mt5_client.get_indicators)
1    atr                  float      H4 `atr` via get_atr(symbol) — default "4H"
2    macd                 float      H1 snapshot `macd`
3    trend_strength       float      MTF strength string -> TREND_STRENGTH_ENCODING
4    trend_score          float      H4 ma_trend/RSI bucket: {40,65,70,75,85}
5    momentum_score       float      H1 RSI bucket: {40,65,85}
6    volatility_score     float      H1 volatility bucket: {20,35,55,80}
7    market_regime        float      encoded, see REGIME_ENCODING
8    session              float      encoded from UTC hour, see SESSION_ENCODING
9    direction            float      SELL=0.0, BUY=1.0
===  ===================  =========  =========================================

Three features were frozen constants — now fixed
------------------------------------------------
All three were verified empirically before and after. They are recorded here
because the fixes changed live signal generation, and because a reader needs to
know why the calibration constants in ``config.py`` exist.

**1. trend_strength** — was always ``0.0``. ``main.py`` passed
``mtf.strength if isinstance(mtf.strength, (int, float)) else 0.0``, but
``MultiTimeframeData.strength`` is a *string* ("weak"/"moderate"/"strong"), so
the guard never passed. Now encoded by :func:`encode_trend_strength` from
``config.TREND_STRENGTH_VALUES``. An unrecognised value maps to
``TREND_STRENGTH_DEFAULT`` (0.0), kept distinct from a genuine "weak" (25.0) so
"not measured" and "measured, weak" never collide.

**2. volatility_score** — was always ``55.0``. ``get_volatility_score_from_snapshot``
reads a ``volatility`` key that no code path emitted, so every lookup missed and
fell through to the neutral default. ``mt5_client.get_indicators`` now derives it
from ATR relative to price — the volatility measure the pipeline already
computes — bucketed by ``config.VOLATILITY_PCT_*``.

**3. market_regime** — was always ``TRENDING``. ``ma_trend`` returned "sideways"
only when ``price == ma20`` *exactly*, a float equality that does not occur, so
the H4 trend direction was never "neutral". ``get_indicators`` now treats price
within ``config.MA_TREND_FLAT_ATR_MULT`` ATRs of MA20 as flat, which makes
RANGING reachable; with volatility_score live, HIGH_VOLATILITY and
LOW_VOLATILITY become reachable too.

Training reproduces all three through the same code (`live_parity_features`
mirrors `get_indicators`, and both call this module's encoders), so there is one
calibration, not two.

"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

import config as _cfg

# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

FEATURE_NAMES: tuple = (
    "rsi",
    "atr",
    "macd",
    "trend_strength",
    "trend_score",
    "momentum_score",
    "volatility_score",
    "market_regime",
    "session",
    "direction",
)

FEATURE_COUNT = len(FEATURE_NAMES)

# Encodings. `.get(key, default)` semantics are part of the contract: an
# unrecognised value must land on the documented default rather than raise,
# because live market data occasionally produces regimes this map predates.
#
# NOTE: `get_market_regime_from_snapshot` can return HIGH_VOLATILITY and
# LOW_VOLATILITY, neither of which is a key here — both encode to 0.0, the same
# value as RANGING. That collision is inherited from the original inference code
# and is preserved deliberately so training and live agree; it is recorded in
# KNOWN_ISSUES.md rather than silently "fixed" on one side only.
REGIME_ENCODING: Dict[str, int] = {
    "RANGING": 0,
    "TRENDING": 1,
    "ranging": 0,
    "trending": 1,
}
REGIME_DEFAULT = 0

SESSION_ENCODING: Dict[str, int] = {
    "asia": 0,
    "london": 1,
    "new_york": 2,
    "Asia": 0,
    "London": 1,
    "NY": 2,
}
SESSION_DEFAULT = 0

DIRECTION_ENCODING: Dict[str, int] = {"SELL": 0, "BUY": 1}
DIRECTION_DEFAULT = 0

# `MultiTimeframeData.strength` is a string. main.py used to guard it with
# isinstance(..., (int, float)), which never passed, so the model always got
# 0.0. Values come from config so live and training share one calibration.
TREND_STRENGTH_ENCODING: Dict[str, float] = {
    k: float(v) for k, v in _cfg.TREND_STRENGTH_VALUES.items()
}
TREND_STRENGTH_DEFAULT = float(_cfg.TREND_STRENGTH_DEFAULT)

# Defaults applied when a value is None/absent. These mirror the `or` fallbacks
# in the original predict_with_v2 exactly.
MISSING_DEFAULTS: Dict[str, float] = {
    "rsi": 0.0,
    "atr": 0.0,
    "macd": 0.0,
    "trend_strength": 0.0,
    "trend_score": 50.0,
    "momentum_score": 50.0,
    "volatility_score": 50.0,
}

# Previously frozen in production, now live (see module docstring). Kept as an
# explicit empty mapping rather than deleted: the training pipeline checks it to
# decide which constant features are *expected*, and a regression that re-freezes
# one of these must surface as an unexpected constant, not be waved through.
LIVE_CONSTANT_FEATURES: Dict[str, float] = {}


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

def encode_regime(market_regime: Any) -> float:
    return float(REGIME_ENCODING.get(market_regime, REGIME_DEFAULT))


def encode_session(session: Any) -> float:
    return float(SESSION_ENCODING.get(session, SESSION_DEFAULT))


def encode_direction(direction: Any) -> float:
    return float(DIRECTION_ENCODING.get(direction, DIRECTION_DEFAULT))


def encode_trend_strength(strength: Any) -> float:
    """Map MTF alignment strength to a number.

    Accepts the string the analyser actually produces ("weak"/"moderate"/
    "strong"), and passes a real number straight through so a future numeric
    source needs no change here. Anything unrecognised becomes
    TREND_STRENGTH_DEFAULT, which is deliberately distinct from the value for
    a genuine "weak" — "we could not measure it" and "we measured it and it is
    weak" must not collide.
    """
    if isinstance(strength, bool):
        return TREND_STRENGTH_DEFAULT
    if isinstance(strength, (int, float)):
        return float(strength)
    if strength is None:
        return TREND_STRENGTH_DEFAULT
    return TREND_STRENGTH_ENCODING.get(str(strength).strip().lower(),
                                       TREND_STRENGTH_DEFAULT)


def session_from_hour(hour_utc: int) -> str:
    """UTC hour -> session name.

    The single definition of the session boundaries. Live calls it with the
    current hour; training calls it with the bar's own timestamp, so a
    historical row is labelled with the session it actually occurred in.
    """
    h = int(hour_utc) % 24
    if 0 <= h < 7:
        return "asia"
    if 7 <= h < 13:
        return "london"
    return "new_york"


def session_now() -> str:
    """Session for the current wall-clock UTC hour (live path)."""
    return session_from_hour(datetime.now(timezone.utc).hour)


def session_from_timestamp(unix_seconds: float) -> str:
    """Session for a historical bar (training path)."""
    return session_from_hour(
        datetime.fromtimestamp(float(unix_seconds), tz=timezone.utc).hour
    )


# ---------------------------------------------------------------------------
# The builder — the one function both paths call
# ---------------------------------------------------------------------------

def _num(value: Any, default: float) -> float:
    """Mirror the original `float(x or default)` semantics precisely.

    Note this treats 0 and 0.0 as "missing" and substitutes the default, because
    `0 or 5` is `5` in Python. That is what the deployed inference code did, so
    it is what training must do too — being *consistent* matters more here than
    being ideal, and changing it would silently shift the input distribution.
    """
    if value is None:
        return float(default)
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return float(default)


def build_feature_vector(
    *,
    rsi: Any,
    atr: Any,
    macd: Any,
    trend_strength: Any,
    trend_score: Any,
    momentum_score: Any,
    volatility_score: Any,
    market_regime: Any,
    session: str,
    direction: Any,
) -> List[float]:
    """Assemble the 10-feature vector in wire order.

    Every argument is keyword-only: a positional call that silently transposed
    two features would produce a plausible-looking vector and a model that
    quietly learned the wrong thing.

    ``session`` is passed in rather than read from the clock, so training can
    supply the historical bar's session and live can supply the current one
    without the two paths diverging on anything else.
    """
    return [
        _num(rsi, MISSING_DEFAULTS["rsi"]),
        _num(atr, MISSING_DEFAULTS["atr"]),
        _num(macd, MISSING_DEFAULTS["macd"]),
        encode_trend_strength(trend_strength),
        _num(trend_score, MISSING_DEFAULTS["trend_score"]),
        _num(momentum_score, MISSING_DEFAULTS["momentum_score"]),
        _num(volatility_score, MISSING_DEFAULTS["volatility_score"]),
        encode_regime(market_regime),
        encode_session(session),
        encode_direction(direction),
    ]


def as_named_dict(vector: Sequence[float]) -> Dict[str, float]:
    """Zip a vector back to names — for logging and parity diffing."""
    if len(vector) != FEATURE_COUNT:
        raise ValueError(
            f"expected {FEATURE_COUNT} features, got {len(vector)}"
        )
    return dict(zip(FEATURE_NAMES, (float(v) for v in vector)))


def describe() -> str:
    return f"{FEATURE_COUNT} features in order: {', '.join(FEATURE_NAMES)}"
