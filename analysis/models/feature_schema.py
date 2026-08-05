from __future__ import annotations

from typing import Any, Dict, List


# IMPORTANT:
# This order must match exactly the XGBoost model feature_names inside
# models/exit/exit_model.json
FEATURE_ORDER: List[str] = [
    "mfe",
    "mae",
    "entry_atr",
    "entry_rsi",
    "entry_adx",
    "market_regime",
    "trade_duration",
    "spread",
    "volume",
    "session",
    "trend_h1",
    "trend_h4",
]


_MARKET_REGIME_MAP: Dict[str, float] = {
    "trending": 0.0,
    "weak trend": 1.0,
    "ranging": 2.0,
    "volatile": 3.0,
    "unknown": 4.0,
}


_SESSION_MAP: Dict[str, float] = {
    "asia": 0.0,
    "london": 1.0,
    "new_york": 2.0,
    "new york": 2.0,
    "unknown": 3.0,
    "none": 3.0,
    "": 3.0,
}


def _safe_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        if isinstance(x, str):
            s = x.strip().lower()
            if s == "" or s in {"none", "null", "nan"}:
                return default
            return float(s)
        return float(x)
    except Exception:
        return default


def _encode_market_regime(v: Any) -> float:
    if v is None:
        return _MARKET_REGIME_MAP["unknown"]
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    return _MARKET_REGIME_MAP.get(s, _MARKET_REGIME_MAP["unknown"])


def _encode_session(v: Any) -> float:
    if v is None:
        return _SESSION_MAP["unknown"]
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    # Normalize separators
    s = s.replace(" ", "_")
    if s == "new_york":
        return _SESSION_MAP["new_york"]
    return _SESSION_MAP.get(s, _SESSION_MAP["unknown"])


def build_feature_vector(d: Dict[str, Any]) -> List[float]:
    """Build feature vector strictly in FEATURE_ORDER.

    Any missing feature becomes 0.0 (except market_regime/session which get
    explicit encoding).

    Returns:
        List[float] length == len(FEATURE_ORDER)
    """

    vec: List[float] = []

    for name in FEATURE_ORDER:
        raw = d.get(name, None)

        if name == "market_regime":
            vec.append(float(_encode_market_regime(raw)))
            continue

        if name == "session":
            vec.append(float(_encode_session(raw)))
            continue

        vec.append(_safe_float(raw, default=0.0))

    return vec

