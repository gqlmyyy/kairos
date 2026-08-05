# Trading Bot V3 - analysis/features/ml_dataset_builder.py

"""Convert execution_dataset rows into ML dataset rows ready for XGBoost.

Goal: No external preprocessing required after this layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger
from data.storage.database import get_execution_dataset  # type: ignore

logger = get_logger("ml_dataset_builder")

# If True: drop rows with missing critical features.
STRICT_MODE = True


# Minimal encoding maps (can be extended without changing the contract).
_SESSION_ENCODING: Dict[str, int] = {
    "asia": 1,
    "london": 2,
    "new_york": 3,
    "tokyo": 1,
    "": 0,
    None: 0,  # type: ignore[assignment]
}

_MARKET_REGIME_ENCODING: Dict[Any, int] = {
    # numeric (expected from feature_builder may already be signed)
    -1: 0,
    0: 1,
    1: 2,

    # strings
    "bear": 0,
    "bearish": 0,
    "neutral": 1,
    "flat": 1,
    "bull": 2,
    "bullish": 2,
    "trending": 2,
    "range": 1,
    "": 1,
    None: 1,  # type: ignore[assignment]
}


_CRITICAL_NUMERIC = {
    "rsi",
    "atr",
    "macd",
    "trend_strength",
    "momentum_score",
    "volatility_score",
    "spread",
    "ai_score",
    "sentiment_score",
    "news_impact_score",
}

_CRITICAL_CATEGORICAL = {
    "market_regime",
    "session",
}


_DIRECTIONAL_FEATURES = {
    "macd",
    "trend_strength",
    "momentum_score",
}


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _encode_session(session: Any) -> int:
    if session is None:
        return 0
    key = str(session).strip().lower()
    if not key:
        return 0
    return _SESSION_ENCODING.get(key, 0)


def _encode_market_regime(market_regime: Any) -> int:
    if market_regime is None:
        return _MARKET_REGIME_ENCODING[None]  # type: ignore[index]

    # numeric path
    if isinstance(market_regime, (int, float)):
        key = int(market_regime)
        return _MARKET_REGIME_ENCODING.get(key, 1)

    # string path
    key = str(market_regime).strip().lower()
    if not key:
        return _MARKET_REGIME_ENCODING[""]
    return _MARKET_REGIME_ENCODING.get(key, 1)


def explain_rejected_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Explain why a row was rejected by STRICT_MODE."""
    from data_quality import explain_rejected_row as _exp

    return _exp(row)


def build_ml_row(execution_row: Dict[str, Any]) -> Optional[Tuple[List[float], float]]:

    """Build a single ML row (X, y) from an execution_dataset row.

    Returns:
        (X, y) where X is float vector and y is target.
        If STRICT_MODE drops the row, returns None.
    """

    if not execution_row:
        return None

    # Map execution_dataset column names -> required features
    # execution_dataset uses expected_* and actual_* columns.
    # We build using actual_* first; fall back to expected_* only if actual missing.
    def feat(name: str) -> Any:
        actual_key = f"actual_{name}"
        if actual_key in execution_row and execution_row.get(actual_key) is not None:
            return execution_row.get(actual_key)
        expected_key = f"expected_{name}"
        return execution_row.get(expected_key)

    critical_values: Dict[str, Any] = {}
    for k in _CRITICAL_NUMERIC:
        critical_values[k] = feat(k)

    for k in _CRITICAL_CATEGORICAL:
        critical_values[k] = feat(k)

    # Strict mode: drop rows with missing critical features.
    if STRICT_MODE:
        missing = [k for k, v in critical_values.items() if v is None]
        if missing:
            return None

    # Fill numeric missing with 0.0; directional -> 0.0 neutral.
    numeric_features: Dict[str, float] = {}
    for k in _CRITICAL_NUMERIC:
        v = critical_values.get(k)
        if k in _DIRECTIONAL_FEATURES:
            numeric_features[k] = 0.0 if v is None else _safe_float(v, 0.0)
        else:
            numeric_features[k] = 0.0 if v is None else _safe_float(v, 0.0)

    market_regime_raw = critical_values.get("market_regime")
    session_raw = critical_values.get("session")

    market_regime_encoded = float(_encode_market_regime(market_regime_raw))
    session_encoded = float(_encode_session(session_raw))

    # Feature vector in required order
    X = [
        numeric_features["rsi"],
        numeric_features["atr"],
        numeric_features["macd"],
        numeric_features["trend_strength"],
        numeric_features["momentum_score"],
        numeric_features["volatility_score"],
        market_regime_encoded,
        session_encoded,
        numeric_features["spread"],
        numeric_features["ai_score"],
        numeric_features["sentiment_score"],
        numeric_features["news_impact_score"],
    ]

    # Target y
    y_raw = execution_row.get("actual_pnl")
    if y_raw is None:
        return None
    y = 1.0 if float(y_raw) > 0 else 0.0

    return X, y


def _get_all_execution_dataset_rows() -> List[Dict[str, Any]]:
    # Local import to avoid circular imports.
    from data.storage.database import get_conn

    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM execution_dataset")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def build_dataset_from_db(strict_mode: Optional[bool] = None) -> Tuple[List[List[float]], List[float]]:
    """Read execution_dataset from SQLite and build cleaned ML dataset.

    Args:
        strict_mode: override module STRICT_MODE.

    Returns:
        (X_train, y_train)
    """

    global STRICT_MODE
    if strict_mode is not None:
        STRICT_MODE = bool(strict_mode)

    rows = _get_all_execution_dataset_rows()

    accepted: List[List[float]] = []
    rejected = 0

    for r in rows:
        built = build_ml_row(r)
        if built is None:
            rejected += 1
            continue
        X, y = built
        accepted.append(X)
        # Keep y in same order

    # We must compute y_train from accepted Xs; easiest: rebuild again.
    # To avoid two passes over DB rows, do one pass storing y too.
    # Since we already fetched rows, do the proper approach in a single pass.
    # (We keep this as a fallback safety; but will redo in one pass below.)

    accepted_X: List[List[float]] = []
    y_train: List[float] = []
    rejected = 0

    for r in rows:
        built = build_ml_row(r)
        if built is None:
            rejected += 1
            continue
        X, y = built
        accepted_X.append(X)
        y_train.append(y)

    clean_ratio = (len(accepted_X) / max(len(rows), 1)) * 100.0

    logger.info(
        "ML dataset build: accepted=%d rejected=%d clean_ratio=%.2f%% total=%d strict_mode=%s",
        len(accepted_X),
        rejected,
        clean_ratio,
        len(rows),
        STRICT_MODE,
    )

    return accepted_X, y_train

