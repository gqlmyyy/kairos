"""The one entry gate into the baseline models. `models/baseline` is the only
source; there is no fallback and no legacy path.

    main.py
      -> baseline_gate.predict_entry(symbol, timeframe, entry_direction)
      -> load_model(symbol, timeframe)      models/baseline/<S>/<TF>/model.json ONLY
      -> vendored feature pipeline          same code that trained the artifacts
      -> xgb.Booster.predict                p_win in [0, 1]
      -> dict(p_win, available, status, reason)

Result-dict shape matches what main.py already consumes from the previous
gate, so the call site changes by three lines and nothing downstream moves.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

from analysis.baseline import VENDOR_DIR  # noqa: F401  (bootstraps sys.path)

logger = get_logger("baseline_gate")

import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402

from src.config.loader import load_config  # noqa: E402  (vendored, see package doc)
from src.features.live import LiveFeaturePipeline  # noqa: E402  (vendored)

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# The single source of trained models. Every load in this module resolves
# under this root and nowhere else.
# ---------------------------------------------------------------------------
BASELINE_ROOT = REPO_ROOT / "models" / "baseline"

#: Frozen training config copied from the xgbooost rebuild (config/*.yaml).
CONFIG_DIR = Path(__file__).resolve().parent / "vendor" / "config"

TIMEFRAME_MINUTES = {"M15": 15, "H1": 60, "H4": 240}
SUPPORTED_TIMEFRAMES = ("M15", "H1", "H4")

#: Per-timeframe schema fingerprint of the shipped artifacts. A file whose
#: feature count disagrees with its folder is not the model it claims to be.
EXPECTED_SCHEMA_COUNT = {"H4": 66, "H1": 136, "M15": 203}

#: History bars fetched per timeframe for feature computation. Covers every
#: warm-up in the schema (EMA50, ADX, rolling spread median 200, day levels)
#: with margin, on completed candles only.
FETCH_BARS = 1200

STATUS_OK = "OK"
STATUS_MODEL_MISSING = "ML_MODEL_MISSING"
STATUS_NOT_COMPATIBLE = "ML_GATE_INVALID"
STATUS_FEATURE_INVALID = "ML_GATE_INVALID"
STATUS_PREDICTION_ERROR = "ML_PREDICTION_ERROR"

_ENTRY_DIRECTION_ENCODING = {"BUY": 1.0, "SELL": -1.0}

_lock = threading.RLock()
_models: Dict[tuple, xgb.Booster] = {}
_model_names: Dict[tuple, List[str]] = {}
_cfg = None


class BaselineModelError(Exception):
    """A baseline artifact cannot prove what it claims to be."""


def model_path(symbol: str, timeframe: str) -> Path:
    """The one path a (symbol, timeframe) model may live at."""
    return BASELINE_ROOT / symbol.upper() / timeframe.upper() / "model.json"


def _config():
    global _cfg
    with _lock:
        if _cfg is None:
            _cfg = load_config(CONFIG_DIR)
        return _cfg


def load_model(symbol: str, timeframe: str) -> xgb.Booster:
    """Load the artifact for (symbol, timeframe) from models/baseline ONLY.

    Raises BaselineModelError when the file is absent, outside the expected
    schema, or unreadable -- never falls back to another model or path.
    """
    symbol = symbol.upper()
    timeframe = timeframe.upper()
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise BaselineModelError(
            f"unsupported timeframe {timeframe!r}; known: {list(SUPPORTED_TIMEFRAMES)}")

    key = (symbol, timeframe)
    with _lock:
        cached = _models.get(key)
    if cached is not None:
        return cached

    path = model_path(symbol, timeframe)
    if not path.is_file():
        raise BaselineModelError(
            f"no baseline artifact at {path} "
            f"(source of truth is {BASELINE_ROOT}; there is no fallback)")

    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        names = list(meta["learner"]["feature_names"])
        objective = str(meta["learner"]["objective"]["name"])
    except Exception as exc:  # noqa: BLE001
        raise BaselineModelError(f"unreadable artifact {path}: {exc}") from None

    expected = EXPECTED_SCHEMA_COUNT[timeframe]
    if len(names) != expected:
        raise BaselineModelError(
            f"{path}: {len(names)} features, expected {expected} for {timeframe} "
            f"-- this is not the artifact its folder claims")

    if objective != "binary:logistic":
        raise BaselineModelError(f"{path}: objective {objective!r} is not binary:logistic")

    try:
        booster = xgb.Booster()
        booster.load_model(str(path))
    except Exception as exc:  # noqa: BLE001
        raise BaselineModelError(f"{path}: xgboost load failed: {exc}") from None

    with _lock:
        _models[key] = booster
        _model_names[key] = names
    logger.info("[VERIFY] BASELINE MODEL LOADED symbol=%s tf=%s features=%d path=%s",
                symbol, timeframe, len(names), path)
    return booster


# ---------------------------------------------------------------------------
# Candles -> canonical frames -> feature row
# ---------------------------------------------------------------------------

def fetch_frames(symbol: str, timeframes: Optional[List[str]] = None) -> Dict[str, "Any"]:
    """Completed candles per timeframe as the canonical DataFrame the vendored
    pipeline expects (timestamp tz-aware UTC, symbol, timeframe, OHLC, spread,
    tick_volume). Spread comes from MT5's rate struct via get_candles."""
    import pandas as pd

    from data.market.mt5_client import get_candles

    if timeframes is None:
        timeframes = list(SUPPORTED_TIMEFRAMES)
    frames: Dict[str, Any] = {}
    for tf in timeframes:
        rows = get_candles(symbol, tf, FETCH_BARS)
        if not rows:
            raise BaselineModelError(
                f"no completed candles from MT5 for {symbol} {tf} -- cannot build features")
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["symbol"] = symbol.upper()
        df["timeframe"] = tf
        if "spread" not in df.columns:
            raise BaselineModelError(
                "candle feed carries no spread column -- spread_points/spread_relative/"
                "spread_ma are model features and cannot be fabricated")
        frames[tf] = df[["timestamp", "symbol", "timeframe",
                         "open", "high", "low", "close", "spread", "tick_volume"]]
    return frames


def context_timeframes(timeframe: str) -> List[str]:
    """Higher timeframes the entry timeframe's model was trained with."""
    return list(_config().context_timeframes_for(timeframe.upper()))


def compute_feature_row(symbol: str, timeframe: str) -> Dict[str, float]:
    """The feature row of the last CLOSED entry candle, in the model's own
    schema. Computed by the vendored training pipeline -- not a reimplementation."""
    timeframe = timeframe.upper()
    frames = fetch_frames(symbol, [timeframe] + context_timeframes(timeframe))
    pipeline = LiveFeaturePipeline(_config(), symbol.upper(), timeframe)
    series, _specs = pipeline.compute(frames)

    row = {str(k): float(v) for k, v in series.items()}
    # entry_direction is trade-side metadata: the dataset builder wrote the
    # labelled side (BUY=+1, SELL=-1) into feature 0. The engine does not emit
    # it; the gate injects it from the direction actually being scored.
    row.setdefault("entry_direction", float("nan"))
    return row


def _encode_direction(direction: Any) -> float:
    value = _ENTRY_DIRECTION_ENCODING.get(str(direction).strip().upper())
    if value is None:
        raise BaselineModelError(
            f"unrecognised entry direction {direction!r}; known: BUY/SELL")
    return value


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _result(p_win, available, status, reason, symbol, timeframe,
            model: Optional[str] = None) -> Dict[str, Any]:
    return {"p_win": p_win, "available": available, "status": status,
            "reason": reason, "symbol": symbol, "timeframe": timeframe,
            "model": model}


def predict_entry(symbol: str, timeframe: Optional[str] = None,
                  entry_direction: Any = "BUY") -> Dict[str, Any]:
    """Score one entry on the (symbol, timeframe) baseline model.

    Returns the dict shape main.py already consumes:
        {"p_win": float|None, "available": bool, "status": str, "reason": str}
    Never raises for expected operating conditions -- a problem becomes an
    unavailable result that names its cause.
    """
    timeframe = (timeframe or "H1").upper()
    symbol = symbol.upper()
    try:
        booster = load_model(symbol, timeframe)
        model_names = _model_names[(symbol, timeframe)]
    except BaselineModelError as exc:
        logger.error("[BASELINE_GATE] %s — entry BLOCKED. symbol=%s tf=%s reason=%s",
                     STATUS_MODEL_MISSING, symbol, timeframe, exc)
        return _result(None, False, STATUS_MODEL_MISSING, str(exc), symbol, timeframe)

    try:
        side = _encode_direction(entry_direction)
        row = compute_feature_row(symbol, timeframe)
        row["entry_direction"] = side
    except BaselineModelError as exc:
        logger.error("[BASELINE_GATE] %s — entry BLOCKED. symbol=%s tf=%s reason=%s",
                     STATUS_FEATURE_INVALID, symbol, timeframe, exc)
        return _result(None, False, STATUS_FEATURE_INVALID, str(exc), symbol, timeframe)
    except Exception as exc:  # noqa: BLE001 - report, never substitute
        logger.error("[BASELINE_GATE] %s — entry BLOCKED. symbol=%s tf=%s reason=%s",
                     STATUS_FEATURE_INVALID, symbol, timeframe, exc)
        return _result(None, False, STATUS_FEATURE_INVALID,
                       f"{type(exc).__name__}: {exc}", symbol, timeframe)

    missing = [n for n in model_names if n not in row]
    if missing:
        reason = (f"feature pipeline produced {len(model_names) - len(missing)}/"
                  f"{len(model_names)} columns; missing: {missing[:5]}")
        logger.error("[BASELINE_GATE] %s — entry BLOCKED. symbol=%s tf=%s reason=%s",
                     STATUS_NOT_COMPATIBLE, symbol, timeframe, reason)
        return _result(None, False, STATUS_NOT_COMPATIBLE, reason, symbol, timeframe)

    values = [row[n] for n in model_names]
    if not all(math.isfinite(v) for v in values):
        bad = [n for n, v in zip(model_names, values) if not math.isfinite(v)]
        reason = f"non-finite features: {bad[:5]}"
        logger.error("[BASELINE_GATE] %s — entry BLOCKED. symbol=%s tf=%s reason=%s",
                     STATUS_FEATURE_INVALID, symbol, timeframe, reason)
        return _result(None, False, STATUS_FEATURE_INVALID, reason, symbol, timeframe)

    try:
        dmatrix = xgb.DMatrix([values], feature_names=list(model_names))
        p_win = float(booster.predict(dmatrix)[0])
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        logger.error("[BASELINE_GATE] %s — entry BLOCKED. symbol=%s tf=%s reason=%s",
                     STATUS_PREDICTION_ERROR, symbol, timeframe, reason)
        return _result(None, False, STATUS_PREDICTION_ERROR, reason, symbol, timeframe)

    if not math.isfinite(p_win) or not 0.0 <= p_win <= 1.0:
        reason = f"probability {p_win} outside [0, 1]"
        logger.error("[BASELINE_GATE] %s — entry BLOCKED. symbol=%s tf=%s reason=%s",
                     STATUS_PREDICTION_ERROR, symbol, timeframe, reason)
        return _result(None, False, STATUS_PREDICTION_ERROR, reason, symbol, timeframe)

    logger.info("[VERIFY] BASELINE PREDICTION symbol=%s tf=%s model=%s direction=%s "
                "p_win=%.4f available=True", symbol, timeframe,
                model_path(symbol, timeframe), str(entry_direction).upper(), p_win)
    return _result(p_win, True, STATUS_OK, "ok", symbol, timeframe,
                   model=str(model_path(symbol, timeframe)))


def probe_availability() -> bool:
    """True when every shipped (symbol, timeframe) artifact loads and matches
    its schema fingerprint. Used by the boot banner only -- never gates a trade
    by itself."""
    from config import SYMBOLS
    try:
        for symbol in SYMBOLS:
            for timeframe in SUPPORTED_TIMEFRAMES:
                load_model(symbol, timeframe)
        return True
    except Exception as exc:  # noqa: BLE001 - diagnostics must never stop boot
        logger.error("[BASELINE_GATE] availability probe failed: %s", exc)
        return False
