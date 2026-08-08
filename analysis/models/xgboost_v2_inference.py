# Trading Bot V3 - analysis/models/xgboost_v2_inference.py
# استخدام نموذج XGBoost الجديد (متدرب على بيانات تاريخية حقيقية من QuantDinger)
 
import xgboost as xgb
import os
from datetime import datetime, timezone
from utils.logger import get_logger
from analysis.models import entry_feature_contract as contract
 
logger = get_logger("xgboost_v2")
 
_model = None
MODEL_PATH = "models/entry/entry_model.json"

# The feature vector this path actually builds, in order. Named explicitly so
# the contract check can report *which* feature mismatched rather than only a
# count, and so a future model carrying feature_names can be compared by name.
LIVE_FEATURE_NAMES = (
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

_REGIME_ENC = {"RANGING": 0, "TRENDING": 1, "ranging": 0, "trending": 1}
_SESSION_ENC = {"asia": 0, "london": 1, "new_york": 2, "Asia": 0, "London": 1, "NY": 2}
_DIRECTION_ENC = {"SELL": 0, "BUY": 1}


def _as_dict(result) -> dict:
    """Flatten a GateResult into the dict shape main.py already consumes.

    `available` stays the key the decision gate branches on, so a blocked gate
    reaches `final_decision_valid = False` through the existing path.
    """
    return {
        "p_win": result.p_win,
        "available": result.available,
        "status": result.status,
        "reason": result.reason,
    }
 
 
def load_v2_model():
    """تحميل النموذج (مرة واحدة فقط - cached)"""
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        logger.info(f"[VERIFY] XGBOOST MODEL LOADED path={MODEL_PATH} available=False")
        logger.warning(f"XGBoost v2 model not found at {MODEL_PATH}")
        return None

    try:
        _model = xgb.Booster()
        _model.load_model(MODEL_PATH)
        logger.info(f"[VERIFY] XGBOOST MODEL LOADED path={MODEL_PATH} available=True")
        logger.info("XGBoost v2 model loaded successfully")
        return _model
    except Exception as e:
        logger.info(f"[VERIFY] XGBOOST MODEL LOADED path={MODEL_PATH} available=False")
        logger.error(f"Failed to load XGBoost v2 model: {e}")
        return None

 
 
def get_session_now() -> str:
    """تحديد الجلسة الحالية حسب الساعة UTC"""
    h = datetime.now(timezone.utc).hour
    if 0 <= h < 7:
        return "asia"
    if 7 <= h < 13:
        return "london"
    return "new_york"
 
 
def predict_with_v2(
    rsi: float, atr: float, macd: float,
    trend_strength: float, trend_score: float,
    momentum_score: float, volatility_score: float,
    market_regime: str, direction: str
) -> dict:
    """
    التنبؤ باستخدام النموذج الجديد (10 features بنفس ترتيب التدريب)
 
    Returns:
        {"p_win": float (0-1), "available": bool}
    """
    model = load_v2_model()
    if model is None:
        return _as_dict(contract.model_missing(f"model not loadable from {MODEL_PATH}"))

    session = get_session_now()
    features = [
        float(rsi or 0),
        float(atr or 0),
        float(macd or 0),
        float(trend_strength or 0),
        float(trend_score or 50),
        float(momentum_score or 50),
        float(volatility_score or 50),
        float(_REGIME_ENC.get(market_regime, 0)),
        float(_SESSION_ENC.get(session, 0)),
        float(_DIRECTION_ENC.get(direction, 0)),
    ]

    # ------------------------------------------------------------------
    # Hard contract check BEFORE predicting.
    #
    # XGBoost silently accepts a short vector, treating absent columns as
    # `missing` and following default branch directions. The deployed
    # artifact expects 65 features while this path supplies 10, so every
    # probability it produced was unrelated to the trade — BUY and SELL
    # even received identical values. Validate first, predict second.
    # ------------------------------------------------------------------
    model_contract = contract.contract_from_booster(model, model_version=f"v1@{MODEL_PATH}")
    reason = contract.validate_features(features, model_contract, supplied_names=LIVE_FEATURE_NAMES)
    if reason is not None:
        result = contract.invalid(reason, model_contract)
        logger.error(
            "[VERIFY] XGBOOST PREDICTION status=%s available=False reason=%s",
            result.status, reason,
        )
        return _as_dict(result)

    try:
        dmatrix = xgb.DMatrix([features], feature_names=list(LIVE_FEATURE_NAMES))
        p_win = float(model.predict(dmatrix)[0])
    except Exception as e:
        return _as_dict(contract.prediction_error(f"{type(e).__name__}: {e}", model_contract))

    bad = contract.validate_probability(p_win)
    if bad is not None:
        return _as_dict(contract.prediction_error(bad, model_contract))

    logger.info(f"[VERIFY] XGBOOST PREDICTION p_win={p_win:.3f} available=True")
    return _as_dict(contract.ok(p_win, model_contract))
 
 
def should_trade_v2(p_win: float, threshold: float = 0.60) -> bool:
    """قرار الدخول النهائي بناءً على احتمالية النجاح"""
    return p_win >= threshold
 
 
def get_size_multiplier(p_win: float) -> float:
    """ضبط حجم الصفقة بناءً على ثقة النموذج (وزن حقيقي لـ XGBoost في القرار)"""
    if p_win >= 0.85:
        return 1.5
    elif p_win >= 0.75:
        return 1.2
    elif p_win >= 0.60:
        return 1.0
    elif p_win >= 0.50:
        return 0.5
    else:
        return 0.0  # لا تدخل