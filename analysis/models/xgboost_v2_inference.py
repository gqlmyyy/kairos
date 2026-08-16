# Trading Bot V3 - analysis/models/xgboost_v2_inference.py
# استخدام نموذج XGBoost الجديد (متدرب على بيانات تاريخية حقيقية من QuantDinger)
 
import xgboost as xgb
import os
from datetime import datetime, timezone
from utils.logger import get_logger
from analysis.models import entry_feature_contract as contract
from analysis.models import entry_model_metadata as metadata
from analysis.models.entry_feature_spec import (
    FEATURE_NAMES as LIVE_FEATURE_NAMES,
    build_feature_vector,
    session_now,
)

logger = get_logger("xgboost_v2")

_model = None
_metadata = None
MODEL_PATH = "models/entry/entry_model.json"

# The feature order, encodings and missing-value policy live in
# analysis/models/entry_feature_spec.py, which the training pipeline imports
# too. Keeping a second copy here is what let training and serving drift apart
# in the first place, so this module now only re-exports the name.


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
    global _model, _metadata
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        logger.info(f"[VERIFY] XGBOOST MODEL LOADED path={MODEL_PATH} available=False")
        logger.warning(f"XGBoost v2 model not found at {MODEL_PATH}")
        return None

    # Load into a local and publish to the cache only on success.
    #
    # This used to assign `_model = xgb.Booster()` and *then* call load_model()
    # on it. When load_model raised — a truncated file, a partially-written
    # model, a read racing a replacement — the global was already holding an
    # empty Booster. The call correctly returned None, but every later call hit
    # the `if _model is not None` fast path and handed back that empty Booster,
    # whose num_features() raises. The gate then reported
    # "ML_GATE_INVALID: model feature count could not be determined" forever,
    # pointing at the feature contract instead of the real cause, and the
    # process never recovered even once the file on disk was fixed.
    try:
        booster = xgb.Booster()
        booster.load_model(MODEL_PATH)
    except Exception as e:
        logger.info(f"[VERIFY] XGBOOST MODEL LOADED path={MODEL_PATH} available=False")
        logger.error(f"Failed to load XGBoost v2 model: {e}")
        return None

    # Provenance gate. A booster that parses is not yet a model we may serve:
    # nothing inside the artifact says which feature schema, which label
    # definition, or which dataset it came from. Four separate trainers in this
    # repository have written to this one path with three different schemas, so
    # "it loaded" is not evidence of anything. Refuse anything that cannot
    # prove its identity, and refuse it here rather than at predict time, so a
    # mismatched artifact is never held in the cache.
    try:
        meta = metadata.load(MODEL_PATH)
    except metadata.MetadataError as e:
        logger.info(f"[VERIFY] XGBOOST MODEL LOADED path={MODEL_PATH} available=False")
        logger.error(f"[ML_CONTRACT] refusing to serve {MODEL_PATH}: {e}")
        return None

    for reason in (
        metadata.validate_against_booster(meta, booster),
        metadata.validate_for_serving(meta, live_feature_names=LIVE_FEATURE_NAMES),
    ):
        if reason is not None:
            logger.info(f"[VERIFY] XGBOOST MODEL LOADED path={MODEL_PATH} available=False")
            logger.error(f"[ML_CONTRACT] refusing to serve {MODEL_PATH}: {reason}")
            return None

    _model = booster
    _metadata = meta
    logger.info(f"[VERIFY] XGBOOST MODEL LOADED path={MODEL_PATH} available=True")
    logger.info("XGBoost v2 model loaded: %s", meta.describe())
    return _model


def loaded_metadata():
    """Metadata of the currently cached model, or None when nothing is served."""
    return _metadata

 
 
def get_session_now() -> str:
    """تحديد الجلسة الحالية حسب الساعة UTC.

    Kept as a thin alias: the session boundaries themselves now live in
    entry_feature_spec so the training pipeline derives a historical bar's
    session with the identical rule.
    """
    return session_now()
 
 
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

    # Built through the shared spec so training and serving cannot diverge.
    features = build_feature_vector(
        rsi=rsi,
        atr=atr,
        macd=macd,
        trend_strength=trend_strength,
        trend_score=trend_score,
        momentum_score=momentum_score,
        volatility_score=volatility_score,
        market_regime=market_regime,
        session=session_now(),
        direction=direction,
    )

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