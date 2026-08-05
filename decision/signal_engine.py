# Trading Bot V3 - decision/signal_engine.py
# Signal generation from AI analysis

from utils.logger import get_logger
from core.models import AINewsAnalysis

logger = get_logger("signal_engine")

def bias_to_direction(bias: str) -> str:
    if bias == "bullish":
        return "BUY"
    elif bias == "bearish":
        return "SELL"
    return "NEUTRAL"

def get_signal_strength(score: float, confidence: float) -> str:
    if score >= 80 and confidence >= 0.80:
        return "قوي جداً"
    elif score >= 70 and confidence >= 0.70:
        return "قوي"
    elif score >= 55 and confidence >= 0.60:
        return "متوسط"
    else:
        return "ضعيف"

def generate_signal(ai_analysis: AINewsAnalysis) -> dict:
    direction = bias_to_direction(ai_analysis.bias)
    score = ai_analysis.impact_score
    confidence = ai_analysis.confidence
    strength = get_signal_strength(score, confidence)

    logger.info(
        f"[VERIFY] SIGNAL_INPUT bias={ai_analysis.bias} score={score} confidence={confidence}"
    )

    # الإشارة صحيحة فقط إذا:
    # 1. الثقة فوق 0.6
    # 2. الاتجاه واضح
    # 3. الـ score فوق 55
    rule_conf = (confidence >= 0.60)
    rule_score = (score >= 55)
    rule_bias = (ai_analysis.bias in ["bullish", "bearish"])

    is_valid = (
        rule_conf
        and rule_bias
        and rule_score
    )

    # VERIFY: rules individually
    logger.info(
        f"[VERIFY] SIGNAL_RULE rule=confidence>=0.60 actual={confidence} result={rule_conf}"
    )
    logger.info(
        f"[VERIFY] SIGNAL_RULE rule=score>=55 actual={score} result={rule_score}"
    )
    logger.info(
        f"[VERIFY] SIGNAL_RULE rule=bias in [bullish,bearish] actual={ai_analysis.bias} result={rule_bias}"
    )

    signal = {
        "direction": direction,
        "score": score,
        "confidence": confidence,
        "bias": ai_analysis.bias,
        "reason": ai_analysis.reason,
        "strength": strength,
        "is_valid": is_valid
    }

    # VERIFY: final
    logger.info(
        f"[VERIFY] SIGNAL_FINAL direction={direction} score={score} confidence={confidence} valid={is_valid}"
    )

    if not is_valid:
        reasons = []
        if not rule_conf:
            reasons.append("confidence below threshold")
        if not rule_score:
            reasons.append("score below threshold")
        if not rule_bias:
            reasons.append("neutral bias")

        logger.info("[VERIFY] SIGNAL_FAILURE_REASONS")
        for r in reasons:
            logger.info(f"* {r}")

        # best-effort location info
        try:
            import inspect
            frame = inspect.currentframe()
            # currentframe can be None depending on runtime
            logger.info(
                f"[VERIFY] SIGNAL_FAILURE_LOCATION file={__file__} function=generate_signal"
            )
        except Exception:
            pass

    logger.debug(f"Signal: {direction} score={score} conf={confidence:.2f} strength={strength} valid={is_valid}")
    return signal
