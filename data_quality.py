from __future__ import annotations

from typing import Any, Dict, List, Tuple

from utils.logger import get_logger

logger = get_logger("data_quality")


def _is_number(v: Any) -> bool:
    if v is None:
        return False
    try:
        fv = float(v)
    except Exception:
        return False
    if fv != fv:  # NaN
        return False
    return True


def validate_execution_row(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate one execution_dataset row.

    Rules (strict):
    - expected_* required fields must exist (no None)
    - indicators required fields must exist (rsi/atr/macd/trend_strength/session)
    - closed rows must have actual_pnl non-None
    - realistic ranges:
        RSI: [0,100]
        ATR: > 0

    Returns:
      (accepted, reasons)
    """
    reasons: List[str] = []

    if not isinstance(row, dict):
        return False, ["row_is_not_dict"]

    status = row.get("status")

    critical_expected_keys = [
        "expected_entry",
        "expected_rsi",
        "expected_macd",
        "expected_session",
        "expected_atr",
        "expected_trend_strength",
        # Ensure ML can build row under STRICT_MODE
        "expected_momentum_score",
        "expected_volatility_score",
        "expected_spread",
        "expected_ai_score",
        "expected_sentiment_score",
        "expected_news_impact_score",
        "expected_market_regime",
    ]

    # expected_* must be present regardless of open/closed (training uses expected snapshot)
    for k in critical_expected_keys:
        if row.get(k) is None:
            reasons.append(f"{k}_is_none")

    # If you store expected_* as numeric, enforce not NaN
    for k in ["expected_rsi", "expected_atr", "expected_trend_strength", "expected_macd", "expected_entry"]:
        if row.get(k) is not None and not _is_number(row.get(k)):
            reasons.append(f"{k}_not_number")

    if row.get("expected_rsi") is not None:
        try:
            rsi = float(row.get("expected_rsi"))
            if rsi < 0.0 or rsi > 100.0:
                reasons.append("expected_rsi_out_of_range")
        except Exception:
            reasons.append("expected_rsi_parse_error")

    if row.get("expected_atr") is not None:
        try:
            atr = float(row.get("expected_atr"))
            if atr <= 0.0:
                reasons.append("expected_atr_le_0")
        except Exception:
            reasons.append("expected_atr_parse_error")

    if row.get("expected_entry") is not None:
        try:
            ep = float(row.get("expected_entry"))
            if ep <= 0.0:
                reasons.append("expected_entry_le_0")
        except Exception:
            reasons.append("expected_entry_parse_error")

    # Closed label rules
    if status == "closed":
        if row.get("actual_pnl") is None:
            reasons.append("actual_pnl_is_none")
        # execution_quality_score may be None in some legacy reconciliations.
        # Training uses strict expected_* gate; we don't hard-reject on quality_score.


    accepted = len(reasons) == 0
    return accepted, reasons


def explain_rejected_row(row: Dict[str, Any]) -> Dict[str, Any]:
    accepted, reasons = validate_execution_row(row)
    return {
        "accepted": accepted,
        "missing_or_invalid_fields": reasons,
        "reason": "STRICT_MODE reject" if not accepted else "ok",
    }

