from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from utils.logger import get_logger

logger = get_logger("decision_fusion")


@dataclass
class Decision:
    decision: str  # Continue|MoveSL|MoveTP|PartialClose|ClosePosition
    decision_score: float
    confidence: float
    reasons: List[str]
    rule_score: float = 0.0
    model_score: float = 0.0


class DecisionFusionEngine:
    """Only fuses signals and returns Decision. No execution."""

    def __init__(self) -> None:
        pass

    def fuse(
        self,
        rule_results: List[Any],
        red_flags: List[str],
        xgb: Dict[str, Optional[float]],
        snapshot: Dict[str, Any],
        risk_status: Optional[Dict[str, Any]] = None,
    ) -> Decision:
        # rule_score aggregate
        rule_score = 0.0
        reasons: List[str] = []
        suggested = []
        for rr in rule_results:
            try:
                rule_score += float(getattr(rr, "rule_score", 0) or 0)
                reasons.extend(list(getattr(rr, "reasons", []) or []))
                suggested.extend(list(getattr(rr, "suggested_actions", []) or []))
            except Exception:
                continue

        exit_prob = xgb.get("exit_probability") if xgb else None
        model_score = float(exit_prob) if exit_prob is not None else 0.0

        # Fusion: simple (keep existing scoring behavior)
        decision_score = rule_score + model_score + (0.1 * len(red_flags))
        confidence = max(0.0, min(1.0, model_score))

        # Exit-model consult path (only when RedFlagDetector requests it)
        # should_consult_exit_model must be provided by RedFlagDetector inside snapshot["..."].
        should_consult_exit_model = bool(
            snapshot.get("_red_flag_report", {})
            and snapshot.get("_red_flag_report", {}).get("should_consult_exit_model")
        )
        if should_consult_exit_model and exit_prob is not None:

            # If model adapter returned features_incomplete, do not consult model at all.
            if bool(xgb.get("features_incomplete")):
                logger = __import__("utils.logger", fromlist=["get_logger"]).get_logger("exit_model_consult")
                order_id = None
                try:
                    trade = snapshot.get("trade") or {}
                    order_id = trade.get("order_id")
                except Exception:
                    order_id = None
                logger.warning(f"[EXIT_MODEL] features_incomplete=True -> skip exit-model consult order_id={order_id}")
            else:
                prob = float(exit_prob)
                threshold_applied: str
                final_decision: str


            if prob > 0.90:
                final_decision = "ClosePosition"
                threshold_applied = "CLOSE"
            elif prob >= 0.70:
                final_decision = "MoveSL"
                threshold_applied = "TIGHTEN_SL"
            else:
                final_decision = "Continue"
                threshold_applied = "IGNORED"

            logger = __import__("utils.logger", fromlist=["get_logger"]).get_logger("exit_model_consult")
            logger.info(
                "========== EXIT MODEL CONSULT ==========\n"
                f"FlagCount        : {snapshot.get('_red_flag_report', {}).get('flag_count', 'N/A')}\n"
                f"Probability      : {prob:.4f}\n"
                f"ThresholdApplied : {threshold_applied}\n"
                f"FinalDecision    : {final_decision}\n"
                "========================================="
            )

            if final_decision == "ClosePosition":
                return Decision(
                    "ClosePosition",
                    decision_score,
                    confidence,
                    reasons,
                    rule_score=rule_score,
                    model_score=model_score,
                )

            if final_decision == "MoveSL":
                # Keep existing RuleEngine precedence for actual MoveSL value.
                # We only change the decision type when rules did not request close.
                return Decision(
                    "MoveSL",
                    decision_score,
                    confidence,
                    reasons,
                    rule_score=rule_score,
                    model_score=model_score,
                )

            # IGNORED => continue to existing rule-based decision path.

        # Decide by precedence: ClosePosition if any rule suggests
        for s in suggested:
            try:
                if getattr(s, "action_type", None) == "ClosePosition":
                    return Decision(
                        "ClosePosition",
                        decision_score,
                        confidence,
                        reasons,
                        rule_score=rule_score,
                        model_score=model_score,
                    )
            except Exception:
                continue

        return Decision(
            "Continue",
            decision_score,
            confidence,
            reasons,
            rule_score=rule_score,
            model_score=model_score,
        )


