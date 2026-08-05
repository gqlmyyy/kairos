from __future__ import annotations

from typing import Dict, Any, List

from .rule_types import RuleResult, SuggestedAction

from utils.logger import get_logger

logger = get_logger("stop_loss_breach_rule")

class StopLossBreachRule:
    """Suggest ClosePosition if SL is breached. No direct execution."""

    name = "StopLossBreach"

    def __call__(self, snapshot: Dict[str, Any], state: str) -> RuleResult:
        trade = snapshot["trade"]
        direction = trade["direction"]
        sl = trade.get("sl") or 0
        cur = trade.get("price_current") or 0
        breached = False

        logger.info(

    "[SL_CHECK] ticket=%s direction=%s current=%s sl=%s breached=%s",
    trade.get("order_id"),
    direction,
    cur,
    sl,
    breached,
)        
        if sl and sl > 0 and cur > 0:
            breached = (direction == "buy" and cur <= sl) or (direction == "sell" and cur >= sl)
        if not breached:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        return RuleResult(
            self.name,
            rule_score=1.0,
            reasons=["sl_breach"],
            suggested_actions=[SuggestedAction("ClosePosition", reason="sl_breach")],
        )

