from __future__ import annotations

from typing import Any, Dict, Optional

from config import TRADE_HEALTH_ENABLED, TRADE_HEALTH_MIN_SCORE

from .rule_types import RuleResult, SuggestedAction


class TradeHealthRule:
    """Trade Health (p_win -> health score) defensive rule.

    Pure rule. Suggests ClosePosition when health below threshold.

    Snapshot trade fields expected:
      - order_id
      - symbol

    Snapshot expected_row/db_row expected fields:
      - p_win or expected_ai_confidence proxy
    """

    name = "TradeHealth"

    def __call__(self, snapshot: Dict[str, Any], state: str) -> RuleResult:
        if not TRADE_HEALTH_ENABLED:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        trade = snapshot.get("trade", {})
        symbol = str(trade.get("symbol") or "")
        order_id = str(trade.get("order_id") or "")
        if not order_id or not symbol:
            # allow if values missing, but can't decide
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        expected_row = snapshot.get("expected_row") or snapshot.get("db_row") or {}
        p_win: Optional[float] = expected_row.get("p_win")
        if p_win is None:
            p_win = expected_row.get("expected_p_win")
        if p_win is None:
            p_win = expected_row.get("p_win_v2")

        if p_win is None:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        try:
            p = float(p_win)
        except Exception:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        health_score = p * 100.0
        if health_score < float(TRADE_HEALTH_MIN_SCORE):
            return RuleResult(
                self.name,
                rule_score=1.0,
                reasons=[f"health={health_score:.1f}", f"min={float(TRADE_HEALTH_MIN_SCORE):.1f}"],
                suggested_actions=[SuggestedAction("ClosePosition", reason="trade_health")],
            )

        return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

