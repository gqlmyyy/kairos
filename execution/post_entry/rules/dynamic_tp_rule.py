from __future__ import annotations

from typing import Any, Dict, Optional

from config import DYNAMIC_TP_ENABLED, DYNAMIC_TP_AGGRESSION

from .rule_types import RuleResult, SuggestedAction


class DynamicTPRule:
    """Dynamic TP update (model-driven via p_win).

    Pure rule: suggests MoveTP only.

    Expected snapshot trade fields:
      - entry_price
      - tp
      - direction
      - order_id
    Snapshot expected_row (optional):
      - p_win / expected_ai_confidence proxy
    """

    name = "DynamicTP"

    def __call__(self, snapshot: Dict[str, Any], state: str) -> RuleResult:
        if not DYNAMIC_TP_ENABLED:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        trade = snapshot.get("trade", {})
        direction = str(trade.get("direction") or "").lower()
        if direction not in ("buy", "sell"):
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        if trade.get("tp") is None:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        entry = _f(trade.get("entry_price"))
        cur_tp = _f(trade.get("tp"))
        if entry <= 0 or cur_tp <= 0:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        expected_row = snapshot.get("expected_row") or snapshot.get("db_row") or {}
        p_win: Optional[float] = expected_row.get("p_win")
        if p_win is None:
            p_win = expected_row.get("expected_p_win")
        if p_win is None:
            p_win = expected_row.get("p_win_v2")

        if p_win is None:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        p = _f(p_win)
        if p <= 0:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        # Expand target based on p_win.
        factor = 1.0 + (p - 0.6) * float(DYNAMIC_TP_AGGRESSION)

        if direction == "buy":
            new_tp = entry + (cur_tp - entry) * factor
            if new_tp <= cur_tp:
                return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])
        else:
            new_tp = entry - (entry - cur_tp) * factor
            if new_tp >= cur_tp:
                return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        return RuleResult(
            self.name,
            rule_score=0.6,
            reasons=["dynamic_tp", f"p_win={p:.3f}"] ,
            suggested_actions=[SuggestedAction("MoveTP", value=float(new_tp), reason="dynamic_tp")],
        )


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

