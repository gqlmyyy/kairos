from __future__ import annotations

from typing import Any, Dict

from config import (
    # config.py uses PROTECT_PROFIT_ENABLED name
    PROTECT_PROFIT_ENABLED as PROFIT_PROTECT_ENABLED,
    PROFIT_PROTECT_TRIGGER_POINTS,
    PROFIT_PROTECT_LOCK_POINTS,
    PIP_VALUES,
)


from .rule_types import RuleResult, SuggestedAction


class ProtectOpenProfitRule:
    """Protect Open Profit.

    Pure rule: suggests MoveSL only.

    Expected snapshot trade fields:
      - direction
      - entry_price
      - sl
      - profit_points (if absent -> 0)
    """

    name = "ProtectOpenProfit"

    def __call__(self, snapshot: Dict[str, Any], state: str) -> RuleResult:
        if not PROFIT_PROTECT_ENABLED:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        trade = snapshot.get("trade", {})
        direction = str(trade.get("direction") or "").lower()
        if direction not in ("buy", "sell"):
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        order_id = str(trade.get("order_id") or "")

        entry = _f(trade.get("entry_price"))
        cur_sl = _f(trade.get("sl"))
        profit_points = _f(trade.get("profit_points"))

        if entry <= 0 or profit_points is None:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        if float(profit_points) < float(PROFIT_PROTECT_TRIGGER_POINTS):
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        lock_pts = float(PROFIT_PROTECT_LOCK_POINTS)
        if lock_pts <= 0:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        # CRITICAL: convert points to price units using pip value.
        # PROFIT_PROTECT_LOCK_POINTS is in POINTS (pips), not price units.
        # For XAUUSD: pip = 0.1, so 10 points = 1.0 price unit.
        # For EURUSD: pip = 0.0001, so 10 points = 0.001 price unit.
        symbol = str(trade.get("symbol") or "")
        pip = float(PIP_VALUES.get(symbol, 0.0001))
        lock_price_distance = lock_pts * pip

        is_sell = direction == "sell"
        new_sl = float(entry - lock_price_distance) if is_sell else float(entry + lock_price_distance)

        if cur_sl > 0:
            better = (new_sl < cur_sl) if is_sell else (new_sl > cur_sl)
            if not better:
                return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        reasons = ["protect_open_profit", f"profit_points={profit_points}"]
        return RuleResult(
            self.name,
            rule_score=0.9,
            reasons=reasons,
            suggested_actions=[SuggestedAction("MoveSL", value=float(new_sl), reason="protect_profit")],
        )


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

