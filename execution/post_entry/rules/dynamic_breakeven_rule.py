from __future__ import annotations

from typing import Any, Dict

from config import (
    DYNAMIC_BREAKEVEN_ENABLED,
    DYNAMIC_BE_FAST_EMA,
    DYNAMIC_BE_SLOW_EMA,
    DYNAMIC_BE_FALLBACK_POINTS,
)

from .rule_types import RuleResult, SuggestedAction


class DynamicBreakevenRule:
    """Dynamic breakeven (EMA crossover / fallback points).

    Pure rule: suggests MoveSL to entry.

    Snapshot expected trade fields:
      - direction
      - entry_price
      - sl
      - price_current

    For EMA values, rule reads from snapshot.get("market") or snapshot["expected_row"]
    best-effort.
    """

    name = "DynamicBreakeven"

    def __call__(self, snapshot: Dict[str, Any], state: str) -> RuleResult:
        if not DYNAMIC_BREAKEVEN_ENABLED:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        trade = snapshot.get("trade", {})
        direction = str(trade.get("direction") or "").lower()
        if direction not in ("buy", "sell"):
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        entry = _f(trade.get("entry_price"))
        cur_sl = _f(trade.get("sl"))
        cur_price = _f(trade.get("price_current"))
        if entry <= 0 or cur_price <= 0:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        # prevent worsen SL
        is_sell = direction == "sell"
        best = (cur_sl <= 0) or ((not is_sell and entry > cur_sl) or (is_sell and entry < cur_sl))
        if not best:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        # EMA-based trigger (best-effort using snapshot expected_row)
        expected_row = snapshot.get("expected_row") or snapshot.get("db_row") or {}
        ema_fast = expected_row.get("ema_fast") or expected_row.get("expected_ema_fast")
        ema_slow = expected_row.get("ema_slow") or expected_row.get("expected_ema_slow")

        # fallback: profit points in snapshot if present
        profit_points = expected_row.get("profit_points") or expected_row.get("expected_profit_points")

        should_move = False
        if ema_fast is not None and ema_slow is not None:
            try:
                if is_sell:
                    should_move = float(ema_fast) < float(ema_slow)
                else:
                    should_move = float(ema_fast) > float(ema_slow)
            except Exception:
                should_move = False

        if not should_move and profit_points is not None:
            try:
                should_move = float(profit_points) >= float(DYNAMIC_BE_FALLBACK_POINTS)
            except Exception:
                should_move = False

        if not should_move:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        return RuleResult(
            self.name,
            rule_score=0.7,
            reasons=["dynamic_be"],
            suggested_actions=[SuggestedAction("MoveSL", value=float(entry), reason="dynamic_be_trigger")],
        )


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

