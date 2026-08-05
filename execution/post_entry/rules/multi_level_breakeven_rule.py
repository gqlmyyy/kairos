from __future__ import annotations

from typing import Any, Dict

from config import (
    MULTI_BREAKEVEN_ENABLED,
    LEVEL1_PROFIT_POINTS,
    LEVEL1_SL_OFFSET,
    LEVEL2_PROFIT_POINTS,
    LEVEL3_PROFIT_POINTS,
    LEVEL3_SL_OFFSET,
)

from .rule_types import RuleResult, SuggestedAction


class MultiLevelBreakevenRule:
    """Pure multi-level breakeven.

    Reuses logic of execution/risk_management/multi_level_breakeven.py but without in-module state.
    Uses snapshot trade fields to infer current profit_points and current_sl.
    Suggests MoveSL only.

    Note: since pure rules cannot keep state, we rely on DB flags breakeven_done when present.
    """

    name = "MultiLevelBreakeven"

    def __call__(self, snapshot: Dict[str, Any], state: str) -> RuleResult:
        if not MULTI_BREAKEVEN_ENABLED:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        trade = snapshot.get("trade", {})
        direction = str(trade.get("direction") or "").lower()
        if direction not in ("buy", "sell"):
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        expected_row = snapshot.get("expected_row") or {}

        # infer current SL
        try:
            entry_price = float(trade.get("entry_price") or 0)
        except Exception:
            entry_price = 0.0

        try:
            cur_sl = float(trade.get("sl") or 0)
        except Exception:
            cur_sl = 0.0

        if entry_price <= 0:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        # profit_points is not present in current trade snapshot; best-effort from dataset
        profit_points = None
        # common approach: use expected_row actual_mae/mfe if present; but we follow requirement to reuse logic.
        if "profit_points" in expected_row:
            profit_points = expected_row.get("profit_points")

        if profit_points is None:
            # fallback: approximate using price difference / pip heuristic
            try:
                current_price = float(trade.get("price_current") or 0)
            except Exception:
                current_price = 0.0
            if current_price > 0:
                # pip heuristic
                symbol = str(trade.get("symbol") or "")
                pip = 0.01 if symbol.upper().endswith("JPY") else 0.0001
                if pip > 0:
                    if direction == "sell":
                        profit_points = (entry_price - current_price) / pip
                    else:
                        profit_points = (current_price - entry_price) / pip

        if profit_points is None:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        try:
            pp = float(profit_points)
        except Exception:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        # stage inference
        db_row = snapshot.get("db_row") or {}
        stage_done = int(db_row.get("breakeven_done") or 0) or 0

        is_sell = direction == "sell"

        def is_better(new_sl: float) -> bool:
            if cur_sl <= 0:
                return True
            return (new_sl < cur_sl) if is_sell else (new_sl > cur_sl)

        # Stage 1
        if stage_done < 1 and pp >= float(LEVEL1_PROFIT_POINTS):
            new_sl = entry_price + float(LEVEL1_SL_OFFSET) if is_sell else entry_price - float(LEVEL1_SL_OFFSET)
            if is_better(new_sl):
                return RuleResult(
                    self.name,
                    rule_score=0.7,
                    reasons=["be_level_1"],
                    suggested_actions=[SuggestedAction("MoveSL", value=float(new_sl), reason="multi_be_1")],
                )

        # Stage 2
        if stage_done < 2 and pp >= float(LEVEL2_PROFIT_POINTS):
            new_sl = entry_price
            if is_better(new_sl):
                return RuleResult(
                    self.name,
                    rule_score=0.9,
                    reasons=["be_level_2"],
                    suggested_actions=[SuggestedAction("MoveSL", value=float(new_sl), reason="multi_be_2")],
                )

        # Stage 3
        if stage_done < 3 and pp >= float(LEVEL3_PROFIT_POINTS):
            new_sl = float(entry_price + float(LEVEL3_SL_OFFSET))
            if is_better(new_sl):
                return RuleResult(
                    self.name,
                    rule_score=1.0,
                    reasons=["be_level_3"],
                    suggested_actions=[SuggestedAction("MoveSL", value=float(new_sl), reason="multi_be_3")],
                )

        return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

