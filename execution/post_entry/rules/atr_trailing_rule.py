from __future__ import annotations

from typing import Any, Dict, Optional

from config import ATR_TRAILING_ENABLED, ATR_TRAILING_MULTIPLIER, TRAILING_ACTIVATE_ATR_MULTIPLE

from .rule_types import RuleResult, SuggestedAction


class ATRTrailingRule:
    """Pure rule for ATR trailing.

    Reuses formula from execution/risk_management/atr_trailing.py but does not call MT5.
    Suggests MoveSL when trailing would improve SL.
    """

    name = "ATRTrailing"

    def __call__(self, snapshot: Dict[str, Any], state: str) -> RuleResult:
        trade = snapshot.get("trade", {})
        direction = str(trade.get("direction") or "").lower()
        if direction not in ("buy", "sell"):
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        if not ATR_TRAILING_ENABLED:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        atr = trade.get("atr")
        if atr is None:
            # best-effort: allow expected_row atr if present
            expected_row = snapshot.get("expected_row") or {}
            atr = expected_row.get("expected_atr") or expected_row.get("actual_atr")

        try:
            atr_f = float(atr) if atr is not None else 0.0
        except Exception:
            atr_f = 0.0

        if atr_f <= 0:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        current_price = trade.get("price_current")
        if current_price is None:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        try:
            price = float(current_price)
        except Exception:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        if price <= 0:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        current_sl = trade.get("sl")
        try:
            cur_sl = float(current_sl) if current_sl is not None else 0.0
        except Exception:
            cur_sl = 0.0

        is_sell = direction == "sell"

        # Break-even threshold: trailing only activates after the trade
        # reaches TRAILING_ACTIVATE_ATR_MULTIPLE * ATR in profit.
        entry_price = trade.get("entry_price")
        try:
            entry_f = float(entry_price) if entry_price is not None else 0.0
        except Exception:
            entry_f = 0.0

        if entry_f > 0:
            profit_in_atr = (price - entry_f) / atr_f if not is_sell else (entry_f - price) / atr_f
            activate_threshold = float(TRAILING_ACTIVATE_ATR_MULTIPLE)
            if profit_in_atr < activate_threshold:
                return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        trail_distance = float(ATR_TRAILING_MULTIPLIER) * atr_f
        if trail_distance <= 0:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        new_sl = price + trail_distance if is_sell else price - trail_distance

        if cur_sl <= 0:
            return RuleResult(
                self.name,
                rule_score=0.4,
                reasons=["initial_atr_trail"],
                suggested_actions=[SuggestedAction("MoveSL", value=float(new_sl), reason="atr_trail")],
            )

        better = (new_sl < cur_sl) if is_sell else (new_sl > cur_sl)
        if not better:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        return RuleResult(
            self.name,
            rule_score=0.8,
            reasons=["atr_trail_improve"],
            suggested_actions=[SuggestedAction("MoveSL", value=float(new_sl), reason="atr_trail")],
        )

