from __future__ import annotations

from typing import Any, Dict

from config import PARTIAL_TP_ENABLED, PARTIAL_TP_RATIO

from .rule_types import RuleResult, SuggestedAction


class PartialTPRule:
    """Partial TP + trailing-style SL suggestion.

    Pure rule: never executes MT5/DB/Telegram.

    Inputs expected in snapshot["trade"]:
      - direction (buy/sell)
      - entry_price
      - volume
      - tp (take profit)
      - sl
      - price_current
    """

    name = "PartialTP"

    def __call__(self, snapshot: Dict[str, Any], state: str) -> RuleResult:
        trade = snapshot.get("trade", {})
        direction = str(trade.get("direction") or "").lower()
        if direction not in ("buy", "sell"):
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        if not PARTIAL_TP_ENABLED:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        # State gating: only when opened and not already partial.
        if state not in ("OPENED", "OPENED_ACTIVE", "PARTIAL_CLOSED"):
            # If your state machine uses different labels, keep OPENED-like only.
            pass

        entry = _f(trade.get("entry_price"))
        vol = _f(trade.get("volume"))
        tp = _f(trade.get("tp"))
        cur = _f(trade.get("price_current"))
        cur_sl = _f(trade.get("sl"))

        if entry <= 0 or vol <= 0 or tp is None or tp <= entry or cur <= 0:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        # Compute first partial TP level.
        ratio = float(PARTIAL_TP_RATIO)
        if direction == "sell":
            first_tp = entry - (entry - tp) * ratio
            hit = cur <= first_tp
        else:
            first_tp = entry + (tp - entry) * ratio
            hit = cur >= first_tp

        if not hit:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        half_volume = vol * 0.5
        if half_volume <= 0:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        # trailing-style: suggestion to MoveSL to breakeven (entry)
        suggested_actions = [
            SuggestedAction("PartialClose", value=half_volume, reason="partial_tp_hit"),
        ]

        if cur_sl is None or cur_sl <= 0:
            suggested_actions.append(SuggestedAction("MoveSL", value=entry, reason="be_after_partial"))
        else:
            better = (entry > cur_sl) if direction == "buy" else (entry < cur_sl)
            if better:
                suggested_actions.append(SuggestedAction("MoveSL", value=entry, reason="be_after_partial"))

        return RuleResult(
            self.name,
            rule_score=1.0,
            reasons=["partial_tp_hit", f"first_tp={first_tp}", f"half_volume={half_volume}"],
            suggested_actions=suggested_actions,
        )


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

