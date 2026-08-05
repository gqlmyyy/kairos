from __future__ import annotations

from typing import Any, Dict

from config import MARKET_REGIME_ENABLED

from .rule_types import RuleResult, SuggestedAction


class MarketRegimeRule:
    """Market Regime safety modifier.

    Pure rule: suggests MoveSL/Continue based on regime.

    This rule is best-effort because the new Trade Monitor provides limited indicators.

    Expected snapshot trade fields:
      - direction
    Expected snapshot["market_regime"] or snapshot["trade"].market_regime
    """

    name = "MarketRegime"

    def __call__(self, snapshot: Dict[str, Any], state: str) -> RuleResult:
        if not MARKET_REGIME_ENABLED:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        regime = snapshot.get("market_regime") or snapshot.get("trade", {}).get("market_regime")
        regime = str(regime or "UNKNOWN").lower()

        # Known bad regimes map
        bad = {"ranging", "volatile"}
        if regime in bad:
            # Prefer protecting open profit by moving SL closer to entry.
            trade = snapshot.get("trade", {})
            entry = _f(trade.get("entry_price"))
            cur_sl = _f(trade.get("sl"))
            direction = str(trade.get("direction") or "").lower()
            if entry > 0 and direction in ("buy", "sell"):
                is_sell = direction == "sell"
                # move SL to entry if improves
                better = (cur_sl <= 0) or ((not is_sell and entry > cur_sl) or (is_sell and entry < cur_sl))
                if better:
                    return RuleResult(
                        self.name,
                        rule_score=0.6,
                        reasons=[f"bad_regime:{regime}"],
                        suggested_actions=[SuggestedAction("MoveSL", value=float(entry), reason="regime_protect")],
                    )

            return RuleResult(self.name, 0.3, [f"bad_regime:{regime}"], [SuggestedAction("Continue")])

        return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

