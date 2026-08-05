from __future__ import annotations

from typing import Any, Dict, Iterable, List

from config import CORRELATION_PROTECTION_ENABLED, CORRELATED_PAIRS

from .rule_types import RuleResult, SuggestedAction


class CorrelationProtectionRule:
    """Correlation Protection.

    Pure rule: suggests Continue or ClosePosition depending on whether current
    trade is too correlated with existing open positions.

    Expected snapshot contains:
      - trade: current trade
      - portfolio: optional list of open positions under key "open_positions"
    """

    name = "CorrelationProtection"

    def __call__(self, snapshot: Dict[str, Any], state: str) -> RuleResult:
        if not CORRELATION_PROTECTION_ENABLED:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        trade = snapshot.get("trade", {})
        symbol = str(trade.get("symbol") or "")
        direction = str(trade.get("direction") or "").lower()
        if not symbol or direction not in ("buy", "sell"):
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        open_positions: Iterable[Dict[str, Any]] = snapshot.get("open_positions") or []

        # Determine if any correlated open position exists with same direction.
        for pos in open_positions:
            try:
                psym = str(pos.get("symbol") or "")
                pdir = str(pos.get("direction") or pos.get("type") or "").lower()
                if not psym:
                    continue
                if pdir in ("sell", "short", "-1", "1") and direction == "buy":
                    continue
                # normalize direction
                pdir_norm = "sell" if pdir in ("sell", "short", "-1") else "buy"
                if pdir_norm != direction:
                    continue

                for a, b in CORRELATED_PAIRS:
                    if (symbol.upper() == str(a).upper() and psym.upper() == str(b).upper()) or (
                        symbol.upper() == str(b).upper() and psym.upper() == str(a).upper()
                    ):
                        return RuleResult(
                            self.name,
                            rule_score=0.8,
                            reasons=["correlated_open"],
                            suggested_actions=[SuggestedAction("Continue")],
                        )
            except Exception:
                continue

        return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

