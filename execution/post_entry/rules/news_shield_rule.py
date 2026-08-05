from __future__ import annotations

from typing import Any, Dict

from config import NEWS_SHIELD_ENABLED, NEWS_SHIELD_CLOSE_TRADES

from .rule_types import RuleResult, SuggestedAction


class NewsShieldRule:
    """News Shield.

    Pure rule: decides whether to continue or close positions.

    Since this refactor requires no DB/Telegram/Logger in rules,
    this rule uses best-effort detection by calling the existing
    high-impact detector (which may fetch news).

    Expected snapshot:
      - trade.symbol

    If NEWS_SHIELD_CLOSE_TRADES is True, suggests ClosePosition.
    Otherwise suggests Continue with lower score.
    """

    name = "NewsShield"

    def __call__(self, snapshot: Dict[str, Any], state: str) -> RuleResult:
        if not NEWS_SHIELD_ENABLED:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        trade = snapshot.get("trade", {})
        symbol = str(trade.get("symbol") or "")
        if not symbol:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        try:
            from execution.risk_management.news_shield import is_high_impact_news_now  # best-effort
            blocked = bool(is_high_impact_news_now(symbol))
        except Exception:
            # fail-open inside rule; fusion will rely on other rules
            blocked = False

        if not blocked:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        if NEWS_SHIELD_CLOSE_TRADES:
            return RuleResult(
                self.name,
                rule_score=1.0,
                reasons=["news_shield_high_impact"],
                suggested_actions=[SuggestedAction("ClosePosition", reason="news_shield")],
            )

        return RuleResult(
            self.name,
            rule_score=0.3,
            reasons=["news_shield_high_impact"],
            suggested_actions=[SuggestedAction("Continue")],
        )

