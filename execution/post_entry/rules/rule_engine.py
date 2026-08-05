from __future__ import annotations

from typing import Dict, Any, List

from .rule_types import RuleResult

from .simple_rules import StopLossBreachRule
from .time_stop_rule import TimeStopRule
from .atr_trailing_rule import ATRTrailingRule
from .multi_level_breakeven_rule import MultiLevelBreakevenRule
from .partial_tp_rule import PartialTPRule
from .dynamic_breakeven_rule import DynamicBreakevenRule
from .protect_open_profit_rule import ProtectOpenProfitRule
from .dynamic_tp_rule import DynamicTPRule
from .profit_decay_rule import ProfitDecayRule
from .trade_health_rule import TradeHealthRule
from .news_shield_rule import NewsShieldRule
from .market_regime_rule import MarketRegimeRule
from .correlation_protection_rule import CorrelationProtectionRule


class RuleEngine:


    """Pure rule engine.

    Evaluates all migrated rules and returns RuleResult objects.

    No MT5/Telegram/DB operations here.
    """

    def __init__(self) -> None:
        self._rules = [
            # fixed lifecycle rules
            TimeStopRule(),
            ATRTrailingRule(),
            MultiLevelBreakevenRule(),
            StopLossBreachRule(),

            # migrated legacy rules (pure)
            PartialTPRule(),
            DynamicBreakevenRule(),
            ProtectOpenProfitRule(),
            DynamicTPRule(),
            ProfitDecayRule(),
            TradeHealthRule(),
            NewsShieldRule(),
            MarketRegimeRule(),
            CorrelationProtectionRule(),
        ]

    def evaluate(self, snapshot: Dict[str, Any], state: str) -> List[RuleResult]:
        results: List[RuleResult] = []
        for r in self._rules:
            results.append(r(snapshot, state))
        return results


