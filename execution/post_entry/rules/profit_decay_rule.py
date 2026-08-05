from __future__ import annotations

from typing import Any, Dict, Optional

from config import (
    PROFIT_DECAY_ENABLED,
    PROFIT_DECAY_TRIGGER,
    PROFIT_DECAY_PWIN_THRESHOLD,
)

from .rule_types import RuleResult, SuggestedAction


# In-memory peak profit tracking per order_id (lifetime of this process)
_peak_profit_by_order: Dict[str, float] = {}


class ProfitDecayRule:
    """Profit Decay detector (pure rule).

    Maintains in-memory peak_profit per order_id. No MT5/DB/Telegram side effects.

    Snapshot trade fields expected:
      - order_id
      - profit (current unrealized)
    Snapshot expected_row/db_row optional fields:
      - p_win
    """

    name = "ProfitDecay"

    def __call__(self, snapshot: Dict[str, Any], state: str) -> RuleResult:
        if not PROFIT_DECAY_ENABLED:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        trade = snapshot.get("trade", {})
        order_id = str(trade.get("order_id") or "")
        if not order_id:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        cur_profit = _f(trade.get("profit"))
        # peak update
        prev_peak = _peak_profit_by_order.get(order_id)
        if prev_peak is None or cur_profit > prev_peak:
            _peak_profit_by_order[order_id] = float(cur_profit)
            prev_peak = float(cur_profit)

        peak = float(prev_peak or 0.0)
        if peak <= 0:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        expected_row = snapshot.get("expected_row") or snapshot.get("db_row") or {}
        p_win: Optional[float] = expected_row.get("p_win")
        if p_win is None:
            p_win = expected_row.get("expected_p_win")
        if p_win is None:
            p_win = expected_row.get("p_win_v2")

        reasons = []
        # primary p_win gate
        if p_win is not None:
            try:
                p = float(p_win)
                if p < float(PROFIT_DECAY_PWIN_THRESHOLD):
                    reasons.append(f"pwin_decay:p_win={p:.3f}")
                    return RuleResult(
                        self.name,
                        rule_score=1.0,
                        reasons=reasons,
                        suggested_actions=[SuggestedAction("ClosePosition", reason="profit_decay_pwin")],
                    )
            except Exception:
                pass

        # fallback: profit-vs-peak
        trigger = float(PROFIT_DECAY_TRIGGER)
        decay_ratio = (cur_profit / peak) if peak != 0 else 1.0
        if cur_profit < peak * trigger:
            reasons.append(f"profit_decay:cur={cur_profit:.2f},peak={peak:.2f},trig={trigger:.2f}")
            return RuleResult(
                self.name,
                rule_score=1.0,
                reasons=reasons,
                suggested_actions=[SuggestedAction("ClosePosition", reason="profit_decay_ratio")],
            )

        return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

