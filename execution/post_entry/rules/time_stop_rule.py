from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from config import TIME_STOP_ENABLED, TIME_STOP_MAX_MINUTES

from .rule_types import RuleResult, SuggestedAction


class TimeStopRule:
    """Pure rule wrapper around execution/risk_management/time_stop.py

    Suggests ClosePosition when trade age exceeds limit.
    No MT5/Telegram/DB.
    """

    name = "TimeStop"

    def __call__(self, snapshot: Dict[str, Any], state: str) -> RuleResult:
        trade = snapshot.get("trade", {})
        order_id = str(trade.get("order_id") or "")
        symbol = str(trade.get("symbol") or "")

        if not TIME_STOP_ENABLED:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        if not order_id or not symbol:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        open_time = trade.get("time_open")
        if open_time is None:
            # cannot decide without age
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        try:
            open_ts = float(open_time)
        except Exception:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        max_min = float(TIME_STOP_MAX_MINUTES)
        if max_min <= 0:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        elapsed_min = (time.time() - open_ts) / 60.0
        if elapsed_min < max_min:
            return RuleResult(self.name, 0.0, [], [SuggestedAction("Continue")])

        return RuleResult(
            self.name,
            rule_score=1.0,
            reasons=[f"time_stop:{elapsed_min:.2f}min>={max_min:.2f}min"],
            suggested_actions=[SuggestedAction("ClosePosition", reason="time_stop")],
        )

