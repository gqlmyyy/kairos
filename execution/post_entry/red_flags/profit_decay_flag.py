from __future__ import annotations

from typing import Any, Dict, Optional

from .flag_result import FlagResult


class ProfitDecayFlag:
    """Profit Decay red-flag.

    Trigger logic (wired to existing adapter pattern):
      - current profit: snapshot['trade']['profit']
      - peak/maximum profit proxy: position_state.mfe

    Score:
      - derived from profit_decay_pct ratio (profit_decay_pct / 100)

    Notes:
      - If position_state.mfe is unavailable, the flag is not triggered.
    """

    name = "ProfitDecay"

    @classmethod
    def check(
        cls,
        snapshot: Dict[str, Any],
        position_state: Optional[Any] = None,
    ) -> FlagResult:
        trade = snapshot.get("trade") or {}
        current_profit = trade.get("profit")

        if current_profit is None:
            return FlagResult(
                name=cls.name,
                triggered=False,
                score=0.0,
                severity=0,
                reason="current profit missing from snapshot['trade']['profit']",
                metadata={"source": "snapshot.trade.profit"},
            )

        try:
            current_profit_f = float(current_profit)
        except Exception:
            return FlagResult(
                name=cls.name,
                triggered=False,
                score=0.0,
                severity=0,
                reason="current profit not numeric",
                metadata={"source": "snapshot.trade.profit", "profit": current_profit},
            )

        if position_state is None or not hasattr(position_state, "mfe"):
            return FlagResult(
                name=cls.name,
                triggered=False,
                score=0.0,
                severity=0,
                reason="mfe not available (position_state.mfe missing)",
                metadata={"source": "position_state.mfe"},
            )

        try:
            mfe = float(getattr(position_state, "mfe"))
        except Exception:
            return FlagResult(
                name=cls.name,
                triggered=False,
                score=0.0,
                severity=0,
                reason="mfe not numeric",
                metadata={"source": "position_state.mfe"},
            )

        if mfe <= 0:
            return FlagResult(
                name=cls.name,
                triggered=False,
                score=0.0,
                severity=0,
                reason="mfe unavailable or non-positive",
                metadata={"source": "position_state.mfe", "mfe": mfe},
            )

        # Mirror ratio logic from xgboost_exit_model_adapter.py:
        # profit_decay_pct = (mfe - current_profit) / mfe * 100 when current_profit > 0
        # else profit_decay_pct = 100 + abs(current_profit)
        if current_profit_f > 0:
            profit_decay_pct = max(0.0, (mfe - current_profit_f) / mfe * 100.0)
        else:
            profit_decay_pct = 100.0 + abs(current_profit_f)

        profit_decay_pct = min(profit_decay_pct, 200.0)

        triggered = profit_decay_pct > 70.0

        score = float(profit_decay_pct) / 100.0
        severity = 0
        if triggered:
            severity = 2 if profit_decay_pct < 100.0 else 3

        return FlagResult(
            name=cls.name,
            triggered=bool(triggered),
            score=float(score),
            severity=int(severity),
            reason="profit_decay_exceeds_threshold" if triggered else "profit_decay_below_threshold",
            metadata={
                "profit_decay_pct": profit_decay_pct,
                "current_profit": current_profit_f,
                "mfe": mfe,
                "source_current_profit": "snapshot.trade.profit",
                "source_mfe": "position_state.mfe",
            },
        )

