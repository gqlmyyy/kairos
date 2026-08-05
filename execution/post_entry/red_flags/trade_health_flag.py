from __future__ import annotations

from typing import Any, Dict, Optional

from .flag_result import FlagResult


class TradeHealthFlag:
    """Trade Health red-flag.

    Source of truth (raw, documented via adapter + builder):
      - snapshot['expected_row']['p_win'] (db_row from get_execution_dataset)

    If p_win is missing/non-numeric: triggered=False with explicit reason.
    """

    name = "TradeHealth"

    @classmethod
    def check(
        cls,
        snapshot: Dict[str, Any],
        position_state: Optional[Any] = None,
    ) -> FlagResult:
        expected_row = snapshot.get("expected_row") or {}

        if "p_win" not in expected_row:
            return FlagResult(
                name=cls.name,
                triggered=False,
                score=0.0,
                severity=0,
                reason="p_win missing from expected_row",
                metadata={"source": "snapshot.expected_row.p_win"},
            )

        p_win = expected_row.get("p_win")
        if p_win is None:
            return FlagResult(
                name=cls.name,
                triggered=False,
                score=0.0,
                severity=0,
                reason="p_win is None",
                metadata={"source": "snapshot.expected_row.p_win"},
            )

        try:
            p_win_f = float(p_win)
        except Exception:
            return FlagResult(
                name=cls.name,
                triggered=False,
                score=0.0,
                severity=0,
                reason="p_win not numeric",
                metadata={"source": "snapshot.expected_row.p_win", "p_win": p_win},
            )

        # Mirrors legacy health_score approach from trade_health_rule.py and adapter docs.
        health_score = p_win_f * 100.0

        # Conservative: trigger under 40 (consistent with adapter fallback notes).
        triggered = health_score < 40.0

        score = float(health_score) / 100.0
        severity = 0
        if triggered:
            severity = 2 if health_score < 20.0 else 1

        return FlagResult(
            name=cls.name,
            triggered=bool(triggered),
            score=float(score),
            severity=int(severity),
            reason="trade_health_below_threshold" if triggered else "trade_health_ok",
            metadata={
                "p_win": p_win_f,
                "health_score": health_score,
                "source": "snapshot.expected_row.p_win",
            },
        )

