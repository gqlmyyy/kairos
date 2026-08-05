# Trading Bot V3 - analysis/models/performance_monitor.py

from __future__ import annotations

import collections
import math
from typing import Deque, Dict, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("performance_monitor")


class PerformanceMonitor:
    """Track online inference performance from actual outcomes.

    This is an in-memory tracker. If you want persistence, store into SQLite.
    """

    def __init__(self, rolling_n: int = 50):
        self.rolling_n = rolling_n
        # each item: (p_win, actual_outcome_binary, actual_pnl)
        self._buf: Deque[Tuple[float, Optional[float], Optional[float]]] = collections.deque(maxlen=rolling_n)

    def record(self, p_win: float, actual_pnl: Optional[float]):
        if actual_pnl is None:
            return
        outcome = 1.0 if actual_pnl > 0 else 0.0
        self._buf.append((float(p_win), outcome, float(actual_pnl)))

    def snapshot(self) -> Dict[str, float]:
        if not self._buf:
            return {
                "n": 0,
                "rolling_win_rate": 0.0,
                "avg_p_win": 0.0,
                "profit_factor_approx": 0.0,
                "drawdown_approx": 0.0,
            }

        n = len(self._buf)
        wins = sum(1 for _p, outcome, _pnl in self._buf if outcome == 1.0)
        avg_p_win = sum(p for p, _o, _pnl in self._buf) / n
        total_profit = sum(pnl for _p, _o, pnl in self._buf if pnl is not None and pnl > 0)
        total_loss = abs(sum(pnl for _p, _o, pnl in self._buf if pnl is not None and pnl < 0))
        profit_factor = (total_profit / total_loss) if total_loss > 0 else (total_profit if total_profit > 0 else 0.0)

        # drawdown approx on cumulative pnl over buffer
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for _p, _o, pnl in self._buf:
            if pnl is None:
                continue
            cum += pnl
            peak = max(peak, cum)
            dd = peak - cum
            max_dd = max(max_dd, dd)

        return {
            "n": float(n),
            "rolling_win_rate": wins / n if n else 0.0,
            "avg_p_win": avg_p_win,
            "profit_factor_approx": float(profit_factor),
            "drawdown_approx": float(max_dd),
        }


_global_monitor: Optional[PerformanceMonitor] = None


def get_global_monitor() -> PerformanceMonitor:
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(rolling_n=50)
    return _global_monitor


def get_live_metrics() -> Dict[str, float]:
    """Return live rolling metrics from the global monitor."""
    try:
        return get_global_monitor().snapshot()
    except Exception:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_p_win": 0.0,
            "drawdown": 0.0,
        }


