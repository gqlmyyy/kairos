from __future__ import annotations

from typing import Dict, Any

from data.storage.database import save_performance


class PerformanceRecorder:
    def __init__(self) -> None:
        pass

    def record_on_close(self, closed_event_payload: Dict[str, Any]) -> None:
        # Minimal placeholder: full metrics require deeper reconciliation.
        # Kept non-breaking.
        try:
            symbol = closed_event_payload.get("symbol")
            pnl = float(closed_event_payload.get("pnl") or 0)
            # Save minimal metrics if possible
            save_performance(symbol=symbol, batch_size=0, win_rate=0, profit_factor=0, avg_win=0, avg_loss=0, sharpe=0)
        except Exception:
            pass

