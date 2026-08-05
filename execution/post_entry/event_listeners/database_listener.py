from __future__ import annotations

from typing import Any

from data.storage.database import upsert_execution_actual

from ..events import Event


class DatabaseListener:
    def __call__(self, event: Event) -> None:
        try:
            from utils.logger import get_logger
            logger = get_logger("post_entry_database_listener")
            logger.info(
                f"[DATABASE_LISTENER] event_type={event.event_type} ticket={event.payload.get('ticket') if isinstance(event.payload, dict) else None}"
            )
            et = event.event_type
            p = event.payload
            if et == "TradeClosed":

                # Minimal persistence based on what manager provides.
                upsert_execution_actual(
                    order_id=str(p.get("order_id")),
                    actual_entry=float(p.get("entry") or 0),
                    actual_exit=float(p.get("exit_price") or 0),
                    actual_pnl=float(p.get("pnl") or 0),
                    exit_reason=str(p.get("exit_reason") or ""),
                    exit_probability=p.get("exit_probability"),
                    execution_quality_score=p.get("execution_quality_score"),
                    slippage=p.get("slippage"),
                    spread_at_entry=p.get("spread_at_entry"),
                    price_gap=p.get("price_gap"),
                    actual_indicators_json=p.get("actual_indicators_json"),
                )
        except Exception:
            pass

