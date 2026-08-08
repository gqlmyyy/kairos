from __future__ import annotations

import time
from typing import Any, Dict, List

from utils.logger import get_logger

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    mt5 = None

logger = get_logger("post_entry_trade_monitor")


class TradeMonitor:
    """Reads open positions and computes required online metrics.

    No decision-making here.
    """

    def __init__(self) -> None:
        pass

    def get_open_positions(self) -> List[Dict[str, Any]]:
        if mt5 is None:
            return []
        # Runs every POST_ENTRY_LOOP_INTERVAL_SEC (5s) alongside the main
        # cycle, reconciliation and the watchdog, so it must take the shared
        # session lock rather than racing them on the single IPC channel.
        from data.market.mt5_session import mt5_call

        with mt5_call():
            positions = mt5.positions_get()
        out: List[Dict[str, Any]] = []
        if not positions:
            return out
        try:
            for p in positions:
                d = p._asdict() if hasattr(p, "_asdict") else p.__dict__
                symbol = d.get("symbol") or d.get("Symbol")
                ticket = d.get("ticket") or d.get("position_id") or d.get("id")
                if not symbol or ticket is None:
                    continue
                ptype = d.get("type")
                direction = "buy" if str(ptype).lower() in ["0", "buy", "long", "1"] else "sell"
                now_ts = int(time.time())
                logger.info(
                    f"[MONITOR_ID_DIAG] ts={now_ts} symbol={symbol} d_ticket={d.get('ticket')} d_position_id={d.get('position_id')} d_id={d.get('id')} chosen_ticket={ticket}"
                )
                out.append(
                    {
                        "order_id": str(ticket),
                        "symbol": str(symbol),
                        "direction": direction,
                        "volume": float(d.get("volume") or 0),
                        "entry_price": float(d.get("price_open") or d.get("price") or 0),
                        "sl": float(d.get("sl") or 0),
                        "tp": float(d.get("tp") or 0) if d.get("tp") is not None else None,
                        "profit": float(d.get("profit") or d.get("pnl") or 0),
                        "price_current": float(d.get("price_current") or d.get("price") or 0),
                        "comment": d.get("comment") or "",
                        # MT5 uses "time" for open time, not "time_open"
                        "time_open": d.get("time") or d.get("time_open") or None,
                        # Spread from MT5 - points (not pips)
                        "spread": float(d.get("spread") or 0),
                    }
                )
        except Exception:
            return []
        return out

