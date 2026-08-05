from __future__ import annotations

from telegram.notifier import (
    send,
    notify_alert,
    notify_trade_closed,
)

from ..events import Event


class TelegramListener:
    """Translate events to Telegram messages.

    IMPORTANT: No Telegram calls should exist elsewhere than this listener.
    """

    def __call__(self, event: Event) -> None:
        try:
            from utils.logger import get_logger
            _lg = get_logger("post_entry_telegram_listener")
            _lg.info(
                f"[TELEGRAM_LISTENER] event_type={event.event_type} ticket={event.payload.get('ticket') if isinstance(event.payload, dict) else None}"
            )
            et = event.event_type

            p = event.payload
            if et == "SLModified":
                # Required payload keys (per Task #2)
                ticket = p.get("ticket") or p.get("order_id")
                symbol = p.get("symbol")
                direction = p.get("direction")
                new_sl = p.get("new_sl")
                entry_price = p.get("entry_price")

                # Compute delta in points (fallback point_size=0.0001 if MT5 point not available here)
                try:
                    point_size = 0.0001
                    if entry_price is not None and new_sl is not None and float(point_size) > 0:
                        points = (float(new_sl) - float(entry_price)) / float(point_size)
                    else:
                        points = 0.0
                except Exception:
                    points = 0.0

                # Determine "zone": according to requested wording (+X points) we use points sign
                # If direction isn't buy/sell, still follow points sign.
                if points >= 0:
                    zone = f"منطقة ربح (+{points:.0f} نقاط)"
                else:
                    zone = f"منطقة خسارة ({points:.0f} نقاط)"

                reason = p.get("reason") or "Stop Loss Modified"

                send(
                    "🛡️ تعديل وقف الخسارة\n"
                    f"{symbol} | {direction}\n"
                    f"الوقف الجديد: {new_sl}\n"
                    f"الحالة: {zone}\n"
                    f"السبب: {reason}"
                )
            elif et == "TPModified":
                send(f"🎯 TP Updated: {p.get('symbol')} {p.get('direction')} -> {p.get('new_tp')}")
            elif et == "PartialClosed":
                send(f"✂️ Partial Close: {p.get('symbol')} vol={p.get('closed_volume')}")
            elif et == "TradeClosed":
                notify_trade_closed(
                    symbol=p.get("symbol"),
                    direction=p.get("direction"),
                    pnl=float(p.get("pnl") or 0),
                    reason=p.get("exit_reason") or "",
                    size=p.get("size") or 0,
                    entry=p.get("entry") or 0,
                    exit_price=p.get("exit_price") or 0,
                )
            else:
                # no-op by default
                pass
        except Exception:
            pass

