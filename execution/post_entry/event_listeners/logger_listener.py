from __future__ import annotations

from utils.logger import get_logger

from ..events import Event

logger = get_logger("post_entry_event_logger")


class LoggerListener:
    def __init__(self) -> None:
        pass

    def __call__(self, event: Event) -> None:
        try:
            logger.info(
                f"[LOGGER_LISTENER] event_type={event.event_type} ticket={event.payload.get('ticket') if isinstance(event.payload, dict) else None}"
            )
        except Exception:
            pass


