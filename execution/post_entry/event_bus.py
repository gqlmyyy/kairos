from __future__ import annotations

import time
from typing import Callable, Dict, List, Any

from .events import Event


class EventBus:
    def __init__(self) -> None:
        self._listeners: Dict[str, List[Callable[[Event], Any]]] = {}

    def register(self, event_type: str, handler: Callable[[Event], Any]) -> None:
        self._listeners.setdefault(event_type, []).append(handler)

    def publish(self, event: Event) -> None:
        handlers = list(self._listeners.get(event.event_type, []))
        for h in handlers:
            try:
                h(event)
            except Exception:
                # fail-safe: do not break management loop
                pass


def default_event_bus() -> EventBus:
    bus = EventBus()
    return bus


class DedupEventBus(EventBus):
    """Optional dedup layer for noisy events."""

    def __init__(self, ttl_sec: float = 10.0) -> None:
        super().__init__()
        self._ttl_sec = float(ttl_sec)
        self._seen: Dict[str, float] = {}

    def publish(self, event: Event) -> None:
        key = f"{event.event_type}:{event.payload.get('order_id','')}:{event.payload.get('new_sl','')}:{event.payload.get('new_tp','')}"
        now = time.time()
        old = self._seen.get(key)
        if old is not None and now - old <= self._ttl_sec:
            return
        self._seen[key] = now
        super().publish(event)

