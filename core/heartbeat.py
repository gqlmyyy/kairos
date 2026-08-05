from __future__ import annotations

import time
import threading

# Single-process heartbeat for inter-thread coordination.
# If main ever splits into multiple processes, switch to a file/DB based heartbeat.

_last_activity_epoch_sec: float = 0.0
_lock = threading.Lock()


def beat() -> None:
    """Update last activity timestamp to now."""
    global _last_activity_epoch_sec
    with _lock:
        _last_activity_epoch_sec = time.time()


def seconds_since_last_beat() -> float:
    """Return seconds since last beat; inf if never."""
    with _lock:
        if not _last_activity_epoch_sec:
            return float("inf")
        return time.time() - _last_activity_epoch_sec