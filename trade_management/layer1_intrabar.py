"""Layer 1 - Intrabar management.

Two responsibilities, both about candle boundaries:

1. ``can_open_new_entry`` - new entries are only generated once a candle has
   actually closed. Between closes the last stable decision stands.
2. ``IntrabarState`` - tracks the last seen closed-candle timestamp so callers
   can tell "new candle" from "same candle", and counts bars since entry for
   Layer 4.

Broker server time is used, never wall clock: that is what makes this correct
across broker timezone offsets, weekend gaps and DST changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from utils.logger import get_logger

from . import tm_config as C

logger = get_logger("tm.intrabar")

# Seconds per bar, for converting elapsed time into a bar count.
_TF_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "H6": 21600, "H12": 43200,
    "D1": 86400, "W1": 604800,
}


@dataclass
class IntrabarState:
    """Per-symbol memory of the last closed candle seen."""

    last_candle_ts: Dict[str, int] = field(default_factory=dict)

    def is_new_candle(self, symbol: str, candle_ts: Optional[int]) -> bool:
        if candle_ts is None:
            return False
        return self.last_candle_ts.get(symbol) != int(candle_ts)

    def commit(self, symbol: str, candle_ts: Optional[int]) -> None:
        if candle_ts is not None:
            self.last_candle_ts[symbol] = int(candle_ts)


@dataclass(frozen=True)
class IntrabarDecision:
    allow_new_entry: bool
    candle_ts: Optional[int]
    reason: str = ""


def can_open_new_entry(
    symbol: str,
    state: IntrabarState,
    timeframe: Optional[str] = None,
    candle_ts: Optional[int] = None,
) -> IntrabarDecision:
    """Decide whether a fresh entry signal may be acted on for ``symbol``.

    ``candle_ts`` may be injected (tests, replay); otherwise it is read from the
    broker. A missing timestamp blocks new entries rather than falling back to
    wall clock — an unknown candle boundary is not a safe basis for entering.
    """
    tf = timeframe or C.INTRABAR_ENTRY_TIMEFRAME

    if candle_ts is None:
        candle_ts = _fetch_last_closed_candle_ts(symbol, tf)

    if candle_ts is None:
        logger.error(
            "[TM_L1_INTRABAR] %s %s candle boundary unavailable - blocking new entries",
            symbol, tf,
        )
        return IntrabarDecision(False, None, "candle_boundary_unavailable")

    if not state.is_new_candle(symbol, candle_ts):
        return IntrabarDecision(False, candle_ts, "same_candle")

    return IntrabarDecision(True, candle_ts, "new_candle")


def bars_since(open_ts: Optional[float], now_ts: float, timeframe: Optional[str] = None) -> int:
    """Closed candles elapsed since ``open_ts``. Returns 0 on missing input."""
    if not open_ts or not now_ts or now_ts <= open_ts:
        return 0
    tf = str(timeframe or C.INTRABAR_ENTRY_TIMEFRAME).strip().upper()
    seconds = _TF_SECONDS.get(tf)
    if not seconds:
        logger.warning("[TM_L1_INTRABAR] unknown timeframe %s; bars_since -> 0", tf)
        return 0
    return int((float(now_ts) - float(open_ts)) // seconds)


def _fetch_last_closed_candle_ts(symbol: str, timeframe: str) -> Optional[int]:
    try:
        from data.market.candle_boundary import get_last_completed_candle_time

        return get_last_completed_candle_time(symbol, timeframe=timeframe)
    except Exception as exc:
        logger.error("[TM_L1_INTRABAR] candle fetch failed %s %s: %s", symbol, timeframe, exc)
        return None
