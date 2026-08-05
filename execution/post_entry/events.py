from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List


@dataclass(frozen=True)
class Event:
    event_type: str
    ts: float
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeClosedEvent:
    order_id: str
    symbol: str
    direction: str
    pnl: float
    exit_reason: str = ""
    # Optional fields used for accurate Telegram payloads
    size: Optional[float] = None
    entry: Optional[float] = None
    exit_price: Optional[float] = None

    decision_score: Optional[float] = None
    rule_score: Optional[float] = None
    model_score: Optional[float] = None
    confidence: Optional[float] = None
    ts: float = 0.0

    def to_event(self) -> Event:
        return Event(
            event_type="TradeClosed",
            ts=self.ts,
            payload={
                "ticket": self.order_id,
                "order_id": self.order_id,
                "symbol": self.symbol,
                "direction": self.direction,
                "pnl": self.pnl,
                "exit_reason": self.exit_reason,
                "size": self.size,
                "entry": self.entry,
                "exit_price": self.exit_price,
                "decision_score": self.decision_score,
                "rule_score": self.rule_score,
                "model_score": self.model_score,
                "confidence": self.confidence,
            },
        )


@dataclass(frozen=True)
class SLModifiedEvent:
    ticket: str
    symbol: str
    direction: str
    old_sl: float
    new_sl: float
    entry_price: float
    reason: str
    ts: float = 0.0

    def to_event(self) -> Event:
        return Event(
            event_type="SLModified",
            ts=self.ts,
            payload={
                "ticket": self.ticket,
                "order_id": self.ticket,
                "symbol": self.symbol,
                "direction": self.direction,
                "old_sl": self.old_sl,
                "new_sl": self.new_sl,
                "entry_price": self.entry_price,
                "reason": self.reason,
            },
        )


@dataclass(frozen=True)
class TPModifiedEvent:
    order_id: str
    symbol: str
    direction: str
    new_tp: float
    ts: float = 0.0

    def to_event(self) -> Event:
        return Event(
            event_type="TPModified",
            ts=self.ts,
            payload={
                "ticket": self.order_id,
                "order_id": self.order_id,
                "symbol": self.symbol,
                "direction": self.direction,
                "new_tp": self.new_tp,
            },
        )


@dataclass(frozen=True)
class PartialClosedEvent:
    order_id: str
    symbol: str
    direction: str
    closed_volume: float
    ts: float = 0.0

    def to_event(self) -> Event:
        return Event(
            event_type="PartialClosed",
            ts=self.ts,
            payload={
                "ticket": self.order_id,
                "order_id": self.order_id,
                "symbol": self.symbol,
                "direction": self.direction,
                "closed_volume": self.closed_volume,
            },
        )

