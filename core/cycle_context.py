from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from data.market.market_snapshot import MarketSnapshot


@dataclass(frozen=True)
class CycleContext:
    """Holds cycle-scoped shared references (read-only contract)."""

    snapshot: MarketSnapshot
    # Optional metadata (not required by the rules)
    cycle_id: Optional[int] = None
    symbol_group: Optional[str] = None


def make_cycle_context(snapshot: MarketSnapshot, cycle_id: Optional[int] = None) -> CycleContext:
    return CycleContext(snapshot=snapshot, cycle_id=cycle_id)

