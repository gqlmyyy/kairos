from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class MarketSnapshot:
    """Immutable-ish market data snapshot for a single cycle.

    snapshot.data layout:
        {
            "EURUSD": {"H4": {...}, "H1": {...}, "M15": {...}},
            ...
        }
    """

    data: Dict[str, Dict[str, Any]]

    def get(self, symbol: str, timeframe: str) -> Any:
        return self.data.get(symbol, {}).get(timeframe, None)

