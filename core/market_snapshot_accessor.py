from __future__ import annotations

from typing import Any

from data.market.market_snapshot import MarketSnapshot


def get_tf(snapshot: MarketSnapshot, symbol: str, timeframe: str) -> Any:
    """Convenience accessor (read-only)."""
    return snapshot.get(symbol, timeframe)

