from __future__ import annotations

from typing import Any, Dict, Iterable, List

from utils.logger import get_logger
from config import TF_TREND, TF_DECISION, TF_TIMING
from data.market.hybrid_client import get_indicators_hybrid
from data.market.market_snapshot import MarketSnapshot

logger = get_logger("market_snapshot_builder")


class MarketSnapshotBuilder:
    """Single source of truth builder.

    All modules must receive the built snapshot and must not perform market fetches.

    Cache scope: only inside this builder instance.
    """

    def __init__(self):
        # cache_key: (symbol, timeframe)
        self._cache: Dict[tuple[str, str], Any] = {}

    def _fetch_symbol_timeframe(self, symbol: str, timeframe: str) -> Any:
        key = (symbol, timeframe)
        if key in self._cache:
            return self._cache[key]

        # Only this builder performs network/database fetches.
        data = get_indicators_hybrid(symbol, timeframe=timeframe)
        self._cache[key] = data
        return data

    def build(
        self,
        symbols: Iterable[str],
        timeframes: List[str] | None = None,
    ) -> MarketSnapshot:
        if timeframes is None:
            timeframes = [TF_TREND, TF_DECISION, TF_TIMING]

        snapshot_data: Dict[str, Dict[str, Any]] = {}

        for symbol in symbols:
            snapshot_data[symbol] = {}
            for tf in timeframes:
                snapshot_data[symbol][tf] = self._fetch_symbol_timeframe(symbol, tf)

            logger.info(
                "[SNAPSHOT_BUILT] symbol=%s tfs=%s",
                symbol,
                ",".join(timeframes),
            )

        return MarketSnapshot(data=snapshot_data)

