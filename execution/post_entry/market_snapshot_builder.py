from __future__ import annotations

import time
from typing import Any, Dict, Optional


class MarketSnapshotBuilder:
    """Builds unified market + trade snapshot with caching.

    Classification of data during trade lifecycle:
    - STATIC (cache per ticket): symbol, volume, direction, entry_price, sl, tp, order_id
    - PER-CANDLE (cache per H1/H4): atr, rsi, adx, ema, market_regime
    - TICK (fresh every cycle): profit, price_current, spread, time_open_hours

    This implementation caches static and per-candle data to avoid recalculation.
    """

    def __init__(self) -> None:
        # Cache: ticket -> static fields (never change during trade)
        self._static_cache: Dict[int, Dict[str, Any]] = {}

        # Cache: ticket -> per-candle data with timestamp
        self._candle_cache: Dict[int, Dict[str, Any]] = {}
        self._candle_cache_time: Dict[int, float] = {}

        # Cache TTL for per-candle data (15 minutes - matches H4 candle mostly)
        self._CANDLE_CACHE_TTL_SEC = 900

    def _get_candle_timestamp(self) -> float:
        """Get current candle window timestamp for caching.

        Aligns to H1 boundaries to reduce API calls.
        """
        return int(time.time() / 3600) * 3600

    def get_static_fields(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and cache static fields (never change during trade lifecycle).

        These fields are constant from open until close.
        """
        ticket = int(trade.get("order_id") or 0)
        if not ticket:
            # Fallback: extract directly
            return {
                "symbol": trade.get("symbol"),
                "direction": trade.get("direction"),
                "volume": trade.get("volume"),
                "entry_price": trade.get("entry_price"),
                "sl": trade.get("sl"),
                "tp": trade.get("tp"),
                "order_id": trade.get("order_id"),
            }

        # Check cache
        if ticket in self._static_cache:
            return self._static_cache[ticket]

        # Build and cache
        static = {
            "symbol": trade.get("symbol"),
            "direction": trade.get("direction"),
            "volume": trade.get("volume"),
            "entry_price": trade.get("entry_price"),
            "sl": trade.get("sl"),
            "tp": trade.get("tp"),
            "order_id": trade.get("order_id"),
        }
        self._static_cache[ticket] = static
        return static

    def get_candle_fields(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Get per-candle fields with caching.

        ATR, RSI, ADX, EMA only change at candle boundaries (H1/H4).
        Cache for up to 15 minutes to reduce API calls.
        """
        ticket = int(trade.get("order_id") or 0)
        if not ticket:
            # No cache - return from trade dict
            return {
                "atr": trade.get("atr"),
                "rsi": trade.get("rsi"),
                "adx": trade.get("adx"),
                "ema": trade.get("ema"),
                "trend": trade.get("trend"),
                "market_regime": trade.get("market_regime") or "UNKNOWN",
            }

        now = time.time()
        cached_time = self._candle_cache_time.get(ticket, 0)

        # Check if cache is still valid
        if ticket in self._candle_cache and (now - cached_time) < self._CANDLE_CACHE_TTL_SEC:
            return self._candle_cache[ticket]

        # Build fresh candle data
        candle_data = {
            "atr": trade.get("atr"),
            "rsi": trade.get("rsi"),
            "adx": trade.get("adx"),
            "ema": trade.get("ema"),
            "trend": trade.get("trend"),
            "market_regime": trade.get("market_regime") or "UNKNOWN",
        }

        # Cache it
        self._candle_cache[ticket] = candle_data
        self._candle_cache_time[ticket] = now

        return candle_data

    def build_snapshot(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Build unified market + trade snapshot.

        Optimized to avoid recalculating static/candle data every cycle.
        """
        # Get cached static fields
        static = self.get_static_fields(trade)

        # Get cached candle fields
        candle = self.get_candle_fields(trade)

        # Get dynamic fields (always fresh)
        dynamic = {
            "profit": trade.get("profit"),
            "price_current": trade.get("price_current"),
            "spread": trade.get("spread") or 0.0,
            "time_open": trade.get("time_open"),
        }

        # Build snapshot
        snap: Dict[str, Any] = {
            "trade": trade,
            # Static fields (cached)
            "symbol": static.get("symbol"),
            "direction": static.get("direction"),
            "volume": static.get("volume"),
            "entry_price": static.get("entry_price"),
            "sl": static.get("sl"),
            "tp": static.get("tp"),
            # Candle fields (cached)
            "atr": candle.get("atr"),
            "rsi": candle.get("rsi"),
            "adx": candle.get("adx"),
            "ema": candle.get("ema"),
            "trend": candle.get("trend"),
            "market_regime": candle.get("market_regime"),
            # Dynamic fields (fresh every tick)
            "profit": dynamic.get("profit"),
            "price_current": dynamic.get("price_current"),
            "spread": dynamic.get("spread"),
            "time_open": dynamic.get("time_open"),
            # Portfolio/market status (from trade dict or defaults)
            "news_status": trade.get("news_status") or "unknown",
            "portfolio_exposure": trade.get("portfolio_exposure") or 0.0,
            "equity": trade.get("equity") or None,
            "drawdown": trade.get("drawdown") or None,
            "correlation": trade.get("correlation") or None,
        }

        return snap

    def invalidate_cache(self, ticket: int) -> None:
        """Clear cache for a closed position.

        Call this when a trade is closed to free memory.
        """
        self._static_cache.pop(ticket, None)
        self._candle_cache.pop(ticket, None)
        self._candle_cache_time.pop(ticket, None)

    def clear_all_caches(self) -> None:
        """Clear all caches - useful for testing or memory management."""
        self._static_cache.clear()
        self._candle_cache.clear()
        self._candle_cache_time.clear()