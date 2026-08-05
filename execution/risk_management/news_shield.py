"""News Shield

Prevents trading during high-impact news windows.

This module is defensive and should not break reconciliation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

from utils.logger import get_logger

from config import (
    NEWS_SHIELD_ENABLED,
    NEWS_SHIELD_MINUTES_BEFORE,
    NEWS_SHIELD_CLOSE_TRADES,
)

logger = get_logger("news_shield")

# global activation flag for the current runtime
news_shield_active: bool = False


def is_high_impact_news_now(symbol: str) -> bool:
    """Best-effort high impact news detector.

    Strategy:
    - Try to use fetch_rss_news() items and check impact/confidence/score fields.
    - Fallback: if no reliable scheduling fields exist, return False.
    """
    if not NEWS_SHIELD_ENABLED:
        return False

    try:
        from data.news.fetcher import fetch_rss_news  # type: ignore

        news = fetch_rss_news()
        if not news:
            return False


        symbol_u = str(symbol).upper() if symbol else ""
        horizon_sec = float(NEWS_SHIELD_MINUTES_BEFORE) * 60.0
        now = time.time()

        for item in news:
            try:
                item_symbol = str(item.get("symbol", "") or item.get("ticker", "") or "").upper()
                if symbol_u and item_symbol and item_symbol != symbol_u:
                    continue

                # 1) explicit impact/score/confidence
                for key in ("impact_score", "confidence", "score"):
                    v = item.get(key)
                    if v is None:
                        continue
                    try:
                        if float(v) > 80:
                            return True
                    except Exception:
                        pass

                # 2) explicit time field if present
                dt = item.get("time") or item.get("datetime") or item.get("published")
                if dt is not None:
                    try:
                        dt_f = float(dt)
                        if 0 <= (dt_f - now) <= horizon_sec:
                            return True
                    except Exception:
                        pass

            except Exception:
                continue

        return False

    except Exception as e:
        # Fail-closed: if we cannot reliably check news, do not allow trading.
        logger.warning(f"[NEWS_SHIELD] Exception during high-impact check => fail-closed: {e}")
        return True



def apply_news_shield(
    open_positions: Iterable[Dict[str, Any]],
    news_data: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Apply the shield behavior.

    - If NEWS_SHIELD_CLOSE_TRADES is True: close all open positions.
    - Always set news_shield_active True.
    """
    global news_shield_active

    if not NEWS_SHIELD_ENABLED:
        news_shield_active = False
        return False

    news_shield_active = True

    # optional close
    if NEWS_SHIELD_CLOSE_TRADES:
        try:
            from execution.reconciliation import _close_trade_mt5  # type: ignore

            for pos in open_positions or []:
                try:
                    oid = str(pos.get("id", pos.get("ticket", "")) or "")
                    if oid:
                        _close_trade_mt5(oid)
                except Exception:
                    continue
        except Exception:
            pass

    # log (best-effort symbol)
    try:
        first = next(iter(open_positions or []), None)
        sym = str(first.get("symbol", "(multi)") if first else "(multi)")
    except Exception:
        sym = "(multi)"

    logger.warning(f"[NEWS_SHIELD] Activated for {sym}. No new trades allowed.")
    return True

