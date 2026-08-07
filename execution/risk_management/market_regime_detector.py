"""Market Regime Detection

ATR/ADX computed from MT5 candles.

Public API (must not change):
  - detect_market_regime(symbol: str, atr: float = None) -> str
  - get_regime_settings(regime: str) -> Dict[str, Any]

Safety:
  - detect_market_regime never raises.
  - detect_market_regime never returns "Unknown" on failure.
  - On any failure to fetch/compute data => return safe default "Normal" and log error.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

from utils.logger import get_logger

from config import (
    MARKET_REGIME_ENABLED,
    REGIME_ADX_THRESHOLD,
    REGIME_LOW_ADX_THRESHOLD,
)

logger = get_logger("market_regime_detector")


def _compute_atr_adx_from_candles(candles: List[Dict[str, Any]], period: int = 14) -> Tuple[Optional[float], Optional[float]]:
    """Best-effort ATR/ADX(14) from candle dicts.

    Expected candle keys (any of these variants accepted):
      - high: c["high"]
      - low:  c["low"]
      - close: c["close"] or c["c"]

    Returns (atr, adx) or (None, None).
    """
    try:
        if not candles or len(candles) < period + 16:
            return None, None

        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []

        for c in candles:
            high_v = c.get("high", None)
            low_v = c.get("low", None)
            close_v = c.get("close", None)
            if close_v is None:
                close_v = c.get("c", None)

            highs.append(float(high_v) if high_v is not None else 0.0)
            lows.append(float(low_v) if low_v is not None else 0.0)
            closes.append(float(close_v) if close_v is not None else 0.0)

        n = len(closes)
        if n < period + 2:
            return None, None

        # True Range
        trs: List[float] = []
        for i in range(n):
            if i == 0:
                trs.append(highs[i] - lows[i])
                continue

            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(float(tr))

        # ATR Wilder smoothing
        if n < period + 1:
            return None, None

        atr = sum(trs[1 : period + 1]) / period
        for i in range(period + 1, n):
            atr = (atr * (period - 1) + trs[i]) / period

        # ADX needs directional movement
        plus_dm: List[float] = []
        minus_dm: List[float] = []
        for i in range(n):
            if i == 0:
                plus_dm.append(0.0)
                minus_dm.append(0.0)
                continue

            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm.append(float(up_move))
                minus_dm.append(0.0)
            elif down_move > up_move and down_move > 0:
                plus_dm.append(0.0)
                minus_dm.append(float(down_move))
            else:
                plus_dm.append(0.0)
                minus_dm.append(0.0)

        # Wilder smoothing init for DI
        tr_sum = sum(trs[1 : period + 1])
        plus_dm_sum = sum(plus_dm[1 : period + 1])
        minus_dm_sum = sum(minus_dm[1 : period + 1])

        if tr_sum == 0:
            return float(atr), 0.0

        # Compute DX series (needs DI)
        dx_list: List[float] = []
        plus_di = 100.0 * (plus_dm_sum / tr_sum)
        minus_di = 100.0 * (minus_dm_sum / tr_sum)
        denom = plus_di + minus_di

        for i in range(period + 1, n):
            # Wilder update
            tr_sum = tr_sum - (tr_sum / period) + trs[i]
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]

            if tr_sum == 0:
                plus_di = 0.0
                minus_di = 0.0
                denom = 0.0
            else:
                plus_di = 100.0 * (plus_dm_sum / tr_sum)
                minus_di = 100.0 * (minus_dm_sum / tr_sum)
                denom = plus_di + minus_di

            if denom == 0:
                dx = 0.0
            else:
                dx = 100.0 * (abs(plus_di - minus_di) / denom)
            dx_list.append(float(dx))

        if not dx_list:
            return float(atr), None

        # ADX smoothing of DX
        if len(dx_list) < period:
            adx = sum(dx_list) / len(dx_list)
            return float(atr), float(adx)

        adx = sum(dx_list[:period]) / period
        for v in dx_list[period:]:
            adx = (adx * (period - 1) + v) / period

        return float(atr), float(adx)

    except Exception:
        return None, None


def _get_atr_adx_from_market(symbol: str, timeframe: str = "H4") -> Tuple[Optional[float], Optional[float]]:
    """Fetch candles from QuantDinger and compute ATR/ADX locally."""
    try:
        # Existing project helper (QuantDinger data source)
        from data.market.client import get_candles  # type: ignore

        candles = get_candles(symbol, timeframe=timeframe, count=120)
        if not candles or len(candles) < 30:
            return None, None

        atr, adx = _compute_atr_adx_from_candles(candles)
        return atr, adx
    except Exception as e:
        logger.error(
            f"[REGIME] QuantDinger ATR/ADX failed symbol={symbol} tf={timeframe} "
            f"exc={type(e).__name__}: {e}"
        )
        return None, None


def detect_market_regime(symbol: str, atr: float = None) -> str:
    """Detect market regime for a symbol (QuantDinger-only, defensive).

    Returns one of: Trending, Weak Trend, Ranging, Volatile, Normal

    Note:
      - Must never return "Unknown".
      - On failure -> return "Normal".
    """
    safe_default = "Normal"

    if not MARKET_REGIME_ENABLED:
        return safe_default

    try:
        sym = str(symbol).strip()
        if not sym:
            return safe_default

        used_atr: Optional[float] = None
        adx: Optional[float] = None

        # ATR may be provided by execution context.
        if atr is not None:
            try:
                atr_f = float(atr)
                if atr_f > 0:
                    used_atr = atr_f
            except Exception:
                used_atr = None

        # Always try to get ADX; if ATR missing compute from QuantDinger.
        q_atr, q_adx = _get_atr_adx_from_market(sym, timeframe="H4")

        if used_atr is None and q_atr is not None and q_atr > 0:
            used_atr = q_atr

        adx = q_adx

        if used_atr is None or used_atr <= 0:
            return safe_default

        # "high ATR" heuristic without MT5 tick: use ATR thresholds only
        # Keep prior heuristic spirit; tuned for common quote magnitudes.
        # If this is wrong for some symbols, regime will degrade to Ranging/Normal safely.
        atr_is_high = used_atr > 0.001

        if adx is not None and adx > 0:
            adx_v = float(adx)

            if adx_v > REGIME_ADX_THRESHOLD:
                return "Trending" if atr_is_high else "Weak Trend"
            if adx_v < REGIME_LOW_ADX_THRESHOLD:
                return "Ranging"
            return "Volatile"

        # No ADX => use ATR-only fallback
        return "Volatile" if atr_is_high else "Ranging"

    except Exception as e:
        logger.error(
            f"[REGIME] detect_market_regime failed symbol={symbol} "
            f"exc={type(e).__name__}: {e}"
        )
        return safe_default


def get_regime_settings(regime: str) -> Dict[str, Any]:
    """Return regime-specific settings.

    Must return valid defaults even when regime is "Normal" or unknown.
    """
    try:
        r = str(regime).strip().lower()

        if r == "trending":
            return {"trailing_multiplier": 1.5, "tp_ratio": 0.8}
        if r == "weak trend":
            return {"trailing_multiplier": 1.0, "tp_ratio": 0.7}
        if r == "ranging":
            return {"trailing_multiplier": 0.8, "tp_ratio": 0.6}
        if r == "volatile":
            return {"trailing_multiplier": 1.2, "tp_ratio": 0.65}

        # Default for "Normal" / Unknown / other
        return {"trailing_multiplier": 2.0, "tp_ratio": 0.5}
    except Exception:
        return {"trailing_multiplier": 2.0, "tp_ratio": 0.5}

