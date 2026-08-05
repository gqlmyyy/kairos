"""analysis/features/historical_dataset_builder.py

Build a historical execution_dataset from QuantDinger only (no CSV, no external data).

Contract:
- Fetch 500 H4 candles per symbol using data.market.client.get_candles()
- Compute indicators directly from candle OHLC data (no external TA libs)
- Create labels by simulating TP/SL hit order inside next 20 candles
- Save rows into execution_dataset via upsert_execution_expected()

This module is intended to be runnable immediately.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from statistics import pstdev
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger
from config import SYMBOLS
from data.market.client import get_candles
from data.storage.database import upsert_execution_expected

logger = get_logger("historical_dataset_builder")


# ----------------------------
# Candle parsing
# ----------------------------

def _parse_candle(c: Dict) -> Optional[Dict[str, float]]:
    """Expected candle fields: time, open, high, low, close."""
    try:
        t = c.get("time", None)
        if t is None:
            return None

        # QuantDinger sometimes uses aliases
        o = c.get("open", c.get("o", None))
        h = c.get("high", c.get("h", None))
        l = c.get("low", c.get("l", None))
        cl = c.get("close", c.get("c", None))
        if o is None or h is None or l is None or cl is None:
            return None

        return {
            "time": float(t),
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(cl),
        }
    except Exception:
        return None


def _session_from_utc(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    h = dt.hour
    # Approx mapping
    if 0 <= h < 7:
        return "asia"
    if 7 <= h < 13:
        return "london"
    if 13 <= h < 20:
        return "newyork"
    return "asia"


# ----------------------------
# Indicator math (no TA libs)
# ----------------------------

def _ema(series: List[float], period: int) -> List[Optional[float]]:
    """Standard EMA; first EMA value starts at index=period-1 with SMA."""
    out: List[Optional[float]] = [None] * len(series)
    if len(series) < period:
        return out

    sma = sum(series[:period]) / period
    out[period - 1] = sma

    alpha = 2.0 / (period + 1)
    for i in range(period, len(series)):
        prev = out[i - 1]
        if prev is None:
            # should not happen after initialization
            prev = series[i - 1]
        out[i] = alpha * series[i] + (1.0 - alpha) * prev
    return out


def _rsi_14(closes: List[float], period: int = 14) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(closes)
    if len(closes) < period + 1:
        return out

    # Use Wilder smoothing
    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(diff))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    rs = (avg_gain / avg_loss) if avg_loss != 0 else float("inf")
    rsi = 100.0 - (100.0 / (1.0 + rs))
    out[period] = rsi

    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gain = diff if diff > 0 else 0.0
        loss = abs(diff) if diff < 0 else 0.0

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        rs = (avg_gain / avg_loss) if avg_loss != 0 else float("inf")
        rsi = 100.0 - (100.0 / (1.0 + rs))
        out[i] = rsi

    return out


def _atr_14(candles: List[Dict[str, float]], period: int = 14) -> List[Optional[float]]:
    """ATR using Wilder smoothing on True Range."""
    out: List[Optional[float]] = [None] * len(candles)
    if len(candles) < period + 1:
        return out

    trs: List[float] = []
    for i in range(1, period + 1):
        prev_close = candles[i - 1]["close"]
        high = candles[i]["high"]
        low = candles[i]["low"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    atr = sum(trs) / period
    out[period] = atr

    for i in range(period + 1, len(candles)):
        prev_close = candles[i - 1]["close"]
        high = candles[i]["high"]
        low = candles[i]["low"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        atr = (atr * (period - 1) + tr) / period
        out[i] = atr

    return out


def _sma(series: List[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(series)
    if len(series) < period:
        return out
    window_sum = sum(series[:period])
    out[period - 1] = window_sum / period

    for i in range(period, len(series)):
        window_sum += series[i] - series[i - period]
        out[i] = window_sum / period
    return out


def _macd_12_26_9(closes: List[float]) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd: List[Optional[float]] = [None] * len(closes)

    for i in range(len(closes)):
        if ema12[i] is None or ema26[i] is None:
            macd[i] = None
        else:
            macd[i] = ema12[i] - ema26[i]

    # Signal line is EMA of MACD values
    # Build a dense list of floats with None -> skip by carrying forward until enough values
    macd_series: List[float] = [v if v is not None else 0.0 for v in macd]
    signal_dense = _ema(macd_series, 9)

    signal: List[Optional[float]] = [None] * len(closes)
    for i in range(len(closes)):
        if macd[i] is None or signal_dense[i] is None:
            signal[i] = None
        else:
            signal[i] = signal_dense[i]

    hist: List[Optional[float]] = [None] * len(closes)
    for i in range(len(closes)):
        if macd[i] is None or signal[i] is None:
            hist[i] = None
        else:
            hist[i] = macd[i] - signal[i]

    return macd, signal, hist


def _volatility_std(closes: List[float], lookback: int = 20) -> List[Optional[float]]:
    """Std of last N returns."""
    out: List[Optional[float]] = [None] * len(closes)
    returns: List[Optional[float]] = [None] * len(closes)
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev == 0:
            returns[i] = 0.0
        else:
            returns[i] = (closes[i] - prev) / prev

    for i in range(lookback, len(closes)):
        window = [r for r in returns[i - lookback + 1 : i + 1] if r is not None]
        if len(window) != lookback:
            out[i] = None
        else:
            # population std
            out[i] = pstdev(window)

    return out


def _trend_strength_ma_slope(ma20: List[Optional[float]], idx: int) -> Optional[float]:
    """Slope proxy: (MA20(i) - MA20(i-5)) / 5."""
    if idx < 5:
        return None
    if ma20[idx] is None or ma20[idx - 5] is None:
        return None
    return (ma20[idx] - ma20[idx - 5]) / 5.0


def _normalize_0_1(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.5
    v = (x - lo) / (hi - lo)
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return v


def _normalize_volatility(std_ret: float) -> float:
    # Heuristic scaling to [0,1] without external quantiles.
    # For FX-like std of returns, typical values are small; clamp with log scaling.
    # std_ret >=0
    if std_ret <= 0:
        return 0.0
    # Map using log1p for stability
    v = math.log1p(std_ret * 1000.0)
    return _normalize_0_1(v, 0.0, 6.0)


def _score_from_slope(slope: float) -> float:
    """Convert slope to 0..100."""
    # slope can be negative/positive. Use tanh-like squashing.
    if slope is None:
        return 50.0
    # scale factor heuristic
    scaled = slope * 10000.0
    v = math.tanh(scaled)  # [-1,1]
    return (v + 1.0) * 50.0


# ----------------------------
# Labeling simulation
# ----------------------------

def _simulate_tp_sl(candles: List[Dict[str, float]], i: int, atr: float) -> Tuple[Optional[int], Optional[str]]:
    """Return (label, exit_reason) where label 1=WIN,0=LOSS.

    SL = ATR * 1.5
    TP = ATR * 2.5

    We assume long/short decided later; TP/SL are applied directionally.
    Here we just detect whether price reaches relative TP/SL first using high/low.
    """
    entry = candles[i]["close"]
    sl = atr * 1.5
    tp = atr * 2.5
    if sl <= 0 or tp <= 0:
        return None, None

    # We don't know direction here; we will check both possibilities later.
    # To keep contract simple, we'll label direction-dependent in caller.
    return None, None


def _simulate_for_direction(
    candles: List[Dict[str, float]],
    i: int,
    atr: float,
    direction: str,
    horizon: int = 20,
) -> Tuple[Optional[int], Optional[str]]:
    entry = candles[i]["close"]
    sl_dist = atr * 1.5
    tp_dist = atr * 2.5
    if sl_dist <= 0 or tp_dist <= 0:
        return None, None

    if direction == "BUY":
        tp_price = entry + tp_dist
        sl_price = entry - sl_dist
        for j in range(i + 1, min(i + horizon + 1, len(candles))):
            hi = candles[j]["high"]
            lo = candles[j]["low"]
            # priority by first hit: compare order by scanning candles sequentially
            if hi >= tp_price:
                return 1, "TP"
            if lo <= sl_price:
                return 0, "SL"
        return None, None

    if direction == "SELL":
        tp_price = entry - tp_dist
        sl_price = entry + sl_dist
        for j in range(i + 1, min(i + horizon + 1, len(candles))):
            hi = candles[j]["high"]
            lo = candles[j]["low"]
            if lo <= tp_price:
                return 1, "TP"
            if hi >= sl_price:
                return 0, "SL"
        return None, None

    return None, None


# ----------------------------
# Main builder
# ----------------------------

def build_historical_dataset(
    count: int = 500,
    horizon: int = 20,
    min_rows_before_labels: int = 60,
) -> None:
    """Build and upsert rows into execution_dataset."""

    # Fetch and build indicators per symbol
    for symbol in SYMBOLS:
        logger.info(f"Fetching historical candles for {symbol} (H4, count={count})...")
        raw = get_candles(symbol, "H4", count=count)

        candles: List[Dict[str, float]] = []
        for c in raw:
            parsed = _parse_candle(c)
            if parsed is not None:
                candles.append(parsed)

        if len(candles) < 200:
            logger.warning(f"Not enough candles for {symbol}: {len(candles)} (skip)")
            continue

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]

        rsi_list = _rsi_14(closes, 14)
        atr_list = _atr_14(candles, 14)
        macd_list, signal_list, hist_list = _macd_12_26_9(closes)

        ma20_list = _sma(closes, 20)
        ma50_list = _sma(closes, 50)

        momentum_list: List[Optional[float]] = [None] * len(closes)
        for i in range(len(closes)):
            if i - 10 >= 0:
                momentum_list[i] = closes[i] - closes[i - 10]

        vol_std_list = _volatility_std(closes, 20)

        # Labels: iterate indices with enough lookback
        saved = 0
        for i in range(len(candles)):
            if i < max(50, 14, 26) or i < min_rows_before_labels:
                continue

            atr = atr_list[i]
            ma20 = ma20_list[i]
            ma50 = ma50_list[i]
            rsi = rsi_list[i]
            macd = macd_list[i]
            momentum = momentum_list[i]
            vol_std = vol_std_list[i]

            if atr is None or ma20 is None or ma50 is None or rsi is None or macd is None or momentum is None or vol_std is None:
                continue

            # Direction rule from user:
            if ma20 > ma50 and momentum > 0:
                direction = "BUY"
            elif ma20 < ma50 and momentum < 0:
                direction = "SELL"
            else:
                continue

            # Simulate TP/SL
            label, reason = _simulate_for_direction(candles, i, atr, direction, horizon=horizon)
            if label is None or reason is None:
                continue

            entry = candles[i]["close"]
            sl_dist = atr * 1.5
            tp_dist = atr * 2.5

            # required fields (no None)
            expected_session = _session_from_utc(candles[i]["time"])

            # expected_trend_strength = MA slope (raw) -> we store as slope score 0-100
            slope = _trend_strength_ma_slope(ma20_list, i)
            trend_score_0_100 = _score_from_slope(slope if slope is not None else 0.0)

            # momentum normalized 0-100 (heuristic: scale by ATR)
            # momentum can be large for XAUUSD; normalize by atr
            mom_scaled = momentum / max(atr, 1e-12)
            mom_norm_signed = math.tanh(mom_scaled / 2.0)  # [-1,1]
            expected_momentum_score = (mom_norm_signed + 1.0) * 50.0

            expected_volatility_score = _normalize_volatility(float(vol_std)) * 100.0

            # expected_market_regime based on ATR: higher ATR -> TRENDING else RANGING
            # Use per-symbol normalization with simple threshold.
            # Without dataset quantiles, set threshold on ATR ratio to entry.
            atr_ratio = atr / max(entry, 1e-12)
            expected_market_regime = "TRENDING" if atr_ratio >= 0.006 else "RANGING"

            # MACD: keep raw macd as-is; ml_dataset_builder may encode/clamp.
            expected_macd = float(macd)

            expected_rsi = float(rsi)
            expected_atr = float(atr)

            # session expected_* contract in ml_dataset_builder encoding expects categorical strings.
            # We'll keep lowercase for mapping keys (asia/london/new_york in builder uses lower + keys).
            if expected_session == "newyork":
                expected_session = "new_york"

            expected_ai_score = 50.0
            expected_ai_confidence = 0.5
            expected_final_score = 50.0
            expected_sentiment_score = 50.0
            expected_trend_score = float(trend_score_0_100)
            expected_news_impact_score = 50.0

            expected_entry = float(entry)

            # actual_pnl:
            if label == 1:
                actual_pnl = float(tp_dist)
            else:
                actual_pnl = -float(sl_dist)

            order_id = f"HIST_{symbol}_{int(candles[i]['time'])}"

            upsert_execution_expected(
                order_id=order_id,
                symbol=symbol,
                direction=direction,
                expected_entry=expected_entry,
                expected_final_score=float(expected_final_score),
                expected_ai_score=float(expected_ai_score),
                expected_ai_confidence=float(expected_ai_confidence),
                expected_trend_score=float(expected_trend_score),
                expected_momentum_score=float(expected_momentum_score),
                expected_sentiment_score=float(expected_sentiment_score),
                expected_volatility_score=float(expected_volatility_score),

                expected_rsi=float(expected_rsi),
                expected_macd=float(expected_macd),
                expected_session=str(expected_session),
                expected_spread=0.0,
                expected_atr=float(expected_atr),
                expected_trend_strength=float(slope if slope is not None else 0.0),
                expected_market_regime=str(expected_market_regime),
                expected_news_impact_score=float(expected_news_impact_score),

                expected_indicators_json=None,
                strategy="HISTORICAL",
            )

            # Also fill actual_* so trainer can learn from historical rows.
            # This uses upsert_execution_actual-like fields, but we only have upsert_execution_expected.
            # execution_dataset schema includes actual_*; we must call upsert_execution_actual.
            from data.storage.database import upsert_execution_actual

            # actual_exit_reason is not a column in DB; we store it via actual_indicators_json.
            actual_indicators_json = {"exit_reason": reason}

            # Choose an actual_exit price consistent with direction.
            actual_exit = expected_entry + float(tp_dist) if label == 1 and direction == "BUY" else (
                expected_entry - float(tp_dist) if label == 1 and direction == "SELL" else (
                expected_entry - float(sl_dist) if label == 0 and direction == "BUY" else expected_entry + float(sl_dist)
            ))

            upsert_execution_actual(
                order_id=order_id,
                actual_entry=float(expected_entry),
                actual_exit=float(actual_exit),
                actual_pnl=float(actual_pnl),
                spread_at_entry=0.0,
                slippage=0.0,
                execution_delay_ms=None,
                execution_quality_score=50.0,
                price_gap=0.0,
                actual_indicators_json=str(actual_indicators_json),
            )

            saved += 1

        logger.info(f"{symbol}: historical dataset rows saved={saved}")


if __name__ == "__main__":
    build_historical_dataset()
    print("✅ Historical dataset built!")

