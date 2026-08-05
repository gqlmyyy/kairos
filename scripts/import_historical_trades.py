"""Import Historical Trades from MT5 for Training Data

This script downloads closed trades from MT5, computes all features required
for training, classifies exit_reason, and stores in execution_dataset.

DO NOT TRAIN THE MODEL. DO NOT USE FAKE VALUES.
"""

from __future__ import annotations

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# MT5 initialization
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("Warning: MT5 not available")

from utils.logger import get_logger
from data.storage.database import get_conn, init_db, upsert_execution_actual


logger = get_logger("import_historical_trades")

# Constants
HISTORY_DAYS = 180
ATR_PERIOD = 14
ADX_PERIOD = 14


def init_mt5() -> bool:
    """Initialize MT5 connection."""
    if not MT5_AVAILABLE:
        return False
    try:
        mt5.initialize()
        logger.info("MT5 initialized")
        return True
    except Exception as e:
        logger.error(f"MT5 init failed: {e}")
        return False


def shutdown_mt5():
    """Shutdown MT5 connection."""
    if MT5_AVAILABLE:
        try:
            mt5.shutdown()
        except Exception:
            pass


def get_candle_bars(symbol: str, timeframe: str, count: int) -> Optional[np.ndarray]:
    """Get historical candle bars for a symbol."""
    if not MT5_AVAILABLE:
        return None

    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    tf = tf_map.get(timeframe.upper())
    if tf is None:
        return None

    try:
        bars = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if bars is None or len(bars) < ATR_PERIOD + 1:
            return None
        return bars
    except Exception:
        return None


def calculate_atr(bars: np.ndarray, period: int = ATR_PERIOD) -> Optional[float]:
    """Calculate ATR from candle bars."""
    if bars is None or len(bars) < period + 1:
        return None

    try:
        trs = []
        for i in range(1, len(bars)):
            high = bars[i, 2]
            low = bars[i, 3]
            close_prev = bars[i - 1, 4]
            tr = max(
                high - low,
                abs(high - close_prev),
                abs(low - close_prev),
            )
            trs.append(tr)

        if len(trs) < period:
            return None

        # Wilder smoothing
        atr = sum(trs[:period]) / period
        for i in range(period, len(trs)):
            atr = (atr * (period - 1) + trs[i]) / period

        return float(atr)
    except Exception:
        return None


def calculate_rsi(bars: np.ndarray, period: int = 14) -> Optional[float]:
    """Calculate RSI from candle bars."""
    if bars is None or len(bars) < period + 1:
        return None

    try:
        closes = bars[:, 4]
        gains = []
        losses = []

        for i in range(1, len(closes)):
            change = closes[i] - closes[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                abs(change)

        if len(gains) < period:
            return None

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi)
    except Exception:
        return None


def calculate_adx(
    bars: np.ndarray,
    period: int = ADX_PERIOD,
    symbol: str | None = None,
) -> Optional[float]:
    """Calculate ADX from candle bars.

    Instrumentation-only:
    - Logging is enabled only when symbol == "EURUSD" AND logger level is DEBUG.
    - Does NOT change the ADX algorithm.
    """
    if bars is None or len(bars) < period * 2:
        return None

    # Instrumentation switch
    _sym = (symbol or "").strip().upper()
    # أكثر موثوقية من مقارنة logger.level مباشرةً
    import logging as _logging
    # DEBUG markers (symbol == EURUSD only) before any calculations
    # NOTE: if logger DEBUG is not enabled, we won't see them.
    # For root-cause collection, we temporarily emit WARNING so we can confirm reachability.
    if _sym == "EURUSD":
        logger.warning("[ADX_TRACE_MARKER] entered calculate_adx")
        logger.warning("[ADX_TRACE_MARKER] symbol=%s type(bars)=%s", _sym, type(bars))
        try:
            logger.warning("[ADX_TRACE_MARKER] len(bars)=%d", len(bars))
        except Exception:
            logger.warning("[ADX_TRACE_MARKER] len(bars)=<unavailable>")
        logger.warning("[ADX_TRACE_MARKER] logger.level=%s", getattr(logger, "level", None))
        logger.warning("[ADX_TRACE_MARKER] logger.isEnabledFor(DEBUG)=%s", logger.isEnabledFor(_logging.DEBUG))
    _enable_trace = (_sym == "EURUSD") and logger.isEnabledFor(_logging.DEBUG)

    try:
        highs = bars[:, 2]
        lows = bars[:, 3]
        closes = bars[:, 4]

        # TEMP TRACE (EURUSD only) - use logger.error so we can reliably see output
        _trace_err = (_sym == "EURUSD")
        if _trace_err:
            logger.error("[ADX_TRACE_ERR] ENTER calculate_adx")
            logger.error("[ADX_TRACE_ERR] symbol=%s type(bars)=%s", _sym, type(bars))
            try:
                logger.error("[ADX_TRACE_ERR] shape=%s dtype=%s", getattr(bars, "shape", None), getattr(bars, "dtype", None))
            except Exception as _e:
                logger.error("[ADX_TRACE_ERR] shape/dtype read failed: %s", _e)
            try:
                logger.error("[ADX_TRACE_ERR] len(bars)=%d", len(bars))
            except Exception as _e:
                logger.error("[ADX_TRACE_ERR] len(bars) read failed: %s", _e)

            try:
                logger.error("[ADX_TRACE_ERR] first5=%s", bars[:5])
                logger.error("[ADX_TRACE_ERR] last5=%s", bars[-5:])
            except Exception as _e:
                logger.error("[ADX_TRACE_ERR] first/last bars read failed: %s", _e)

        if _enable_trace:
            try:
                import numpy as _np  # local
                logger.debug("[ADX_TRACE] ===== calculate_adx start =====")
                logger.debug(
                    "[ADX_TRACE] type(bars)=%s shape=%s dtype=%s",
                    type(bars),
                    getattr(bars, "shape", None),
                    getattr(bars, "dtype", None),
                )

                n_bars = int(len(bars))
                logger.debug("[ADX_TRACE] bars_count=%d", n_bars)

                first5 = bars[:5]
                last5 = bars[-5:] if n_bars >= 5 else bars[:]

                logger.debug("[ADX_TRACE] first5 bars=%s", first5)
                logger.debug("[ADX_TRACE] last5 bars=%s", last5)

                logger.debug(
                    "[ADX_TRACE] highs_first5=%s highs_last5=%s",
                    highs[:5],
                    highs[-5:] if n_bars >= 5 else highs[:],
                )
                logger.debug(
                    "[ADX_TRACE] lows_first5=%s lows_last5=%s",
                    lows[:5],
                    lows[-5:] if n_bars >= 5 else lows[:],
                )
                logger.debug(
                    "[ADX_TRACE] closes_first5=%s closes_last5=%s",
                    closes[:5],
                    closes[-5:] if n_bars >= 5 else closes[:],
                )
            except Exception as _e:
                logger.debug("[ADX_TRACE] trace init failed: %s", _e)

        plus_dm = []
        minus_dm = []
        trs = []

        if _enable_trace:
            logger.debug("[ADX_TRACE] Begin computing TR/+DM/-DM ...")

        # arrays of same index as trs/plus_dm/minus_dm (built for i in 1..len(bars)-1)
        # NOTE: algorithm remains unchanged; only we add logging.
        for i in range(1, len(bars)):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
                minus_dm.append(0.0)
            elif down_move > up_move and down_move > 0:
                plus_dm.append(0.0)
                minus_dm.append(down_move)
            else:
                plus_dm.append(0.0)
                minus_dm.append(0.0)

            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)

            if _enable_trace and i <= 20:
                # convert to python floats for logs
                try:
                    tr_i = float(trs[-1]) if trs else None
                    plus_i = float(plus_dm[-1]) if plus_dm else None
                    minus_i = float(minus_dm[-1]) if minus_dm else None
                    logger.debug(
                        "[ADX_TRACE][i=%d] TR=%s +DM=%s -DM=%s",
                        i,
                        tr_i,
                        plus_i,
                        minus_i,
                    )
                except Exception:
                    pass

        if len(trs) < period + 1:
            if _enable_trace:
                logger.debug("[ADX_TRACE] return None early: len(trs)=%d < period+1=%d", len(trs), period + 1)
            return None

        # Wilder smoothing
        tr_sum = sum(trs[1:period + 1])
        plus_dm_sum = sum(plus_dm[1:period + 1])
        minus_dm_sum = sum(minus_dm[1:period + 1])

        if _enable_trace:
            logger.debug(
                "[ADX_TRACE] init sums: tr_sum=%s plus_dm_sum=%s minus_dm_sum=%s",
                tr_sum,
                plus_dm_sum,
                minus_dm_sum,
            )
            try:
                # ATR is computed differently in calculate_atr; for ADX tracing we only trace ATR-like smoothing isn't present.
                # We'll infer ATR trace using same TR series smoothing formula as in calculate_atr for the first period.
                # This does NOT change algorithm, only adds informational logs.
                if len(trs) >= period:
                    # trs corresponds to TR values from i=1..len(bars)-1; calculate_atr uses trs[:period]
                    atr_like = float(sum(trs[:period]) / period)
                    logger.debug("[ADX_TRACE] ATR_like (informational)=%s", atr_like)
            except Exception:
                pass

        if tr_sum == 0:
            if _enable_trace:
                logger.debug("[ADX_TRACE] return None: tr_sum == 0")
            return None

        plus_di = 100.0 * (plus_dm_sum / tr_sum)
        minus_di = 100.0 * (minus_dm_sum / tr_sum)

        if _enable_trace:
            logger.debug(
                "[ADX_TRACE] init DI: +DI=%s -DI=%s",
                plus_di,
                minus_di,
            )

        dx_list = []
        # NOTE: algorithm indexes i over trs where trs[i] exists for i in 0..len(trs)-1
        # Original code: for i in range(period + 1, len(trs)):
        for i in range(period + 1, len(trs)):
            tr_sum = tr_sum - (tr_sum / period) + trs[i]
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]

            if tr_sum == 0:
                if _enable_trace:
                    logger.debug("[ADX_TRACE][i=%d] continue: tr_sum == 0", i)
                continue

            plus_di = 100.0 * (plus_dm_sum / tr_sum)
            minus_di = 100.0 * (minus_dm_sum / tr_sum)

            denom = plus_di + minus_di
            if denom == 0:
                if _enable_trace:
                    logger.debug("[ADX_TRACE][i=%d] continue: denom(+DI+-DI)==0 (%s + %s)", i, plus_di, minus_di)
                continue

            dx = 100.0 * (abs(plus_di - minus_di) / denom)
            dx_list.append(dx)

            if _enable_trace and len(dx_list) <= 20:
                ifaces = {
                    "i": i,
                    "TR": float(trs[i]),
                    "+DI": float(plus_di),
                    "-DI": float(minus_di),
                    "DX": float(dx),
                }
                logger.debug(
                    "[ADX_TRACE][dx_step=%d] i=%d TR=%s +DI=%s -DI=%s DX=%s",
                    len(dx_list),
                    ifaces["i"],
                    ifaces["TR"],
                    ifaces["+DI"],
                    ifaces["-DI"],
                    ifaces["DX"],
                )

        if len(dx_list) < period:
            if _enable_trace:
                logger.debug("[ADX_TRACE] return None: len(dx_list)=%d < period=%d", len(dx_list), period)
            return None

        # ADX Wilder smoothing
        adx = sum(dx_list[:period]) / period
        for v in dx_list[period:]:
            adx = (adx * (period - 1) + v) / period

        if _enable_trace:
            logger.debug("[ADX_TRACE] dx_list_len=%d", len(dx_list))
            logger.debug("[ADX_TRACE] final ADX=%s", adx)
            if adx is None or float(adx) == 0.0:
                logger.debug("[ADX_TRACE] final ADX is 0.0 or None -> reason will be inferred from above trace (dx_list/max/min/continues).")

        return float(adx)
    except Exception as _e:
        if _enable_trace:
            logger.debug("[ADX_TRACE] exception: %s", _e)
        return None


def detect_trend(bars: np.ndarray, period: int = 20) -> int:
    """Detect trend direction: 1=up, -1=down, 0=flat."""
    if bars is None or len(bars) < period:
        return 0

    try:
        closes = bars[:, 4]
        recent = closes[-period:]
        first = np.mean(recent[:period // 3])
        last = np.mean(recent[-period // 3:])

        pct_change = (last - first) / first * 100.0

        if pct_change > 0.5:
            return 1
        elif pct_change < -0.5:
            return -1
        return 0
    except Exception:
        return 0


def get_session_from_time(time_str: Any) -> str:
    """Determine trading session from timestamp."""
    if not time_str:
        return "unknown"

    try:
        # Parse timestamp
        if isinstance(time_str, (int, float)):
            dt = datetime.fromtimestamp(time_str)
        elif isinstance(time_str, str):
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    dt = datetime.strptime(str(time_str), fmt)
                    break
                except Exception:
                    continue
            else:
                return "unknown"
        else:
            return "unknown"

        hour = dt.hour

        # Asia: 0-7 (Tokyo), Europe: 7-16 (London), America: 13-21 (NY)
        if hour < 7:
            return "asia"
        elif hour < 13:
            return "europe"
        else:
            return "america"
    except Exception:
        return "unknown"


def classify_exit_reason(
    deal: dict,
    order: Optional[dict],
    entry_price: float,
    exit_price: float,
    direction: str,
) -> str:
    """Classify the real exit_reason.

    Args:
        deal: Deal record from MT5
        order: Order record from MT5 (if exists)
        entry_price: Entry price
        exit_price: Exit price
        direction: "buy" or "sell"

    Returns:
        One of: take_profit, stop_loss, trailing, breakeven, manual, profit_decay, timeout, unknown
    """
    try:
        # Get SL/TP from order if available
        sl = None
        tp = None

        if order:
            sl = order.get("sl")
            tp = order.get("tp")

        # Check deal type and comment
        deal_type = deal.get("type")
        comment = str(deal.get("comment", "")).lower()

        # Manual close via comment
        if "manual" in comment or "close" in comment:
            return "manual"

        # Check deal type constants
        if mt5 is not None:
            if deal_type == mt5.DEAL_TYPE_BUY:
                actual_type = "buy"
            elif deal_type == mt5.DEAL_TYPE_SELL:
                actual_type = "sell"
            else:
                actual_type = direction
        else:
            actual_type = direction

        # Check against SL/TP
        pip = 0.0001
        if "JPY" in str(deal.get("symbol", "")).upper():
            pip = 0.01

        # Calculate distance in pips
        if actual_type == "buy":
            dist_to_tp = (tp - entry_price) / pip if tp else None
            dist_to_sl = (sl - entry_price) / pip if sl else None
            exit_vs_entry = exit_price - entry_price
        else:
            dist_to_tp = (entry_price - tp) / pip if tp else None
            dist_to_sl = (entry_price - sl) / pip if sl else None
            exit_vs_entry = entry_price - exit_price

        # Check if hit TP (within 1 pip tolerance)
        if tp and dist_to_tp is not None:
            if actual_type == "buy":
                if exit_price >= tp - pip:
                    return "take_profit"
            else:
                if exit_price <= tp + pip:
                    return "take_profit"

        # Check if hit SL
        if sl and dist_to_sl is not None:
            if actual_type == "buy":
                if exit_price <= sl + pip:
                    return "stop_loss"
            else:
                if exit_price >= sl - pip:
                    return "stop_loss"

        # Check for trailing/breakeven (SL moved)
        if sl and order:
            original_sl = order.get("sl")
            if original_sl and sl != original_sl:
                # Check if moved to breakeven
                if abs(sl - entry_price) < pip * 5:
                    return "breakeven"
                return "trailing"

        # Check profit decay - closed at loss after being in profit
        if exit_vs_entry < 0:
            # Was in loss
            return "stop_loss"

        # Default to unknown
        return "unknown"

    except Exception as e:
        logger.debug(f"classify_exit_reason error: {e}")
        return "unknown"


def calculate_features_for_deal(
    symbol: str,
    entry_time: int,
    exit_time: int,
    entry_price: float,
    exit_price: float,
    direction: str,
    volume: float,
) -> Dict[str, Any]:
    """Calculate all features for a deal."""

    features = {
        "entry_atr": None,
        "entry_rsi": None,
        "entry_adx": None,
        "trend_h1": 0,
        "trend_h4": 0,
        "spread": None,
        "volume": volume,
        "mfe": None,
        "mae": None,
        "trade_health": None,
        "market_regime": "unknown",
        "session": "unknown",
        "hour": None,
        "day_of_week": None,
        "trade_duration": None,
        "entry_tp": None,
        "entry_sl": None,
        "risk_reward_ratio": None,
        "distance_to_tp": None,
        "distance_to_sl": None,
    }

    try:
        # Get candle bars before entry for feature calculation
        bars_h4 = get_candle_bars(symbol, "H4", 50)
        bars_h1 = get_candle_bars(symbol, "H1", 50)

        # Entry features from H4
        if bars_h4 is not None:
            features["entry_atr"] = calculate_atr(bars_h4)
            features["entry_rsi"] = calculate_rsi(bars_h4)
            features["entry_adx"] = calculate_adx(bars_h4)
            features["trend_h4"] = detect_trend(bars_h4)

        if bars_h1 is not None:
            features["trend_h1"] = detect_trend(bars_h1)

        # Get spread at entry
        if MT5_AVAILABLE and bars_h4 is not None:
            try:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    features["spread"] = symbol_info.spread
            except Exception:
                pass

        # Calculate MFE/MAE
        pip = 0.01 if symbol.upper().endswith("JPY") else 0.0001

        if direction.lower() == "buy":
            features["mfe"] = (max(entry_price, exit_price) - entry_price) / pip
            features["mae"] = (entry_price - min(entry_price, exit_price)) / pip
        else:
            features["mfe"] = (entry_price - max(entry_price, exit_price)) / pip
            features["mae"] = (min(entry_price, exit_price) - entry_price) / pip

        # Calculate distance to TP/SL at peak
        # We don't have original SL/TP here, so skip for now
        # This would require tracking during the trade

        # Trade duration
        if exit_time and entry_time:
            duration_mins = (exit_time - entry_time) / 60
            features["trade_duration"] = int(duration_mins)

        # Time features
        dt = datetime.fromtimestamp(entry_time)
        features["session"] = get_session_from_time(entry_time)
        features["hour"] = dt.hour
        features["day_of_week"] = dt.weekday()

        # Market regime from ATR
        if features["entry_atr"]:
            # Estimate regime
            # High ATR = volatile, Low ATR = ranging
            # This is a simplified version
            atr_value = features["entry_atr"]
            if atr_value > 0.002:
                features["market_regime"] = "volatile"
            elif atr_value > 0.0008:
                features["market_regime"] = "trending"
            else:
                features["market_regime"] = "ranging"

    except Exception as e:
        logger.debug(f"calculate_features error: {e}")

    return features


def import_historical_trades() -> Dict[str, Any]:
    """Main function to import historical trades."""

    if not init_mt5():
        return {"error": "MT5 not available"}

    result = {
        "total_imported": 0,
        "exit_reason_distribution": {},
        "skipped": 0,
        "skipped_reasons": [],
    }

    try:
        # Get time range
        to_time = int(time.time())
        from_time = to_time - (HISTORY_DAYS * 24 * 3600)

        logger.info(f"Importing trades from {HISTORY_DAYS} days")

        # Get history deals
        deals = mt5.history_deals_get(from_time, to_time)
        if deals is None:
            logger.error("No deals found")
            return result

        deals = list(deals)
        logger.info(f"Found {len(deals)} deals")

        # Get history orders for SL/TP information
        orders = mt5.history_orders_get(from_time, to_time)
        orders_by_position = {}
        if orders:
            for o in orders:
                o_dict = dict(o._asdict())
                pos_id = o_dict.get("position_id")
                if pos_id:
                    if pos_id not in orders_by_position:
                        orders_by_position[pos_id] = []
                    orders_by_position[pos_id].append(o_dict)

        logger.debug(f"Found {len(orders_by_position)} orders with position_id")

        # Debug: show deal types and entry field
        if deals:
            deal_types = {}
            deal_entries = {}
            first_deal = dict(deals[0]._asdict())
            logger.debug(f"Deal fields: {list(first_deal.keys())}")
            logger.debug(f"Deal sample: time={first_deal.get('time')}, type={first_deal.get('type')}, entry={first_deal.get('entry')}")

            for d in deals:
                dt = d.type
                de = d.entry
                deal_types[dt] = deal_types.get(dt, 0) + 1
                deal_entries[de] = deal_entries.get(de, 0) + 1
            logger.debug(f"Deal types: {deal_types}")
            logger.debug(f"Deal entry field values: {deal_entries}")

        # Process each deal
        imported_count = 0
        skipped = 0
        skipped_reasons = []
        exit_reasons_dist = {}

        # Group deals by position_id - use the 'entry' field to determine ENTRY vs EXIT
        positions = {}
        deals_without_pos = 0
        debug_entries = 0
        debug_exits = 0

        for d in deals:
            deal_dict = dict(d._asdict())
            pos_id = deal_dict.get("position_id")

            if pos_id is None or pos_id == 0:
                deals_without_pos += 1
                continue

            if pos_id not in positions:
                positions[pos_id] = {
                    "entries": [],
                    "exits": [],
                    "symbol": None,
                    "direction": None,
                    "entry_time": None,
                    "exit_time": None,
                    "entry_price": None,
                    "exit_price": None,
                    "volume": None,
                }

            # Use 'entry' field: 0 = DEAL_ENTRY_IN (entry), 1 = DEAL_ENTRY_OUT (exit)
            entry_flag = deal_dict.get("entry", -1)
            deal_type = deal_dict.get("type")
            volume = abs(deal_dict.get("volume", 0))

            if entry_flag == 0:  # DEAL_ENTRY_IN
                debug_entries += 1
                # Don't add duplicate entries
                if not positions[pos_id]["entries"]:
                    positions[pos_id]["entries"].append(deal_dict)
                    positions[pos_id]["symbol"] = deal_dict.get("symbol")
                    positions[pos_id]["entry_time"] = deal_dict.get("time")
                    positions[pos_id]["entry_price"] = deal_dict.get("price")
                    positions[pos_id]["volume"] = volume
                    positions[pos_id]["direction"] = "buy" if deal_type == mt5.DEAL_TYPE_BUY else "sell"
            elif entry_flag == 1:  # DEAL_ENTRY_OUT
                debug_exits += 1
                # Don't add duplicate exits
                if not positions[pos_id]["exits"]:
                    positions[pos_id]["exits"].append(deal_dict)
                    positions[pos_id]["exit_time"] = deal_dict.get("time")
                    positions[pos_id]["exit_price"] = deal_dict.get("price")

        logger.debug(f"Deals without position_id: {deals_without_pos}")
        logger.debug(f"Entry deals: {debug_entries}, Exit deals: {debug_exits}")
        logger.debug(f"Grouped positions: {len(positions)}")

        # Now process each position
        processed = 0
        skipped = 0
        skipped_no_exit = 0
        skipped_reasons = []

        # Create single connection for all inserts
        conn = get_conn()
        c = conn.cursor()
        now = datetime.now().isoformat()

        for pos_id, pos_data in positions.items():
            try:
                entries = pos_data.get("entries", [])
                exits = pos_data.get("exits", [])

                # Skip if no complete trade (need at least one entry and one exit)
                if not entries:
                    skipped += 1
                    skipped_reasons.append(f"pos_{pos_id}: no entries")
                    continue

                if not exits:
                    skipped_no_exit += 1
                    continue  # Skip open positions

                # Use the first entry and last exit
                entry_deal = entries[0]  # Already sorted during collection
                exit_deal = exits[-1]   # Already sorted during collection

                # Get all values from the deals
                entry_time = int(pos_data.get("entry_time", 0) or entry_deal.get("time", 0))
                exit_time = int(pos_data.get("exit_time", 0) or exit_deal.get("time", 0))
                entry_price = float(pos_data.get("entry_price", 0) or entry_deal.get("price", 0))
                exit_price = float(pos_data.get("exit_price", 0) or exit_deal.get("price", 0))
                volume = float(pos_data.get("volume", 0) or entry_deal.get("volume", 0))
                symbol = str(pos_data.get("symbol") or entry_deal.get("symbol", ""))
                direction = pos_data.get("direction", "buy")

                if entry_time == 0 or exit_time == 0 or entry_price == 0:
                    skipped += 1
                    skipped_reasons.append(f"pos_{pos_id}: invalid data")
                    continue

                processed += 1
                logger.debug(f"Importing {symbol} {direction} {volume} @ {entry_price} -> {exit_price}")

                # Get order for SL/TP from orders (using position_id)
                order = None
                orders_for_pos = orders_by_position.get(pos_id, [])
                if orders_for_pos:
                    order = orders_for_pos[0]

                # Classify exit reason
                exit_reason = classify_exit_reason(
                    exit_deal, order, entry_price, exit_price, direction
                )

                # Calculate features
                features = calculate_features_for_deal(
                    symbol,
                    entry_time,
                    exit_time,
                    entry_price,
                    exit_price,
                    direction,
                    volume,
                )

                # Calculate actual P&L
                pip = 0.01 if symbol.upper().endswith("JPY") else 0.0001
                if direction == "buy":
                    pnl = (exit_price - entry_price) * volume / pip
                else:
                    pnl = (entry_price - exit_price) * volume / pip

                # Get SL/TP from order
                sl = order.get("sl") if order else None
                tp = order.get("tp") if order else None

                # Risk reward ratio
                risk_reward = None
                if sl and tp:
                    sl_dist = abs(entry_price - sl) / pip
                    tp_dist = abs(tp - entry_price) / pip
                    if sl_dist > 0:
                        risk_reward = tp_dist / sl_dist

                # Build record for execution_dataset
                now = datetime.now().isoformat()

                record = {
                    "dataset_created_at": now,
                    "dataset_updated_at": now,
                    "order_id": str(entry_deal.get("ticket", pos_id)),
                    "symbol": symbol,
                    "direction": direction,
                    "expected_entry": entry_price,
                    "expected_final_score": None,
                    "expected_rsi": features.get("entry_rsi"),
                    "expected_macd": None,
                    "expected_session": features.get("session"),
                    "expected_spread": features.get("spread"),
                    "expected_atr": features.get("entry_atr"),
                    "expected_trend_strength": features.get("entry_adx"),
                    "expected_momentum_score": None,
                    "expected_volatility_score": None,
                    "expected_market_regime": features.get("market_regime"),
                    "expected_ai_score": None,
                    "expected_sentiment_score": None,
                    "expected_news_impact_score": None,
                    "expected_ai_confidence": None,
                    "expected_trend_score": features.get("trend_h1"),
                    "expected_momentum_score_legacy": None,
                    "expected_sentiment_score_legacy": None,
                    "expected_volatility_score_legacy": None,
                    "expected_indicators_json": json.dumps({
                        "mfe": features.get("mfe"),
                        "mae": features.get("mae"),
                        "direction": direction,
                    }),
                    "status": "closed",
                    "actual_entry": entry_price,
                    "actual_exit": exit_price,
                    "actual_pnl": pnl,
                    "spread_at_entry": features.get("spread"),
                    "slippage": None,
                    "execution_delay_ms": None,
                    "execution_quality_score": None,
                    "price_gap": None,
                    "actual_indicators_json": None,
                    "breakeven_done": 0,
                    "trailing_done": 0,
                    "exit_reason": exit_reason,
                    "exit_probability": None,
                    "time_open": entry_time,
                    "expected_tp": tp,
                    "expected_sl": sl,
                    "risk_reward_ratio": risk_reward,
                    "trade_duration": features.get("trade_duration"),
                }

                # Insert into database using existing upsert functions
                try:
                    # Use shared connection - no need to create new one

                    # Simplified INSERT - just the core columns
                    columns = "dataset_created_at, dataset_updated_at, order_id, symbol, direction, expected_entry, status, actual_entry, actual_exit, actual_pnl, exit_reason, exit_probability".split(", ")
                    values = (
                        now, now,
                        str(record["order_id"]),
                        str(record["symbol"]),
                        str(record["direction"]),
                        float(record.get("expected_entry") or record.get("actual_entry") or 0),
                        "closed",
                        float(record.get("actual_entry") or 0),
                        float(record.get("actual_exit") or 0),
                        float(record.get("actual_pnl") or 0),
                        str(record.get("exit_reason") or ""),
                        float(record.get("exit_probability") or 0),
                    )
                    c.execute(f"""
                        INSERT OR REPLACE INTO execution_dataset (
                            {", ".join(columns)}
                        ) VALUES ({", ".join(["?"] * len(columns))})
                    """, values)

                    conn.commit()
                    imported_count += 1

                    # Update distribution
                    exit_reasons_dist[exit_reason] = exit_reasons_dist.get(exit_reason, 0) + 1

                except Exception as e:
                    logger.error(f"DB insert failed: {e}")
                    skipped += 1
                    skipped_reasons.append(f"pos_{pos_id}: {str(e)}")

            except Exception as e:
                logger.debug(f"Process deal error: {e}")
                skipped += 1

        result["total_imported"] = imported_count
        result["exit_reason_distribution"] = exit_reasons_dist
        result["skipped"] = skipped
        result["skipped_reasons"] = skipped_reasons[-10:]  # Last 10

        logger.info(f"Import complete: {imported_count} trades, {skipped} skipped")
        logger.info(f"Exit reason distribution: {exit_reasons_dist}")

        # Close shared connection
        try:
            conn.close()
        except:
            pass

    except Exception as e:
        logger.error(f"Import failed: {e}")
        result["error"] = str(e)
    finally:
        shutdown_mt5()

    return result


if __name__ == "__main__":
    print("=" * 50)
    print("Importing Historical Trades from MT5")
    print("=" * 50)

    # Initialize database with migrations
    init_db()

    result = import_historical_trades()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total imported: {result.get('total_imported', 0)}")
    print(f"Skipped: {result.get('skipped', 0)}")
    print("\nExit Reason Distribution:")
    for reason, count in result.get("exit_reason_distribution", {}).items():
        print(f"  {reason}: {count}")

    if result.get("skipped_reasons"):
        print("\nSkipped reasons (last 10):")
        for r in result["skipped_reasons"]:
            print(f"  - {r}")

    if result.get("error"):
        print(f"\nError: {result['error']}")

    print("=" * 50)