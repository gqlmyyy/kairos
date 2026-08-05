# Trading Bot V3 - execution/mt5_direct.py
# Direct order execution using MetaTrader5 official library (no QuantDinger for trading)

import time
from typing import Optional

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    mt5 = None

from utils.logger import get_logger
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

from data.storage.database import close_trade_db_by_order_id, upsert_execution_actual

logger = get_logger("mt5_direct")


def _ensure_mt5_initialized() -> bool:
    if mt5 is None:
        logger.error("FATAL: MetaTrader5 library is not available. Install MetaTrader5 package.")
        return False

    try:
        # Some environments already initialized; calling again is safe.
        if not mt5.terminal_info():
            if not mt5.initialize():
                logger.error("FATAL: mt5.initialize() failed")
                return False
    except Exception as e:
        logger.error(f"FATAL: mt5 initialize check failed: {e}")
        return False

    # Login
    try:
        if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            logger.error(f"FATAL: mt5.login() failed: {mt5.last_error()}")
            return False
    except Exception as e:
        logger.error(f"FATAL: mt5.login() exception: {e}")
        return False

    return True


def _ensure_symbol_selected(symbol: str) -> bool:
    try:
        if not mt5.symbol_select(symbol, True):
            logger.error(f"FATAL: mt5.symbol_select({symbol}) failed")
            return False
    except Exception as e:
        logger.error(f"FATAL: mt5.symbol_select exception for {symbol}: {e}")
        return False
    return True


def _get_positions_qd_like_dicts() -> list:
    """Return MT5 positions as QuantDinger-like dicts (defensive)."""
    if mt5 is None:
        return []
    if not _ensure_mt5_initialized():
        return []

    try:
        positions = mt5.positions_get()
        if not positions:
            return []
        out = []
        for p in positions:
            d = p._asdict() if hasattr(p, "_asdict") else dict(p.__dict__)
            symbol = d.get("symbol") or ""
            ticket = d.get("ticket") or d.get("position_id") or d.get("id") or ""
            ptype = d.get("type")
            direction = "buy"
            try:
                if ptype in (1, "1"):
                    direction = "sell"
                elif str(ptype).lower() in ["sell", "short"]:
                    direction = "sell"
                else:
                    direction = "buy"
            except Exception:
                direction = "buy"

            out.append(
                {
                    "id": str(ticket) if ticket is not None else "",
                    "ticket": str(ticket) if ticket is not None else "",
                    "symbol": symbol,
                    "type": direction,
                    "direction": direction,
                    "volume": float(d.get("volume") or d.get("Volume") or 0),
                    "price_open": float(d.get("price_open") or d.get("price") or 0),
                    "sl": float(d.get("sl") or 0),
                    "tp": d.get("tp", None),
                    "profit": float(d.get("profit") or d.get("pnl") or 0),
                    "comment": d.get("comment") or "",
                    "price_current": float(d.get("price_current") or d.get("price") or 0),
                    "atr": d.get("atr", None),
                    "pip": d.get("pip", None),
                    # MT5 raw dict: use "time" as primary (not "time_open")
                    "time_open": d.get("time") or d.get("time_open") or None,
                }
            )
        return out
    except Exception as e:
        logger.error(f"[MT5_DIRECT] get positions failed: {e}")
        return []


def get_open_positions_mt5() -> list:
    return _get_positions_qd_like_dicts()


def close_trade_mt5(trade_id) -> bool:
    """Close MT5 position by ticket/order_id (defensive)."""
    if mt5 is None:
        logger.error("[MT5_DIRECT] close_trade_mt5: mt5 library not available")
        return False
    if not _ensure_mt5_initialized():
        return False

    try:
        ticket_str = str(trade_id)
        if not ticket_str or not ticket_str.isdigit():
            logger.error(f"[MT5_DIRECT] close_trade_mt5 invalid ticket={trade_id}")
            return False
        ticket_int = int(ticket_str)

        pos_list = mt5.positions_get(ticket=ticket_int)
        if not pos_list:
            return False

        p = pos_list[0]
        d = p._asdict() if hasattr(p, "_asdict") else p.__dict__

        symbol = d.get("symbol")
        volume = d.get("volume")
        ptype = d.get("type")

        if not symbol or volume is None:
            return False

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        # MT5: if position type is BUY(0) => close with SELL; if SELL(1) => close with BUY
        is_buy = (int(ptype) == 0) if isinstance(ptype, (int, float, str)) else str(ptype).lower() == "0"
        order_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        price = float(tick.bid if is_buy else tick.ask)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "position": ticket_int,
            "price": price,
            "deviation": 20,
            "magic": 0,
            "comment": "Bot V3",
        }

        logger.info(f"[MT5_DIRECT] close_trade_mt5 ticket={ticket_int} request={request}")
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"[MT5_DIRECT] close_trade_mt5 order_send returned None last_error={mt5.last_error()}")
            return False
        if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
            logger.error(f"[MT5_DIRECT] close_trade_mt5 failed retcode={getattr(result,'retcode',None)} result={result}")
            return False
        return True
    except Exception as e:
        logger.error(f"[MT5_DIRECT] close_trade_mt5 exception: {e}")
        return False


def close_all_trades_mt5() -> int:
    """Close all open MT5 positions. Returns number of successful closes (best-effort)."""
    if mt5 is None:
        return 0
    if not _ensure_mt5_initialized():
        return 0

    try:
        positions = mt5.positions_get()
        if not positions:
            return 0

        closed = 0
        for p in positions:
            d = p._asdict() if hasattr(p, "_asdict") else p.__dict__
            ticket = d.get("ticket")
            if ticket is None:
                continue

            # Capture real pnl/exit price BEFORE closing (data disappears after close)
            order_id = str(ticket)
            actual_pnl = float(d.get("profit") or 0)
            actual_exit = float(d.get("price_current") or 0)
            actual_entry = float(d.get("price_open") or 0)

            if close_trade_mt5(ticket):
                closed += 1

                try:
                    close_trade_db_by_order_id(order_id, pnl=actual_pnl)
                except Exception as e:
                    logger.error(
                        f"[MT5_DIRECT] close_trade_db_by_order_id failed order_id={order_id}: {e}"
                    )

                try:
                    upsert_execution_actual(
                        order_id=order_id,
                        actual_entry=actual_entry,
                        actual_exit=actual_exit,
                        actual_pnl=actual_pnl,
                        exit_reason="Manual emergency close via Telegram",
                        spread_at_entry=None,
                        slippage=None,
                        execution_delay_ms=None,
                        execution_quality_score=None,
                        price_gap=None,
                        actual_indicators_json=None,
                        exit_probability=None,
                    )
                except Exception as e:
                    logger.error(f"[MT5_DIRECT] upsert_execution_actual failed order_id={order_id}: {e}")

        return closed
    except Exception as e:
        logger.error(f"[MT5_DIRECT] close_all_trades_mt5 exception: {e}")
        return 0


def check_mt5_status_mt5() -> bool:
    """Best-effort MT5 connectivity check (local terminal)."""
    if mt5 is None:
        return False
    try:
        if not _ensure_mt5_initialized():
            return False
        info = mt5.terminal_info()
        return info is not None
    except Exception:
        return False


def _get_supported_filling_modes(symbol: str) -> int:
    """Detect supported filling modes for a symbol using bitwise check.
    
    Returns a bitmask of supported filling modes from symbol_info.filling_mode.
    Falls back to 0 if unavailable.
    
    MT5 filling mode flags:
        SYMBOL_FILLING_FOK     = 1  (Fill or Kill)
        SYMBOL_FILLING_IOC     = 2  (Immediate or Cancel)
        SYMBOL_FILLING_RETURN  = 4  (Return remaining volume)
    """
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"[FILLING_MODE] symbol_info returned None for {symbol}")
            return 0
        filling_mode = getattr(symbol_info, "filling_mode", 0)
        logger.info(
            f"[FILLING_MODE] {symbol}: raw filling_mode={filling_mode} "
            f"(FOK={bool(filling_mode & 1)}, IOC={bool(filling_mode & 2)}, RETURN={bool(filling_mode & 4)})"
        )
        return int(filling_mode)
    except Exception as e:
        logger.error(f"[FILLING_MODE] Failed to get filling_mode for {symbol}: {e}")
        return 0


def _validate_and_adjust_sl_tp(symbol: str, live_price: float, sl: float, tp: float, direction: str) -> tuple:
    """Validate and adjust SL/TP to meet broker's minimum stops_level requirement.
    
    Args:
        symbol: Trading symbol
        live_price: Current market price (ask for BUY, bid for SELL)
        sl: Stop loss price (None if not set)
        tp: Take profit price (None if not set)
        direction: "BUY" or "SELL"
    
    Returns:
        (adjusted_sl, adjusted_tp, was_adjusted, message)
    """
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"[STOPS_LEVEL] symbol_info returned None for {symbol}, skipping validation")
            return sl, tp, False, "symbol_info unavailable"
        
        point = getattr(symbol_info, "point", None)
        stops_level = getattr(symbol_info, "trade_stops_level", None)
        
        if point is None or point <= 0:
            logger.warning(f"[STOPS_LEVEL] Invalid point={point} for {symbol}, skipping validation")
            return sl, tp, False, "invalid point"
        
        if stops_level is None:
            logger.info(f"[STOPS_LEVEL] trade_stops_level is None for {symbol}, no minimum enforced")
            return sl, tp, False, "no stops_level set"
        
        stops_level = int(stops_level)
        min_distance = stops_level * float(point)
        
        logger.info(
            f"[STOPS_LEVEL] {symbol}: point={point}, stops_level={stops_level} points, "
            f"min_distance={min_distance:.5f} ({(min_distance/float(point)):.1f} points)"
        )
        
        adjusted_sl = sl
        adjusted_tp = tp
        was_adjusted = False
        adjustments = []
        
        # Validate SL
        if sl is not None and sl > 0:
            sl_distance = abs(live_price - sl)
            if sl_distance < min_distance:
                # Adjust SL to meet minimum distance
                if direction == "BUY":
                    adjusted_sl = live_price - min_distance
                else:  # SELL
                    adjusted_sl = live_price + min_distance
                was_adjusted = True
                adjustments.append(f"SL adjusted from {sl:.5f} to {adjusted_sl:.5f} (distance: {sl_distance:.5f} -> {min_distance:.5f})")
        
        # Validate TP
        if tp is not None and tp > 0:
            tp_distance = abs(tp - live_price)
            if tp_distance < min_distance:
                # Adjust TP to meet minimum distance
                if direction == "BUY":
                    adjusted_tp = live_price + min_distance
                else:  # SELL
                    adjusted_tp = live_price - min_distance
                was_adjusted = True
                adjustments.append(f"TP adjusted from {tp:.5f} to {adjusted_tp:.5f} (distance: {tp_distance:.5f} -> {min_distance:.5f})")
        
        if was_adjusted:
            message = "; ".join(adjustments)
            logger.warning(f"[STOPS_LEVEL] {symbol}: {message}")
        else:
            message = "no adjustment needed"
        
        return adjusted_sl, adjusted_tp, was_adjusted, message
        
    except Exception as e:
        logger.error(f"[STOPS_LEVEL] Validation error for {symbol}: {e}")
        return sl, tp, False, f"error: {e}"


def _validate_sl_tp_order(live_price: float, sl: float, tp: float, direction: str) -> bool:
    """Validate that SL/TP are on the correct side of live_price.
    
    Args:
        live_price: Current MT5 price (ask for BUY, bid for SELL)
        sl: Stop loss price
        tp: Take profit price
        direction: "BUY" or "SELL"
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    if direction == "BUY":
        if sl >= live_price:
            raise ValueError(
                f"[SAFETY] BUY order invalid: SL ({sl}) must be < live_price ({live_price}). "
                f"This indicates a price discrepancy between data source and MT5."
            )
        if tp <= live_price:
            raise ValueError(
                f"[SAFETY] BUY order invalid: TP ({tp}) must be > live_price ({live_price}). "
                f"This indicates a price discrepancy between data source and MT5."
            )
    else:  # SELL
        if sl <= live_price:
            raise ValueError(
                f"[SAFETY] SELL order invalid: SL ({sl}) must be > live_price ({live_price}). "
                f"This indicates a price discrepancy between data source and MT5."
            )
        if tp >= live_price:
            raise ValueError(
                f"[SAFETY] SELL order invalid: TP ({tp}) must be < live_price ({live_price}). "
                f"This indicates a price discrepancy between data source and MT5."
            )
    
    return True


def open_trade(symbol, direction, size, sl_distance, tp_distance, reason) -> dict:
    """Open a market deal directly through MT5.

    Args:
        symbol: Trading symbol (e.g. "EURUSD")
        direction: "BUY" or "SELL"
        size: Position size in lots
        sl_distance: Stop loss distance in price units (NOT absolute price)
        tp_distance: Take profit distance in price units (NOT absolute price)
        reason: Trade reason/comment (max 31 chars, ASCII only)
    
    Returns:
        dict with status, order_id, price, etc.
    
    Improvements:
    - Accepts SL/TP as distances (not absolute prices) to eliminate price discrepancy
    - Calculates final SL/TP from live MT5 price at execution time
    - Detects supported filling modes using bitwise check
    - Validates and adjusts SL/TP to meet broker's stops_level requirement
    - Retries with different filling modes (FOK -> IOC -> RETURN) if needed
    - Safety check ensures SL/TP are on correct side of price
    - Comprehensive logging for debugging
    """

    logger.info(
        f"[MT5_DIRECT] open_trade symbol={symbol} direction={direction} volume={size} "
        f"sl_distance={sl_distance} tp_distance={tp_distance}"
    )

    if mt5 is None:
        return {
            "status": "error",
            "error": "MetaTrader5 library is not available",
            "order_id": None,
            "price": None,
            "symbol": symbol,
            "direction": direction,
            "volume": float(size) if size is not None else None,
            "timestamp": time.time(),
            "raw_response": {},
        }

    # 1) Ensure initialize success with credentials
    try:
        initialized = mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    except Exception as e:
        err = str(e)
        logger.error(f"[MT5_DIRECT] mt5.initialize exception: {err}")
        return {
            "status": "error",
            "error": err,
            "order_id": None,
            "price": None,
            "symbol": symbol,
            "direction": direction,
            "volume": float(size) if size is not None else None,
            "timestamp": time.time(),
            "raw_response": {},
        }

    if not initialized:
        last_err = mt5.last_error()
        logger.error(f"[MT5_DIRECT] mt5.initialize failed last_error={last_err}")
        return {
            "status": "error",
            "error": f"mt5.initialize failed: {last_err}",
            "order_id": None,
            "price": None,
            "symbol": symbol,
            "direction": direction,
            "volume": float(size) if size is not None else None,
            "timestamp": time.time(),
            "raw_response": {},
        }

    # 2) Diagnostics: terminal_info + account_info
    try:
        logger.info(f"[MT5_DIRECT] terminal_info: {mt5.terminal_info()}")
    except Exception as e:
        logger.error(f"[MT5_DIRECT] terminal_info exception: {e}")

    try:
        logger.info(f"[MT5_DIRECT] account_info: {mt5.account_info()}")
    except Exception as e:
        logger.error(f"[MT5_DIRECT] account_info exception: {e}")

    # select symbol
    if not _ensure_symbol_selected(symbol):
        return {
            "status": "error",
            "error": f"Cannot select symbol {symbol} in MT5",
            "order_id": None,
            "price": None,
            "symbol": symbol,
            "direction": direction,
            "volume": float(size) if size is not None else None,
            "timestamp": time.time(),
            "raw_response": {},
        }

    if direction == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
    elif direction == "SELL":
        order_type = mt5.ORDER_TYPE_SELL
    else:
        return {
            "status": "error",
            "error": f"Invalid direction: {direction}",
            "order_id": None,
            "price": None,
            "symbol": symbol,
            "direction": direction,
            "volume": float(size) if size is not None else None,
            "timestamp": time.time(),
            "raw_response": {},
        }

    # 3) Live price
    try:
        tick = mt5.symbol_info_tick(symbol)
    except Exception as e:
        logger.error(f"[MT5_DIRECT] symbol_info_tick exception for {symbol}: {e}")
        return {
            "status": "error",
            "error": f"symbol_info_tick exception: {e}",
            "order_id": None,
            "price": None,
            "symbol": symbol,
            "direction": direction,
            "volume": float(size) if size is not None else None,
            "timestamp": time.time(),
            "raw_response": {},
        }

    if tick is None:
        return {
            "status": "error",
            "error": f"symbol_info_tick is None for {symbol}",
            "order_id": None,
            "price": None,
            "symbol": symbol,
            "direction": direction,
            "volume": float(size) if size is not None else None,
            "timestamp": time.time(),
            "raw_response": {},
        }

    live_price = tick.ask if direction == "BUY" else tick.bid
    if live_price is None:
        return {
            "status": "error",
            "error": "Live price is None (tick has no ask/bid)",
            "order_id": None,
            "price": None,
            "symbol": symbol,
            "direction": direction,
            "volume": float(size) if size is not None else None,
            "timestamp": time.time(),
            "raw_response": {},
        }

    # ============================================================
    # NEW: Calculate SL/TP from distances using live MT5 price
    # This eliminates price discrepancy between QuantDinger and MT5
    # ============================================================
    try:
        sl_distance_float = float(sl_distance) if sl_distance is not None else 0.0
        tp_distance_float = float(tp_distance) if tp_distance is not None else 0.0
        
        if direction == "BUY":
            sl = live_price - sl_distance_float
            tp = live_price + tp_distance_float
        else:  # SELL
            sl = live_price + sl_distance_float
            tp = live_price - tp_distance_float
        
        logger.info(
            f"[MT5_DIRECT] Calculated SL/TP from distances: live_price={live_price:.5f} "
            f"sl_distance={sl_distance_float:.5f} tp_distance={tp_distance_float:.5f} "
            f"sl={sl:.5f} tp={tp:.5f}"
        )
    except Exception as e:
        logger.error(f"[MT5_DIRECT] Failed to calculate SL/TP from distances: {e}")
        return {
            "status": "error",
            "error": f"SL/TP calculation failed: {e}",
            "order_id": None,
            "price": None,
            "symbol": symbol,
            "direction": direction,
            "volume": float(size) if size is not None else None,
            "timestamp": time.time(),
            "raw_response": {},
        }

    # ============================================================
    # NEW: Safety check - validate SL/TP are on correct side of price
    # ============================================================
    try:
        _validate_sl_tp_order(live_price, sl, tp, direction)
        logger.info(f"[MT5_DIRECT] Safety check passed: SL/TP correctly positioned relative to live_price")
    except ValueError as e:
        logger.error(f"[MT5_DIRECT] Safety check FAILED: {e}")
        return {
            "status": "error",
            "error": str(e),
            "order_id": None,
            "price": None,
            "symbol": symbol,
            "direction": direction,
            "volume": float(size) if size is not None else None,
            "timestamp": time.time(),
            "raw_response": {},
        }

    # ============================================================
    # Validate and adjust SL/TP to meet broker's stops_level
    # ============================================================
    sl, tp, sl_tp_adjusted, sl_tp_msg = _validate_and_adjust_sl_tp(symbol, live_price, sl, tp, direction)
    if sl_tp_adjusted:
        logger.warning(f"[MT5_DIRECT] SL/TP adjusted for {symbol}: {sl_tp_msg}")

    # ============================================================
    # NEW: Detect supported filling modes using bitwise check
    # ============================================================
    supported_filling_modes = _get_supported_filling_modes(symbol)
    
    # Define filling mode candidates in order of preference
    # FOK (1) -> IOC (2) -> RETURN (4)
    filling_candidates = [
        (mt5.ORDER_FILLING_FOK, "FOK"),
        (mt5.ORDER_FILLING_IOC, "IOC"),
        (mt5.ORDER_FILLING_RETURN, "RETURN"),
    ]
    
    # Filter candidates based on what the broker supports
    # If supported_filling_modes is 0, we don't know what's supported, so try all
    available_filling_modes = []
    if supported_filling_modes == 0:
        # Unknown - try all candidates
        available_filling_modes = filling_candidates
        logger.info(f"[FILLING_MODE] {symbol}: broker filling modes unknown, will try all candidates")
    else:
        # Filter to only supported modes
        for mode_value, mode_name in filling_candidates:
            mode_flag = 0
            if mode_name == "FOK":
                mode_flag = 1
            elif mode_name == "IOC":
                mode_flag = 2
            elif mode_name == "RETURN":
                mode_flag = 4
            
            if supported_filling_modes & mode_flag:
                available_filling_modes.append((mode_value, mode_name))
        
        logger.info(
            f"[FILLING_MODE] {symbol}: supported modes bitmask={supported_filling_modes}, "
            f"available candidates={[name for _, name in available_filling_modes]}"
        )
    
    # If no modes are available, fall back to trying all
    if not available_filling_modes:
        available_filling_modes = filling_candidates
        logger.warning(f"[FILLING_MODE] {symbol}: no supported modes detected, falling back to all candidates")

    # ============================================================
    # 4) Build request and try each filling mode
    # ============================================================
    # NOTE: 'reason' can contain non-latin (Arabic) text and is rejected by MT5.
    # Always send a safe short ASCII comment.
    
    # Ensure required numeric conversions to avoid surprises
    size_float = float(size) if size is not None else 0.0
    price_float = float(live_price) if live_price is not None else 0.0
    
    last_result = None
    last_error = None
    
    for filling_value, filling_name in available_filling_modes:
        logger.info(f"[MT5_DIRECT] Attempting order with filling_mode={filling_name} ({filling_value})")
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": size_float,
            "type": order_type,
            "price": price_float,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 0,
            # MT5 validates comment strictly:
            # - ASCII/latin only
            # - max length 31
            # Any long Arabic text (from 'reason') may cause: (-2, 'Invalid "comment" argument').
            "comment": "Bot V3",
            "type_filling": filling_value,
        }

        logger.debug(f"[MT5_DIRECT] order_send request={request}")

        # Send request
        result = None
        try:
            result = mt5.order_send(request)
        except Exception as e:
            logger.error(f"[MT5_DIRECT] order_send exception with {filling_name}: {e}")
            last_error = str(e)
            continue

        # Handle result None
        if result is None:
            error_info = mt5.last_error()
            logger.error(f"[MT5_DIRECT] order_send returned None with {filling_name} last_error={error_info}")
            last_error = f"order_send returned None: last_error={error_info}"
            continue

        raw = {}
        try:
            raw = result._asdict()
        except Exception:
            raw = {"repr": repr(result)}

        # Check if successful
        if getattr(result, "retcode", None) == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"[MT5_DIRECT] Order SUCCESS with {filling_name}: order={getattr(result, 'order', None)} "
                f"price={getattr(result, 'price', None)} volume={getattr(result, 'volume', None)}"
            )
            return {
                "status": "success",
                "order_id": str(getattr(result, "order", None)),
                "price": getattr(result, "price", None),
                "symbol": symbol,
                "direction": direction,
                "volume": size,
                "timestamp": time.time(),
                "raw_response": raw,
            }

        # Failed with this filling mode - log and try next
        retcode = getattr(result, "retcode", None)
        comment = getattr(result, "comment", None)
        logger.warning(
            f"[MT5_DIRECT] order_send FAILED with {filling_name}: retcode={retcode} "
            f"comment={comment} result={result}"
        )
        
        last_result = result
        last_error = f"order_send failed with {filling_name}. comment={comment} retcode={retcode}"
        
        # If error is not filling-related, don't try other modes
        # retcode 10030 = Unsupported filling mode - try next
        # retcode 10016 = Invalid stops - no point trying other filling modes
        if retcode == 10016:
            logger.error(f"[MT5_DIRECT] Invalid stops error (10016), not retrying with other filling modes")
            break

    # All filling modes failed
    logger.error(f"[MT5_DIRECT] All filling modes exhausted for {symbol}. Last error: {last_error}")
    
    raw_last = {}
    if last_result:
        try:
            raw_last = last_result._asdict()
        except Exception:
            raw_last = {"repr": repr(last_result)}
    
    return {
        "status": "error",
        "error": last_error or "All filling modes failed",
        "order_id": None,
        "price": None,
        "symbol": symbol,
        "direction": direction,
        "volume": size,
        "timestamp": time.time(),
        "raw_response": raw_last,
    }


