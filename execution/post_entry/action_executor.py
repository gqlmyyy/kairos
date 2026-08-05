from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import time
from utils.logger import get_logger


try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    mt5 = None


logger = get_logger("action_executor")


class ActionExecutor:
    """Executes actions on MT5 only. No decision logic, no telegram, no DB."""

    def __init__(self) -> None:
        pass

    def modify_sl(self, order_id: str, symbol: str, direction: str, new_sl: float) -> bool:
        """Modify SL on MT5 only.

        Best-effort: sends TRADE_ACTION_SLTP with tp left as 0.
        This is required so MoveSL is executable end-to-end.
        """
        if mt5 is None:
            return False

        if not symbol or new_sl is None:
            return False

        try:
            ticket_str = str(order_id)
            if not ticket_str.isdigit():
                logger.warning(f"[MT5] modify_sl invalid order_id={order_id}")
                return False
            ticket_int = int(ticket_str)

            positions = mt5.positions_get(ticket=ticket_int)
            if not positions:
                logger.warning(f"[MT5] modify_sl no position ticket={ticket_int}")
                return False
            p = positions[0]
            d = p._asdict() if hasattr(p, "_asdict") else p.__dict__

            # Keep existing TP if present
            current_tp = d.get("tp")
            tp_val = float(current_tp) if current_tp not in (None, 0, 0.0) else 0.0

            # Determine order type for SLTP request
            order_type = mt5.ORDER_TYPE_BUY if str(direction).lower() == "buy" else mt5.ORDER_TYPE_SELL

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"[MT5] modify_sl no tick symbol={symbol}")
                return False
            price = float(tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid or 0.0)

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "sl": float(new_sl),
                "tp": float(tp_val),
                "magic": 0,
                "order": ticket_int,
                "position": ticket_int,
                "type": order_type,
                "price": price,
                "deviation": 20,
            }

            result = mt5.order_send(request)
            if result is None:
                logger.error(f"[MT5] modify_sl order_send returned None request={request}")
                return False

            retcode = getattr(result, "retcode", None)
            comment = getattr(result, "comment", None)
            logger.info(f"[MT5] modify_sl retcode={retcode} comment={comment} ticket={ticket_int} new_sl={new_sl}")
            return retcode == mt5.TRADE_RETCODE_DONE
        except Exception as e:
            logger.error(f"[MT5] modify_sl exception: {type(e).__name__}: {e}")
            return False


    def close_position(self, order_id: str, max_retries: int = 2) -> bool:
        """Close position with retry logic for race conditions.
        
        Args:
            order_id: Position ticket to close
            max_retries: Number of retry attempts for retcode=10036
            
        Returns:
            True if position closed successfully, False otherwise
        """
        if mt5 is None:
            return False
        ticket_str = str(order_id)
        if not ticket_str.isdigit():
            logger.warning(f"[MT5] close_position invalid order_id={order_id}")
            return False
        ticket_int = int(ticket_str)

        # Diagnostic only: do not gate/abort based on the general positions_get()
        positions_list_all = []
        try:
            all_pos = mt5.positions_get()
            if all_pos is not None:
                tmp_list = []
                for pp in all_pos:
                    v = getattr(pp, 'ticket', None)
                    if v is None:
                        v = getattr(pp, 'position_id', None)
                    if v is None:
                        v = getattr(pp, 'id', None)
                    try:
                        tmp_list.append(int(v))
                    except Exception:
                        tmp_list.append(0)
                positions_list_all = tmp_list
        except Exception:
            positions_list_all = []

        logger.info(
            f"[CLOSE_DIAG] ts={int(time.time())} ticket_int={ticket_int} open_position_tickets={positions_list_all}"
        )

        # Real retries: only for the ticket-specific positions_get(ticket=...)
        retry_attempts = 5
        positions = None
        for i in range(retry_attempts):
            positions = mt5.positions_get(ticket=ticket_int)
            if positions is not None:
                break
            if i < retry_attempts - 1:
                time.sleep(0.5)

        # If positions_get(ticket=...) keeps returning None -> IPC/transient failure
        if positions is None:
            # best-effort reconnect (do NOT change order_send/retcode handling)
            logger.warning(
                f"[MT5] close_position IPC unstable after retries; best-effort reconnect ticket={ticket_int}"
            )
            try:
                if mt5 is not None:
                    try:
                        if mt5.terminal_info() is None:
                            mt5.initialize()
                    except Exception as e:
                        logger.warning(
                            f"[MT5] close_position reconnect: terminal_info/initialize exception={type(e).__name__} {e} ticket={ticket_int}"
                        )

                    try:
                        # re-login with same creds (if supported)
                        from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
                        mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
                    except Exception as e:
                        logger.warning(
                            f"[MT5] close_position reconnect: mt5.login exception={type(e).__name__} {e} ticket={ticket_int}"
                        )

                    # final single attempt after reconnect
                    positions = mt5.positions_get(ticket=ticket_int)
            except Exception as e:
                logger.warning(
                    f"[MT5] close_position reconnect: positions_get exception={type(e).__name__} {e} ticket={ticket_int}"
                )
                positions = None

            if positions is None:
                logger.error(
                    f"[MT5] close_position reconnect did not fix IPC (positions_get still None) ticket={ticket_int}"
                )

        if positions is None:
            logger.error(
                f"[MT5] close_position query failed - IPC unstable (ticket query returned None) ticket={ticket_int}"
            )
            return False

        # If we got a list/tuple but it doesn't contain the ticket (or it's empty) -> position not found
        if not positions:
            logger.warning(f"[MT5] close_position position not found (empty) ticket={ticket_int}")
            return False

        # positions_get(ticket=...) usually returns the single position, but be defensive.
        found = None
        for pp in positions:
            v = getattr(pp, "ticket", None)
            if v is None:
                v = getattr(pp, "position_id", None)
            if v is None:
                v = getattr(pp, "id", None)
            try:
                if v is not None and int(v) == ticket_int:
                    found = pp
                    break
            except Exception:
                continue

        if found is None:
            logger.warning(
                f"[MT5] close_position position not found in positions_get(ticket=...) ticket={ticket_int} positions_len={len(positions)}"
            )
            return False

        p = found
        d = p._asdict() if hasattr(p, "_asdict") else p.__dict__
        symbol = d.get("symbol")
        volume = d.get("volume")
        ptype = d.get("type")
        if not symbol or volume is None:
            logger.warning(f"[MT5] close_position missing symbol/volume ticket={ticket_int}")
            return False
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.warning(f"[MT5] close_position no tick symbol={symbol}")
            return False

        is_buy = (int(ptype) == 0) if isinstance(ptype, (int, float, str)) else str(ptype).lower() in ["0", "buy"]
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
            "comment": "PostEntryManager",
        }
        
        # Retry logic for race conditions (retcode=10036: Position doesn't exist)
        last_retcode = None
        for attempt in range(max_retries):
            result = mt5.order_send(request)
            if result is None:
                logger.error(f"[MT5] close_position order_send returned None request={request}")
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                    continue
                return False

            retcode = getattr(result, "retcode", None)
            comment = getattr(result, "comment", None)
            last_retcode = retcode
            
            logger.info(f"[MT5] close_position retcode={retcode} comment={comment} ticket={ticket_int} attempt={attempt+1}/{max_retries}")
            
            # Success
            if retcode == mt5.TRADE_RETCODE_DONE:
                return True
            
            # Retry on specific error codes
            if retcode in [10036, 10013, 10014]:  # Position doesn't exist, invalid request, no changes
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[MT5] close_position retryable error retcode={retcode} ticket={ticket_int}, "
                        f"waiting 1s before retry..."
                    )
                    time.sleep(1.0)
                    # Re-query position before retry
                    positions = mt5.positions_get(ticket=ticket_int)
                    if not positions:
                        logger.error(f"[MT5] close_position position gone during retry ticket={ticket_int}")
                        return False
                    continue
                else:
                    logger.error(
                        f"[MT5] close_position failed after {max_retries} attempts retcode={retcode} "
                        f"comment={comment} ticket={ticket_int}"
                    )
                    return False
            else:
                # Non-retryable error
                logger.error(
                    f"[MT5] close_position non-retryable error retcode={retcode} "
                    f"comment={comment} ticket={ticket_int}"
                )
                return False
        
        return False

