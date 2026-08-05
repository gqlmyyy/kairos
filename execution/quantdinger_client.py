# Trading Bot V3 - execution/quantdinger_client.py
# QuantDinger API client - all communication goes through this

import requests
import time
import re
from typing import Optional

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    mt5 = None

from utils.logger import get_logger
from config import (
    QUANTDINGER_URL,
    QUANTDINGER_USERNAME,
    QUANTDINGER_PASSWORD,
    MAX_RETRIES,
    RETRY_DELAY,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    MT5_PATH,
    QUANTDINGER_ORDER_TYPE_FILLING,
    QUANTDINGER_ORDER_TYPE_FILLING_STRATEGY,
)

from core.exceptions import QuantDingerAuthError, QuantDingerConnectionError

logger = get_logger("qd_client")

_token = None

def login() -> str:
    global _token
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{QUANTDINGER_URL}/api/auth/login",
                json={"username": QUANTDINGER_USERNAME, "password": QUANTDINGER_PASSWORD},
                timeout=10,
            )
            data = r.json()
            if data.get("code") == 1:
                _token = data["data"]["token"]
                logger.info("Logged in to QuantDinger")
                return _token
            elif data.get("code") == 200:
                _token = data.get("data", {}).get("token", "")
                if _token:
                    logger.info("Logged in to QuantDinger (code=200)")
                    return _token
        except Exception as e:
            logger.error(f"Login attempt {attempt+1} failed: {e}")
            time.sleep(RETRY_DELAY)
    raise QuantDingerAuthError("Failed to login to QuantDinger")


_token = None
_token_time = 0
TOKEN_TTL = 3600  # ساعة واحدة

def get_token() -> str:
    global _token, _token_time
    if not _token or (time.time() - _token_time) > TOKEN_TTL:
        return login()
    return _token


def login() -> str:
    global _token, _token_time
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{QUANTDINGER_URL}/api/auth/login",
                json={"username": QUANTDINGER_USERNAME, "password": QUANTDINGER_PASSWORD},
                timeout=10,
            )
            data = r.json()
            if data.get("code") == 1:
                _token = data["data"]["token"]
                _token_time = time.time()
                logger.info("Logged in to QuantDinger")
                return _token
        except Exception as e:
            logger.error(f"Login attempt {attempt+1} failed: {e}")
            time.sleep(RETRY_DELAY)
    raise QuantDingerAuthError("Failed to login to QuantDinger")


def get_headers() -> dict:
    return {"Authorization": f"Bearer {get_token()}", "Content-Type": "application/json"}


def _refresh_on_401(data: dict) -> bool:
    """Returns True if token was refreshed (caller should retry)."""
    if data.get("code") == 401 or "Token" in str(data):
        login()
        return True
    return False


def _clean_comment(text: str) -> str:
    clean = re.sub(r"[^\x20-\x7E]", "", str(text))
    return clean[:31]


def open_trade(symbol, direction, size, sl, tp, reason) -> dict:
    side = "buy" if direction == "BUY" else "sell"


    def _get_filling_mode_mt5(symbol_name: str) -> Optional[int]:
        """Hybrid query: get filling_mode from MetaTrader5 directly (fail-fast => None on any failure)."""
        if mt5 is None:
            logger.error("FATAL: MetaTrader5 library is not available in this environment")
            return None

        try:
            # Ensure MT5 terminal is initialized
            if not mt5.terminal_info():
                if not mt5.initialize():
                    logger.error("FATAL: mt5.initialize() failed; filling_mode queries unavailable")
                    return None

            # Activate symbol in Market Watch (required)
            if not mt5.symbol_select(symbol_name, True):
                logger.error(f"FATAL: Cannot select symbol {symbol_name} in MT5")
                return None

            symbol_info = mt5.symbol_info(symbol_name)
            if symbol_info is None:
                logger.error(f"FATAL: symbol_info is None for {symbol_name}")
                return None

            # Diagnostic log (best-effort)
            try:
                filling_mode_dbg = getattr(symbol_info, "filling_mode", None)
                trade_mode_dbg = getattr(symbol_info, "trade_mode", None)
                logger.info(
                    f"DEBUG: symbol_info for {symbol_name} - filling_mode={filling_mode_dbg}, trade_mode={trade_mode_dbg}"
                )
            except Exception:
                pass

            filling_mode = getattr(symbol_info, "filling_mode", None)
            if filling_mode is None:
                logger.error(f"ERROR: filling_mode is None for {symbol_name} (symbol_info may be incomplete)")
                return None

            try:
                filling_mode_int = int(filling_mode)
            except Exception:
                logger.error(
                    f"ERROR: filling_mode for {symbol_name} is not convertible to int: {filling_mode}"
                )
                return None

            # Broker rule in your case: expect FOK=1
            if filling_mode_int != 1:
                logger.error(
                    f"ERROR: filling_mode is {filling_mode_int}, expected 1 (FOK) for this broker."
                )
                return None

            return filling_mode_int
        except Exception as e:
            logger.error(f"FATAL: MT5 filling_mode query failed for {symbol_name}: {e}")
            return None

    payload_base = {
        "symbol": symbol,
        "side": side,
        "volume": size,
        "stop_loss": sl,
        "take_profit": tp,
        "comment": _clean_comment(reason),
    }

    # ==============================
    # Fail-Fast filling mode
    # ==============================
    dynamic_filling_mode = _get_filling_mode_mt5(symbol)

    # dynamic_filling_mode should already be an int from MT5.
    try:
        dynamic_filling_mode = int(dynamic_filling_mode) if dynamic_filling_mode is not None else None
    except Exception:
        dynamic_filling_mode = None

    logger.info(
        "[VERIFY][EXECUTION_TRUTH] filling_mode for symbol="
        f"{symbol} dynamic_filling_mode={dynamic_filling_mode}"
    )

    if dynamic_filling_mode is None or dynamic_filling_mode == 0:
        last_error = f"FATAL: Cannot fetch filling mode for {symbol}"
        logger.error(last_error)
        return {
            "status": "error",
            "error": last_error,
            "order_id": None,
            "price": None,
            "symbol": symbol,
            "direction": direction,
            "volume": float(size) if size is not None else None,
            "timestamp": time.time(),
            "raw_response": {},
        }

    payload = dict(payload_base)
    payload["type_filling"] = dynamic_filling_mode

    # Attempt #1: with dynamic filling_mode
    logger.debug(
        f"[EXECUTION] attempt#1 trying payload={payload}"
    )

    logger.info(
        f"[EXECUTION] trying order with filling_mode={dynamic_filling_mode} symbol={symbol} direction={direction} size={size}"
    )

    def _extract_error_text(resp_dict):

        if not isinstance(resp_dict, dict):
            return str(resp_dict)
        # QuantDinger may return {error: ...} or {message: ...} or similar
        err = resp_dict.get("error") or resp_dict.get("message") or ""
        if err:
            return str(err)
        return str(resp_dict)

    try:
        r = requests.post(
            f"{QUANTDINGER_URL}/api/mt5/order",
            headers=get_headers(),
            json=payload,
            timeout=15,
        )

        data = r.json() if hasattr(r, "json") else {}
        response = data if isinstance(data, dict) else {}
        last_response = response

        if isinstance(data, dict) and (data.get("success") or data.get("code") in [200, 1]):
            raw = data.get("data", data)
            if not isinstance(raw, dict):
                raw = data

            order_id = raw.get("order_id") if isinstance(raw, dict) else None
            if order_id is None and isinstance(raw, dict):
                order_id = raw.get("id") or raw.get("ticket")

            price = raw.get("price") if isinstance(raw, dict) else None
            if price is None and isinstance(raw, dict):
                price = raw.get("entry") or raw.get("price_open")

            try:
                price = float(price) if price is not None else None
            except Exception:
                price = None

            try:
                order_id = str(order_id) if order_id is not None else None
            except Exception:
                order_id = None

            if price is None:
                try:
                    price = float(payload.get("take_profit", 0)) if payload.get("take_profit") is not None else None
                except Exception:
                    price = None

            volume = float(size) if size is not None else None
            ts = time.time()

            logger.info(
                "[VERIFY][EXECUTION_TRUTH] reached_execution=true "
                f"order_sent=true response={response}"
            )

            return {
                "status": "success",
                "order_id": order_id,
                "price": price,
                "symbol": symbol,
                "direction": direction,
                "volume": volume,
                "timestamp": ts,
                "raw_response": last_response,
            }

        # token refresh retry (single retry only, no filling-mode fallback)
        if isinstance(data, dict) and _refresh_on_401(data):
            # refresh already happened inside _refresh_on_401; attempt once more
            r2 = requests.post(
                f"{QUANTDINGER_URL}/api/mt5/order",
                headers=get_headers(),
                json=payload,
                timeout=15,
            )
            data2 = r2.json() if hasattr(r2, "json") else {}
            if isinstance(data2, dict) and (data2.get("success") or data2.get("code") in [200, 1]):
                raw = data2.get("data", data2)
                if not isinstance(raw, dict):
                    raw = data2

                order_id = raw.get("order_id") if isinstance(raw, dict) else None
                if order_id is None and isinstance(raw, dict):
                    order_id = raw.get("id") or raw.get("ticket")

                price = raw.get("price") if isinstance(raw, dict) else None
                if price is None and isinstance(raw, dict):
                    price = raw.get("entry") or raw.get("price_open")

                try:
                    price = float(price) if price is not None else None
                except Exception:
                    price = None

                try:
                    order_id = str(order_id) if order_id is not None else None
                except Exception:
                    order_id = None

                volume = float(size) if size is not None else None
                ts = time.time()

                logger.info(
                    "[VERIFY][EXECUTION_TRUTH] order success after 401 refresh response="
                    f"{data2}"
                )

                return {
                    "status": "success",
                    "order_id": order_id,
                    "price": price,
                    "symbol": symbol,
                    "direction": direction,
                    "volume": volume,
                    "timestamp": ts,
                    "raw_response": data2,
                }

            last_error = str(data2)
            return {
                "status": "error",
                "error": last_error,
                "order_id": None,
                "price": None,
                "symbol": symbol,
                "direction": direction,
                "volume": float(size) if size is not None else None,
                "timestamp": time.time(),
                "raw_response": data2 if isinstance(data2, dict) else {},
            }

        # If rejected due to Unsupported filling mode -> retry once with type_filling removed
        error_text = _extract_error_text(data)
        logger.info(
            f"[EXECUTION] attempt#1 finished symbol={symbol} error_text={error_text} response={data}"
        )
        if "Unsupported filling mode" in error_text:

            logger.warning(
                f"[EXECUTION] rejected Unsupported filling mode for symbol={symbol} filling_mode={dynamic_filling_mode}. "
                f"Retrying without type_filling. error_text={error_text}"
            )

            payload_retry = dict(payload_base)
            # send once more WITHOUT type_filling

            logger.info(
                f"[EXECUTION] retrying order without type_filling symbol={symbol} direction={direction} size={size}"
            )

            # Ensure type_filling is completely removed (no key at all)
            if "type_filling" in payload_retry:
                del payload_retry["type_filling"]

            logger.debug(
                f"[EXECUTION] attempt#2 (no type_filling) payload={payload_retry}"
            )

            r_retry = requests.post(
                f"{QUANTDINGER_URL}/api/mt5/order",
                headers=get_headers(),
                json=payload_retry,
                timeout=15,
            )

            data_retry = r_retry.json() if hasattr(r_retry, "json") else {}
            last_response = data_retry if isinstance(data_retry, dict) else {}

            if isinstance(data_retry, dict) and (data_retry.get("success") or data_retry.get("code") in [200, 1]):
                raw = data_retry.get("data", data_retry)
                if not isinstance(raw, dict):
                    raw = data_retry

                order_id = raw.get("order_id") if isinstance(raw, dict) else None
                if order_id is None and isinstance(raw, dict):
                    order_id = raw.get("id") or raw.get("ticket")

                price = raw.get("price") if isinstance(raw, dict) else None
                if price is None and isinstance(raw, dict):
                    price = raw.get("entry") or raw.get("price_open")

                try:
                    price = float(price) if price is not None else None
                except Exception:
                    price = None

                try:
                    order_id = str(order_id) if order_id is not None else None
                except Exception:
                    order_id = None

                volume = float(size) if size is not None else None
                ts = time.time()

                logger.info(
                    f"[EXECUTION] success on retry without type_filling symbol={symbol} response={data_retry}"
                )

                return {
                    "status": "success",
                    "order_id": order_id,
                    "price": price,
                    "symbol": symbol,
                    "direction": direction,
                    "volume": volume,
                    "timestamp": ts,
                    "raw_response": data_retry,
                }

            # If retry#2 also unsupported -> retry#3 with type_filling=0
            error_text_retry = _extract_error_text(data_retry)
            if "Unsupported filling mode" in error_text_retry:
                logger.warning(
                    f"[EXECUTION] attempt#2 rejected Unsupported filling mode for symbol={symbol}. Retrying with type_filling=0"
                )

                payload_last = dict(payload_base)
                payload_last["type_filling"] = 0

                logger.debug(
                    f"[EXECUTION] attempt#3 (type_filling=0) payload={payload_last}"
                )

                r_last = requests.post(
                    f"{QUANTDINGER_URL}/api/mt5/order",
                    headers=get_headers(),
                    json=payload_last,
                    timeout=15,
                )
                data_last = r_last.json() if hasattr(r_last, "json") else {}
                last_response = data_last if isinstance(data_last, dict) else {}

                if isinstance(data_last, dict) and (data_last.get("success") or data_last.get("code") in [200, 1]):
                    raw = data_last.get("data", data_last)
                    if not isinstance(raw, dict):
                        raw = data_last

                    order_id = raw.get("order_id") if isinstance(raw, dict) else None
                    if order_id is None and isinstance(raw, dict):
                        order_id = raw.get("id") or raw.get("ticket")

                    price = raw.get("price") if isinstance(raw, dict) else None
                    if price is None and isinstance(raw, dict):
                        price = raw.get("entry") or raw.get("price_open")

                    try:
                        price = float(price) if price is not None else None
                    except Exception:
                        price = None

                    try:
                        order_id = str(order_id) if order_id is not None else None
                    except Exception:
                        order_id = None

                    volume = float(size) if size is not None else None
                    ts = time.time()

                    logger.info(
                        f"[EXECUTION] success on attempt#3 type_filling=0 symbol={symbol} response={data_last}"
                    )

                    return {
                        "status": "success",
                        "order_id": order_id,
                        "price": price,
                        "symbol": symbol,
                        "direction": direction,
                        "volume": volume,
                        "timestamp": ts,
                        "raw_response": data_last,
                    }

                last_error = _extract_error_text(data_last)
                logger.error(
                    f"[EXECUTION] attempt#3 type_filling=0 failed symbol={symbol} error={last_error} raw={data_last}"
                )
                return {
                    "status": "error",
                    "error": last_error,
                    "order_id": None,
                    "price": None,
                    "symbol": symbol,
                    "direction": direction,
                    "volume": float(size) if size is not None else None,
                    "timestamp": time.time(),
                    "raw_response": data_last if isinstance(data_last, dict) else {},
                }

            # retry#2 failed for reasons other than unsupported filling mode
            last_error = error_text_retry
            logger.error(
                f"[EXECUTION] retry without type_filling failed symbol={symbol} error={last_error} raw={data_retry}"
            )
            return {
                "status": "error",
                "error": last_error,
                "order_id": None,
                "price": None,
                "symbol": symbol,
                "direction": direction,
                "volume": float(size) if size is not None else None,
                "timestamp": time.time(),
                "raw_response": data_retry if isinstance(data_retry, dict) else {},
            }

        last_error = str(data)

        return {
            "status": "error",
            "error": last_error,
            "order_id": None,
            "price": None,
            "symbol": symbol,
            "direction": direction,
            "volume": float(size) if size is not None else None,
            "timestamp": time.time(),
            "raw_response": last_response if isinstance(last_response, dict) else {},
        }

    except Exception as e:
        last_error = str(e)
        logger.error(f"open_trade request failed: {last_error}")
        return {
            "status": "error",
            "error": last_error,
            "order_id": None,
            "price": None,
            "symbol": symbol,
            "direction": direction,
            "volume": float(size) if size is not None else None,
            "timestamp": time.time(),
            "raw_response": {},
        }







def close_trade(trade_id) -> bool:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{QUANTDINGER_URL}/api/mt5/close",
                headers=get_headers(),
                json={"ticket": trade_id},
                timeout=10,
            )
            data = r.json()
            if data.get("success") or data.get("code") in [200, 1]:
                logger.info(f"Trade {trade_id} closed")
                return True
            elif _refresh_on_401(data):
                continue
        except Exception as e:
            logger.error(f"Close trade attempt {attempt+1}: {e}")
            time.sleep(RETRY_DELAY)
    return False


def get_open_positions() -> list:
    try:
        r = requests.get(
            f"{QUANTDINGER_URL}/api/mt5/positions",
            headers=get_headers(),
            timeout=10,
        )
        data = r.json()
        if _refresh_on_401(data):
            r = requests.get(
                f"{QUANTDINGER_URL}/api/mt5/positions",
                headers=get_headers(),
                timeout=10,
            )
            data = r.json()
        return data.get("data", [])
    except Exception as e:
        logger.error(f"Get positions error: {e}")
        return []


def get_equity() -> float:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(
                f"{QUANTDINGER_URL}/api/mt5/account",
                headers=get_headers(),
                timeout=10,
            )
            data = r.json()
            if data.get("success"):
                equity = float(data.get("data", data).get("equity", 100))
                return max(equity, 1.0)
            elif _refresh_on_401(data):
                continue
        except Exception as e:
            logger.error(f"Equity fetch attempt {attempt+1}: {e}")
            time.sleep(RETRY_DELAY)
    return 100.0


def connect_mt5() -> bool:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{QUANTDINGER_URL}/api/mt5/connect",
                headers=get_headers(),
                json={
                    "login": MT5_LOGIN,
                    "password": MT5_PASSWORD,
                    "server": MT5_SERVER,
                    "path": MT5_PATH,
                },
                timeout=15,
            )
            data = r.json()
            if data.get("success"):
                logger.info("MT5 connected")
                return True
            elif _refresh_on_401(data):
                continue
        except Exception as e:
            logger.error(f"MT5 connect attempt {attempt+1}: {e}")
            time.sleep(RETRY_DELAY)
    return False


def check_mt5_status() -> bool:
    try:
        r = requests.get(
            f"{QUANTDINGER_URL}/api/mt5/status",
            headers=get_headers(),
            timeout=5,
        )
        return r.json().get("connected", False)
    except Exception:
        return False

