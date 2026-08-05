# Trading Bot V3 - execution/mt5_watchdog.py
# MT5 connection watchdog - uses V1 proven reconnect logic

import time
import threading
from utils.logger import get_logger
from config import QUANTDINGER_URL, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH, WATCHDOG_INTERVAL, WATCHDOG_FAIL_LIMIT

logger = get_logger("mt5_watchdog")

# We import these inside functions to avoid circular imports
def _get_headers():
    from execution.quantdinger_client import get_headers
    return get_headers()

def _login():
    from execution.quantdinger_client import login
    return login()

def check_mt5_connection():
    """Check if MT5 is connected via QuantDinger status endpoint"""
    try:
        import requests
        r = requests.get(
            f"{QUANTDINGER_URL}/api/mt5/status",
            headers=_get_headers(),
            timeout=5
        )
        data = r.json()
        return data.get("connected", False)
    except Exception as e:
        logger.error(f"MT5 check error: {e}")
        return False

def reconnect_mt5(max_retries: int = 3, delay: float = 2.0):
    """Reconnect MT5 using stored credentials with smart retries.

    Requirements:
    - 3 retries with delay=2s
    - reinitialize/refresh token each attempt
    - log clear error when all attempts fail
    """
    for attempt in range(max_retries):
        try:
            import requests

            # Step 1: Refresh token first (V1 approach)
            _login()

            # Step 2: Check if already connected via account endpoint
            r = requests.get(
                f"{QUANTDINGER_URL}/api/mt5/account",
                headers=_get_headers(),
                timeout=5,
            )
            data = r.json()
            if data.get("success"):
                logger.info("MT5 already connected")
                return True

            # Step 3: Connect with full config
            mt5_config = {
                "login": MT5_LOGIN,
                "password": MT5_PASSWORD,
                "server": MT5_SERVER,
                "path": MT5_PATH,
            }
            r2 = requests.post(
                f"{QUANTDINGER_URL}/api/mt5/connect",
                headers=_get_headers(),
                json=mt5_config,
                timeout=15,
            )
            result = r2.json()

            if result.get("success"):
                balance = result.get("account", {}).get("balance", 0)
                logger.info(f"MT5 reconnected! Balance: {balance}")
                return True

            logger.warning(f"MT5 reconnect attempt failed: {result}")
        except Exception as e:
            logger.warning(f"MT5 reconnect attempt error (attempt {attempt+1}/{max_retries}): {e}")

        if attempt < max_retries - 1:
            time.sleep(delay)

    logger.error(f"MT5 reconnect failed after {max_retries} attempts (delay={delay}s)")
    return False

def watchdog_loop():
    consecutive_failures = 0
    
    from telegram.notifier import notify_alert, notify_status
    
    # Initial wait before starting checks
    time.sleep(60)
    
    while True:
        try:
            connected = check_mt5_connection()
            
            if not connected:
                consecutive_failures += 1
                logger.warning(f"MT5 disconnected! Attempt {consecutive_failures}/{WATCHDOG_FAIL_LIMIT}")
                
                if consecutive_failures == 1:
                    notify_alert("⚠️ MT5 disconnected - attempting reconnect...")
                
                success = reconnect_mt5(max_retries=3, delay=2.0)
                
                if success:
                    consecutive_failures = 0
                    logger.info("MT5 reconnected successfully!")
                    notify_status("✅ MT5 reconnected successfully")
                else:
                    if consecutive_failures >= WATCHDOG_FAIL_LIMIT:
                        notify_alert(f"❌ MT5 failed to reconnect after {WATCHDOG_FAIL_LIMIT} attempts!\nCheck MT5 manually")
            else:
                if consecutive_failures > 0:
                    logger.info("MT5 connection restored")
                    notify_status("✅ MT5 connection restored")
                consecutive_failures = 0
                
        except Exception as e:
            logger.error(f"Watchdog error: {e}")
        
        time.sleep(WATCHDOG_INTERVAL)

def start_mt5_watchdog():
    thread = threading.Thread(target=watchdog_loop, daemon=True)
    thread.start()
    logger.info("MT5 watchdog started (V1 reconnect logic)")
