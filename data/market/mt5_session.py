"""Single owner of the MetaTrader5 session.

Every MT5 call in the project should go through this module. It exists to fix
three concrete problems found in the previous architecture:

1. **IPC contention.** Four threads (main cycle, post-entry manager,
   reconciliation, watchdog) called the MetaTrader5 library concurrently. That
   library wraps a single IPC channel to the terminal and is not thread-safe;
   the logs showed 198 ``No IPC connection`` errors and 726 watchdog
   disconnects as a result. A process-wide reentrant lock serialises access.

2. **login() on every call.** Both ``execution/mt5_direct.py`` and
   ``data/market/hybrid_client.py`` called ``mt5.login()`` inside their
   "ensure initialised" helper, i.e. on every candle fetch and every order.
   ``login()`` is a full round trip to the broker, not a local check. Here the
   session is established once and subsequently verified with
   ``account_info()``, which is a cheap local read.

3. **Two competing sessions.** The bot held its own MT5 session while
   QuantDinger held another against the same terminal. With QuantDinger gone
   there is exactly one session, owned here.

Thread safety: ``mt5_call()`` is reentrant, so a function holding the lock may
call another function that also takes it without deadlocking.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Optional

from config import MT5_LOGIN, MT5_PASSWORD, MT5_PATH, MT5_SERVER
from utils.logger import get_logger

logger = get_logger("mt5_session")

try:
    import MetaTrader5 as mt5  # type: ignore
    MT5_AVAILABLE = True
except Exception:  # pragma: no cover - platform without the library
    mt5 = None
    MT5_AVAILABLE = False

# Reentrant: a locked function may call another locked function.
_lock = threading.RLock()
_initialized = False
_login_done = False


@contextmanager
def mt5_call():
    """Serialise access to the MT5 IPC channel.

    Usage::

        with mt5_call():
            positions = mt5.positions_get()
    """
    with _lock:
        yield mt5


def is_available() -> bool:
    """True when the MetaTrader5 library imported successfully."""
    return MT5_AVAILABLE and mt5 is not None


def is_healthy() -> bool:
    """Cheap local check that the session is alive.

    Uses ``account_info()`` rather than ``login()``: it reads terminal state
    without a broker round trip, so it is safe to call on a short interval.
    """
    if not is_available():
        return False
    try:
        with mt5_call():
            return mt5.account_info() is not None
    except Exception as exc:
        logger.warning("[MT5_SESSION] health check raised: %s", exc)
        return False


def ensure_session(force_relogin: bool = False) -> bool:
    """Initialise and log in once; verify cheaply thereafter.

    Args:
        force_relogin: re-authenticate even if a session already looks healthy.
            Used by the watchdog when recovering from a dropped connection.

    Returns:
        True when the session is usable.
    """
    global _initialized, _login_done

    if not is_available():
        logger.error("[MT5_SESSION] MetaTrader5 library not available")
        return False

    # config.py no longer carries hardcoded fallback credentials, so a blank or
    # partially-filled .env now reaches here instead of silently authenticating
    # against somebody else's demo account. Name the missing variables rather
    # than letting mt5.login() return a generic failure code.
    missing = [
        name for name, value in (
            ("MT5_LOGIN", MT5_LOGIN),
            ("MT5_PASSWORD", MT5_PASSWORD),
            ("MT5_SERVER", MT5_SERVER),
        )
        if not value
    ]
    if missing:
        logger.error(
            "[MT5_SESSION] cannot connect: %s not set in .env — "
            "refusing to attempt a login with incomplete credentials",
            ", ".join(missing),
        )
        return False

    with _lock:
        # Fast path: already established and still healthy.
        if _initialized and _login_done and not force_relogin:
            try:
                if mt5.account_info() is not None:
                    return True
            except Exception:
                pass  # fall through and rebuild the session
            logger.warning("[MT5_SESSION] session went stale, re-establishing")
            _initialized = False
            _login_done = False

        # Terminal initialisation.
        try:
            if not mt5.terminal_info():
                ok = mt5.initialize(path=MT5_PATH) if MT5_PATH else mt5.initialize()
                if not ok:
                    logger.error("[MT5_SESSION] initialize() failed: %s", mt5.last_error())
                    return False
            _initialized = True
        except Exception as exc:
            logger.error("[MT5_SESSION] initialize() raised: %s", exc)
            return False

        # Authenticate. This is the expensive step, so it happens here only.
        try:
            if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
                logger.error("[MT5_SESSION] login() failed: %s", mt5.last_error())
                _login_done = False
                return False
            _login_done = True
            account = mt5.account_info()
            logger.info(
                "[MT5_SESSION] connected login=%s server=%s balance=%s",
                MT5_LOGIN, MT5_SERVER,
                getattr(account, "balance", "?") if account else "?",
            )
            return True
        except Exception as exc:
            logger.error("[MT5_SESSION] login() raised: %s", exc)
            _login_done = False
            return False


def ensure_symbol(symbol: str) -> bool:
    """Make a symbol visible in Market Watch so its data can be read."""
    if not ensure_session():
        return False
    try:
        with mt5_call():
            info = mt5.symbol_info(symbol)
            if info is None:
                logger.warning("[MT5_SESSION] unknown symbol %s", symbol)
                return False
            if not info.visible and not mt5.symbol_select(symbol, True):
                logger.warning("[MT5_SESSION] symbol_select(%s) failed", symbol)
                return False
            return True
    except Exception as exc:
        logger.warning("[MT5_SESSION] ensure_symbol(%s) raised: %s", symbol, exc)
        return False


def get_account_info() -> Optional[Any]:
    """Raw MT5 account info, or None when unavailable."""
    if not ensure_session():
        return None
    try:
        with mt5_call():
            return mt5.account_info()
    except Exception as exc:
        logger.warning("[MT5_SESSION] account_info raised: %s", exc)
        return None


def shutdown() -> None:
    """Close the session. Intended for tests and clean process exit."""
    global _initialized, _login_done
    with _lock:
        if is_available():
            try:
                mt5.shutdown()
            except Exception:
                pass
        _initialized = False
        _login_done = False


def _reset_state_for_tests() -> None:
    """Test hook: forget the cached session flags without touching MT5."""
    global _initialized, _login_done
    with _lock:
        _initialized = False
        _login_done = False
