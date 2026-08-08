"""Idempotent order submission — one logical signal, at most one position.

The problem this solves
-----------------------
``mt5_direct.open_trade`` iterates over filling modes and, on either an
``order_send`` exception or a ``None`` result, moves straight to the next mode:

    for filling_value, filling_name in available_filling_modes:
        try:
            result = mt5.order_send(request)
        except Exception:
            continue          # <-- sends a second order
        if result is None:
            continue          # <-- sends a second order

A ``None`` result means *the outcome is unknown*, not *the order failed*. If the
order reached the broker and executed but the reply was lost — and the live logs
show 198 ``No IPC connection`` errors, so that path is real — the retry opens a
second position for the same signal.

The rule enforced here
----------------------
``UNKNOWN`` is never treated as ``FAILED``. Before any retry the broker is
queried for a position matching this signal. A new order is sent only when there
is positive evidence the previous attempt did **not** execute.

Signal identity
---------------
Each logical entry gets a deterministic ``magic`` number derived from
(symbol, direction, signal timestamp). Deterministic matters: a random id per
retry would make the reconciliation lookup useless, because the position left by
the lost attempt would carry the *previous* id.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from utils.logger import get_logger

logger = get_logger("order_idempotency")

# MT5 stores `magic` as a 32-bit signed int; stay comfortably inside it.
_MAGIC_MODULUS = 2_000_000_000

# How far back to look when deciding whether a position belongs to this signal.
POSITION_MATCH_WINDOW_SEC = 120.0


class ExecutionState(str, Enum):
    """Lifecycle of one submission attempt."""

    NEW = "NEW"
    SUBMITTING = "SUBMITTING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    EXECUTED = "EXECUTED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"
    RECONCILING = "RECONCILING"
    CONFIRMED_EXECUTED = "CONFIRMED_EXECUTED"
    CONFIRMED_NOT_EXECUTED = "CONFIRMED_NOT_EXECUTED"


# States that mean "a position exists for this signal — never send another".
TERMINAL_EXECUTED = frozenset({ExecutionState.EXECUTED, ExecutionState.CONFIRMED_EXECUTED})
# States that permit a further attempt.
RETRYABLE = frozenset({ExecutionState.REJECTED, ExecutionState.CONFIRMED_NOT_EXECUTED})


@dataclass
class SignalIdentity:
    """Deterministic identity for one logical entry signal."""

    symbol: str
    direction: str
    signal_ts: int          # candle/decision timestamp, NOT wall clock at retry
    strategy: str = "V3"

    @property
    def key(self) -> str:
        return f"{self.strategy}:{self.symbol}:{self.direction.upper()}:{self.signal_ts}"

    @property
    def magic(self) -> int:
        """Stable 32-bit magic number for this signal.

        Derived from the key so every retry of the same signal carries the same
        value, which is what makes broker-side reconciliation possible.
        """
        digest = hashlib.sha256(self.key.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % _MAGIC_MODULUS


@dataclass
class ExecutionRecord:
    """Tracks one signal's submission across attempts."""

    identity: SignalIdentity
    state: ExecutionState = ExecutionState.NEW
    attempts: int = 0
    order_id: Optional[str] = None
    history: list = field(default_factory=list)

    def transition(self, state: ExecutionState, note: str = "") -> None:
        self.history.append((time.time(), self.state, state, note))
        logger.info(
            "[IDEMPOTENCY] %s: %s -> %s%s",
            self.identity.key, self.state.value, state.value,
            f" ({note})" if note else "",
        )
        self.state = state

    @property
    def already_executed(self) -> bool:
        return self.state in TERMINAL_EXECUTED

    @property
    def may_retry(self) -> bool:
        return self.state in RETRYABLE


def find_position_for_signal(
    identity: SignalIdentity,
    positions_provider,
    now: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Look for a broker position belonging to this signal.

    ``positions_provider`` is a callable returning an iterable of position-like
    objects/dicts, injected so this stays testable without a live terminal.

    Matching is by magic number first (exact), then by
    symbol + direction + recency as a fallback for positions opened before magic
    numbers were in use.
    """
    now = now if now is not None else time.time()

    try:
        positions = positions_provider(identity.symbol)
    except Exception as exc:
        # A failed lookup is NOT evidence of non-execution.
        logger.error(
            "[IDEMPOTENCY] position lookup failed for %s: %s", identity.key, exc
        )
        raise

    if not positions:
        return None

    wanted_is_buy = identity.direction.strip().lower() in {"buy", "long", "0"}

    for position in positions:
        data = _as_dict(position)

        if int(data.get("magic") or 0) == identity.magic:
            logger.warning(
                "[IDEMPOTENCY] found position by magic=%s for %s — already executed",
                identity.magic, identity.key,
            )
            return data

        # Fallback: same symbol, same side, opened inside the window.
        if str(data.get("symbol") or "") != identity.symbol:
            continue
        pos_is_buy = int(data.get("type") or 0) == 0
        if pos_is_buy != wanted_is_buy:
            continue
        opened = float(data.get("time") or 0)
        if opened and (now - opened) <= POSITION_MATCH_WINDOW_SEC:
            logger.warning(
                "[IDEMPOTENCY] found recent %s %s position opened %.1fs ago — "
                "treating as this signal's execution",
                identity.symbol, identity.direction, now - opened,
            )
            return data

    return None


def resolve_unknown_outcome(
    record: ExecutionRecord,
    positions_provider,
    now: Optional[float] = None,
) -> ExecutionState:
    """Decide what an ambiguous submission actually did.

    Returns CONFIRMED_EXECUTED, CONFIRMED_NOT_EXECUTED, or UNKNOWN.

    UNKNOWN is returned when the broker cannot be queried — and UNKNOWN must
    block a retry, because sending again could double the position.
    """
    record.transition(ExecutionState.RECONCILING, "resolving ambiguous result")

    try:
        position = find_position_for_signal(record.identity, positions_provider, now=now)
    except Exception:
        record.transition(
            ExecutionState.UNKNOWN,
            "broker unreachable — cannot prove non-execution, retry refused",
        )
        return ExecutionState.UNKNOWN

    if position is not None:
        record.order_id = str(
            position.get("ticket") or position.get("order") or position.get("id") or ""
        )
        record.transition(ExecutionState.CONFIRMED_EXECUTED, f"ticket={record.order_id}")
        return ExecutionState.CONFIRMED_EXECUTED

    record.transition(
        ExecutionState.CONFIRMED_NOT_EXECUTED, "no matching position at broker"
    )
    return ExecutionState.CONFIRMED_NOT_EXECUTED


def may_send_another_order(record: ExecutionRecord) -> tuple:
    """Gate in front of every (re)submission.

    Returns (allowed, reason).
    """
    if record.already_executed:
        return False, f"signal already executed (state={record.state.value})"
    if record.state is ExecutionState.UNKNOWN:
        return False, "previous attempt outcome unknown — refusing to risk a duplicate"
    if record.state is ExecutionState.SUBMITTING:
        return False, "a submission is already in flight for this signal"
    return True, ""


def _as_dict(position: Any) -> Dict[str, Any]:
    if isinstance(position, dict):
        return position
    if hasattr(position, "_asdict"):
        return position._asdict()
    return getattr(position, "__dict__", {}) or {}
