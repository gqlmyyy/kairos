from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PositionState:
    state: str
    # internal flags for gating
    be_done: bool = False
    partial_done: bool = False
    trailing_active: bool = False
    profit_locked: bool = False
    # MFE/MAE tracking for red flags (persistent via DB)
    mfe: float = 0.0
    mae: float = 0.0
    # Spread cache
    last_spread: float = 0.0


class PositionStateMachine:
    """Simple state machine based on snapshot + DB metadata.

    Current implementation uses only what is available from reconciliation metadata.
    """

    ORDER_STATES = [
        "NEW",
        "OPENED",
        "BREAKEVEN_ACTIVE",
        "PARTIAL_CLOSED",
        "TRAILING_ACTIVE",
        "PROFIT_LOCKED",
        "EXIT_PENDING",
        "CLOSED",
    ]

    def __init__(self) -> None:
        # in-memory per manager cycle could be extended
        pass

    def derive_state(self, snapshot: Dict[str, Any], db_row: Optional[Dict[str, Any]] = None) -> PositionState:
        db_row = db_row or {}

        be_done = bool(db_row.get("breakeven_done") or 0)
        trailing_done = bool(db_row.get("trailing_done") or 0)

        # Persistent mfe/mae (added via DB migration). Default to 0.0 for safety.
        try:
            mfe = float(db_row.get("mfe") or 0.0)
        except Exception:
            mfe = 0.0

        try:
            mae = float(db_row.get("mae") or 0.0)
        except Exception:
            mae = 0.0

        trade = snapshot.get("trade", {})
        # partial/trailing/profit locked not persisted in current schema; infer via sl movement
        state = "OPENED"
        if be_done:
            state = "BREAKEVEN_ACTIVE"
        if trailing_done:
            state = "TRAILING_ACTIVE"

        # fallback partial/profit locked remain false for now
        return PositionState(
            state=state,
            be_done=be_done,
            partial_done=False,
            trailing_active=trailing_done,
            profit_locked=False,
            mfe=mfe,
            mae=mae,
        )

