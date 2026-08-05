"""Risk Governor - Independent Risk Halt Layer

This module is COMPLETELY SEPARATE from trade opening/management logic.

Responsibilities:
  - Monitor: max_consecutive_losses, daily_loss_limit, max_open_positions
  - Halt new entries when cumulative loss exceeds a threshold (in R units)
  - Persist halt state across bot restarts

CRITICAL CONTRACT:
  - The halt ONLY blocks NEW position entries.
  - It NEVER stops management/protection of an already-open position.
  - It NEVER closes open positions.

Defensive: never raises out of check_halt.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from config import (
    RISK_GOVERNOR_MAX_LOSS_R,
    RISK_GOVERNOR_PERSIST,
    MAX_CONSECUTIVE_LOSSES,
    MAX_OPEN_TRADES,
)

logger = get_logger("risk_governor")

# Default state file (in project root)
DEFAULT_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "risk_governor_state.json",
)

# TTL for recorded order_ids (30 days). Prevents unbounded growth.
_DEDUP_ORDER_ID_TTL_DAYS = 30


class RiskGovernor:
    """Independent risk halt governor with persistent state."""

    def __init__(self, state_file: Optional[str] = None) -> None:
        self._state_file = state_file or DEFAULT_STATE_FILE
        self._state: Dict[str, Any] = self._load_state()
        # Dedup: order_ids already recorded as closed (prevents double-counting
        # when trade_manager and reconciliation both close/observe the same trade).
        # NOW PERSISTED: loaded from and saved to the same state file.
        self._recorded_order_ids: set = self._load_recorded_order_ids()

    # ==============================
    # Persistence
    # ==============================
    def _load_state(self) -> Dict[str, Any]:
        default_state = {
            "halted": False,
            "halt_reason": "",
            "halt_sources": [],  # NEW: list of explicit source tags
            "halt_timestamp": None,
            "cumulative_loss_r": 0.0,
            "consecutive_losses": 0,
            "daily_loss_usd": 0.0,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "recorded_order_ids": [],  # NEW: persisted dedup set
        }
        if not RISK_GOVERNOR_PERSIST:
            return default_state

        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                merged = {**default_state, **data}
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if merged.get("date") != today:
                    merged["daily_loss_usd"] = 0.0
                    merged["date"] = today
                # Ensure halt_sources is a list
                if not isinstance(merged.get("halt_sources"), list):
                    merged["halt_sources"] = []

                # ============================================================
                # MIGRATION: Convert old halt_reason (str) to halt_sources (list)
                # ============================================================
                # If old state file has halt_reason but no halt_sources,
                # infer the source from the reason string so the halt
                # is not lost on upgrade.
                if merged.get("halted") and merged.get("halt_reason") and not merged.get("halt_sources"):
                    old_reason = str(merged.get("halt_reason", ""))
                    if "MT5 connection lost" in old_reason or "mt5" in old_reason.lower():
                        merged["halt_sources"] = ["mt5_disconnect"]
                    elif "consecutive" in old_reason.lower() or "cumulative" in old_reason.lower():
                        merged["halt_sources"] = ["risk_limit"]
                    else:
                        merged["halt_sources"] = ["manual"]
                    logger.info(
                        f"[RISK_GOVERNOR] Migrated old halt_reason to halt_sources: "
                        f"{merged['halt_sources']} (old_reason={old_reason})"
                    )
                # ============================================================

                return merged
        except Exception as e:
            logger.error(f"[RISK_GOVERNOR] Failed to load state: {e}")

        return default_state

    def _load_recorded_order_ids(self) -> set:
        """Load recorded order_ids from state file (persisted dedup set)."""
        if not RISK_GOVERNOR_PERSIST:
            return set()
        try:
            raw = self._state.get("recorded_order_ids", [])
            if isinstance(raw, list):
                # Clean up old entries (older than TTL)
                now = time.time()
                ttl_seconds = _DEDUP_ORDER_ID_TTL_DAYS * 86400
                cleaned = set()
                for entry in raw:
                    if isinstance(entry, str):
                        cleaned.add(entry)
                    elif isinstance(entry, dict):
                        ts = entry.get("ts", 0)
                        oid = entry.get("id", "")
                        if oid and (now - ts) < ttl_seconds:
                            cleaned.add(str(oid))
                if len(cleaned) != len(raw):
                    logger.info(
                        f"[RISK_GOVERNOR] Cleaned {len(raw) - len(cleaned)} old order_ids "
                        f"from dedup set (TTL={_DEDUP_ORDER_ID_TTL_DAYS} days)"
                    )
                return cleaned
        except Exception as e:
            logger.error(f"[RISK_GOVERNOR] Failed to load recorded_order_ids: {e}")
        return set()

    def _save_state(self) -> None:
        if not RISK_GOVERNOR_PERSIST:
            return
        try:
            # Persist recorded_order_ids with timestamps for TTL cleanup
            now = time.time()
            recorded_with_ts = []
            for oid in self._recorded_order_ids:
                recorded_with_ts.append({"id": str(oid), "ts": now})

            self._state["recorded_order_ids"] = recorded_with_ts
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[RISK_GOVERNOR] Failed to save state: {e}")

    # ==============================
    # Halt check
    # ==============================
    def is_halted(self) -> bool:
        return bool(self._state.get("halted"))

    def get_halt_reason(self) -> str:
        """Return halt reason string (for backward compatibility)."""
        sources = self._state.get("halt_sources", [])
        if sources:
            return "; ".join(sources)
        return str(self._state.get("halt_reason") or "")

    def get_halt_sources(self) -> List[str]:
        """Return list of active halt source tags."""
        sources = self._state.get("halt_sources", [])
        if isinstance(sources, list):
            return list(sources)
        return []

    def has_halt_source(self, source: str) -> bool:
        """Check if a specific halt source is active."""
        return source in self.get_halt_sources()

    def halt(self, reason: str, source: str = "risk_limit") -> None:
        """Halt new entries.

        Args:
            reason: human-readable reason string (for logging/backward compat).
            source: explicit source tag (e.g. "mt5_disconnect", "risk_limit", "manual").
                    Multiple sources can be active simultaneously.
        """
        self._state["halted"] = True
        self._state["halt_reason"] = reason

        # Add source to the list (dedup)
        sources = self.get_halt_sources()
        if source not in sources:
            sources.append(source)
        self._state["halt_sources"] = sources

        self._state["halt_timestamp"] = time.time()
        self._save_state()
        logger.warning(
            f"[RISK_GOVERNOR] HALTED new entries: reason={reason} source={source} "
            f"active_sources={sources}"
        )

    def resume(self) -> None:
        """Resume new entries (clears ALL halt sources)."""
        self._state["halted"] = False
        self._state["halt_reason"] = ""
        self._state["halt_sources"] = []
        self._state["halt_timestamp"] = None
        self._save_state()
        logger.info("[RISK_GOVERNOR] New entries resumed (all sources cleared)")

    def resume_source(self, source: str) -> bool:
        """Resume only a specific halt source.

        If other sources are still active, the bot remains halted.
        Only resumes fully if this was the last active source.

        Returns:
            True if the bot is now fully resumed (no more active sources),
            False if other halt sources are still active.
        """
        sources = self.get_halt_sources()
        if source in sources:
            sources.remove(source)
            logger.info(
                f"[RISK_GOVERNOR] Resumed source={source}, "
                f"remaining_sources={sources}"
            )
        else:
            logger.info(
                f"[RISK_GOVERNOR] resume_source({source}): source not active, "
                f"active_sources={sources}"
            )

        if not sources:
            # No more active sources -> fully resume
            self._state["halted"] = False
            self._state["halt_reason"] = ""
            self._state["halt_sources"] = []
            self._state["halt_timestamp"] = None
            self._save_state()
            logger.info("[RISK_GOVERNOR] New entries resumed (all sources cleared)")
            return True
        else:
            # Other sources still active -> stay halted
            self._state["halt_sources"] = sources
            self._state["halt_reason"] = "; ".join(sources)
            self._save_state()
            logger.warning(
                f"[RISK_GOVERNOR] Still halted by remaining sources: {sources}"
            )
            return False

    # ==============================
    # Trade close recording
    # ==============================
    def record_trade_close(
        self,
        pnl_usd: float,
        risk_amount_usd: Optional[float] = None,
        won: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Record a closed trade and update halt conditions.

        Args:
            pnl_usd: realized PnL in account currency (negative = loss).
            risk_amount_usd: the dollar amount risked on this trade.
            won: True if the trade was profitable. If None, inferred from pnl_usd.
            **kwargs: can include 'order_id' for dedup (prevents double-counting).
        """
        try:
            pnl = float(pnl_usd) if pnl_usd is not None else 0.0
            risk = float(risk_amount_usd) if risk_amount_usd is not None else None

            # Dedup by order_id if provided (via kwargs only for internal callers)
            order_id = kwargs["order_id"] if "order_id" in kwargs else None
            if order_id is not None:
                oid = str(order_id).strip()
                if oid in self._recorded_order_ids:
                    # Already counted for this order_id - skip to avoid double-counting
                    return
                self._recorded_order_ids.add(oid)
                # Periodic cleanup of old order_ids
                self._cleanup_old_order_ids()

            if won is None:
                won = pnl > 0

            r_multiple = 0.0
            if risk is not None and risk > 0:
                r_multiple = pnl / risk
            else:
                r_multiple = 1.0 if pnl < 0 else 0.0

            if won:
                self._state["consecutive_losses"] = 0
            else:
                self._state["consecutive_losses"] = int(self._state.get("consecutive_losses", 0)) + 1

            # For wins: decrease cumulative loss by the win R-multiple.
            # For losses: increase cumulative loss by the loss magnitude.
            if won:
                self._state["cumulative_loss_r"] = max(
                    0.0,
                    float(self._state.get("cumulative_loss_r", 0.0)) - abs(r_multiple),
                )
            else:
                self._state["cumulative_loss_r"] = float(
                    self._state.get("cumulative_loss_r", 0.0)
                ) + abs(r_multiple)

            self._state["daily_loss_usd"] = float(self._state.get("daily_loss_usd", 0.0)) + min(0.0, pnl)

            self._check_halt_conditions(
                consecutive_losses=int(self._state.get("consecutive_losses", 0)),
                cumulative_loss_r=float(self._state.get("cumulative_loss_r", 0.0)),
            )

            self._save_state()
        except Exception as e:
            logger.error(f"[RISK_GOVERNOR] record_trade_close error: {e}")

    def _cleanup_old_order_ids(self) -> None:
        """Remove order_ids older than TTL (30 days) to prevent unbounded growth."""
        try:
            # Emergency cleanup if the set is very large
            if len(self._recorded_order_ids) > 10000:
                logger.warning(
                    f"[RISK_GOVERNOR] Dedup set size={len(self._recorded_order_ids)} "
                    f"exceeds 10000, performing emergency cleanup"
                )
                ids_list = list(self._recorded_order_ids)
                self._recorded_order_ids = set(ids_list[-5000:])
                logger.info(
                    f"[RISK_GOVERNOR] Emergency cleanup: reduced to "
                    f"{len(self._recorded_order_ids)} order_ids"
                )
        except Exception as e:
            logger.error(f"[RISK_GOVERNOR] _cleanup_old_order_ids error: {e}")

    def _check_halt_conditions(self, consecutive_losses: int, cumulative_loss_r: float) -> None:
        if consecutive_losses >= int(MAX_CONSECUTIVE_LOSSES):
            self.halt(
                f"consecutive_losses={consecutive_losses} >= {MAX_CONSECUTIVE_LOSSES}",
                source="risk_limit",
            )
            return

        if cumulative_loss_r >= float(RISK_GOVERNOR_MAX_LOSS_R):
            self.halt(
                f"cumulative_loss_r={cumulative_loss_r:.2f} >= {RISK_GOVERNOR_MAX_LOSS_R}",
                source="risk_limit",
            )
            return

    # ==============================
    # Entry gate for main loop
    # ==============================
    def can_open_new_position(self, open_position_count: Optional[int] = None) -> tuple[bool, str]:
        if self.is_halted():
            return False, f"RiskGovernor halted: {self.get_halt_reason()}"

        if open_position_count is not None and open_position_count >= int(MAX_OPEN_TRADES):
            return False, f"RiskGovernor: max open positions {open_position_count} >= {MAX_OPEN_TRADES}"

        return True, "OK"


# Module-level singleton for convenience
_governor: Optional[RiskGovernor] = None


def get_risk_governor() -> RiskGovernor:
    global _governor
    if _governor is None:
        _governor = RiskGovernor()
    return _governor