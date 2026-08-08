"""Post-entry manager - the loop that drives trade management.

Deliberately thin. It gathers facts (MT5 positions, DB rows, market readings),
builds a TradeContext, hands it to the trade_management orchestrator, and
executes whatever the orchestrator decided. It contains no decision logic of
its own — the previous generation embedded emergency-stop rules directly here,
which made them untestable and invisible to the layer ordering.

Responsibilities:
  - discover open positions and track their lifecycle
  - assemble the inputs each layer needs
  - execute close / partial-close / modify
  - publish events and persist state
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from config import POST_ENTRY_LOOP_INTERVAL_SEC
from core.heartbeat import beat
from data.storage.database import (
    close_trade_db_by_order_id,
    get_execution_dataset,
    parse_partial_levels_done,
    update_breakeven_done,
    update_execution_mfe_mae,
    update_partial_levels_done,
)
from trade_management import TradeContext, TradeManagementOrchestrator
from trade_management.layer1_intrabar import bars_since
from trade_management.layer1_risk_governor_gate import record_closed_trade
from utils.logger import get_logger

from .action_executor import ActionExecutor
from .event_bus import DedupEventBus
from .event_listeners.database_listener import DatabaseListener
from .event_listeners.logger_listener import LoggerListener
from .event_listeners.telegram_listener import TelegramListener
from .events import SLModifiedEvent, TradeClosedEvent
from .ml_dataset_builder import MLDatasetBuilder
from .performance_recorder import PerformanceRecorder
from .trade_monitor import TradeMonitor

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    mt5 = None

logger = get_logger("post_entry_manager")


class TradeRuntimeState:
    """Per-trade state the layers need but MT5 does not carry."""

    def __init__(self, initial_volume: float, profile: str, settings: Dict[str, Any]) -> None:
        self.initial_volume = initial_volume
        self.profile = profile
        self.settings = settings
        self.breakeven_done = False
        self.partial_levels_done: set = set()
        # R multiples: what the layers reason in.
        self.mfe_r = 0.0
        self.mae_r = 0.0
        # Dollars: what execution_dataset.mfe/mae have always stored, and what
        # the exit-model trainer compares against absolute thresholds.
        self.mfe_usd = 0.0
        self.mae_usd = 0.0


class PostEntryManager:
    def __init__(self, loop_interval_sec: Optional[float] = None) -> None:
        self._interval = float(
            loop_interval_sec if loop_interval_sec is not None else POST_ENTRY_LOOP_INTERVAL_SEC
        )

        self._monitor = TradeMonitor()
        self._executor = ActionExecutor()
        self._orchestrator = TradeManagementOrchestrator()

        self._bus = DedupEventBus(ttl_sec=30.0)
        self._bus.register("SLModified", LoggerListener())
        self._bus.register("TradeClosed", LoggerListener())
        self._bus.register("TradeClosed", DatabaseListener())
        self._bus.register("TradeClosed", TelegramListener())

        self._perf_recorder = PerformanceRecorder()
        self._ml_builder = MLDatasetBuilder()

        self._state: Dict[str, TradeRuntimeState] = {}

    # ------------------------------------------------------------- context
    def _symbol_meta(self, symbol: str) -> Dict[str, float]:
        """Point size and broker stop level, needed by the modify filter."""
        meta = {"point": 0.00001, "stop_level": 0.0}
        if mt5 is None:
            return meta
        try:
            info = mt5.symbol_info(symbol)
            if info is not None:
                meta["point"] = float(getattr(info, "point", 0.00001) or 0.00001)
                meta["stop_level"] = float(getattr(info, "trade_stops_level", 0) or 0)
        except Exception as exc:
            logger.warning("[POST_ENTRY] symbol_info failed for %s: %s", symbol, exc)
        return meta

    def _ensure_state(self, pos: Dict[str, Any], db_row: Dict[str, Any]) -> TradeRuntimeState:
        order_id = str(pos.get("order_id"))
        state = self._state.get(order_id)
        if state is not None:
            return state

        db_row = db_row or {}
        profile, settings = self._orchestrator.resolve_profile(
            stored_profile=db_row.get("entry_profile"),
            regime=db_row.get("expected_market_regime"),
            mtf_aligned=None,
            trend_strength=db_row.get("expected_trend_strength"),
        )

        # Original volume: prefer the DB record, fall back to what is open now.
        initial_volume = float(db_row.get("expected_volume") or pos.get("volume") or 0.0)

        state = TradeRuntimeState(initial_volume, profile, settings)
        state.breakeven_done = bool(db_row.get("breakeven_done") or 0)
        # Restore ladder progress from the database. Without this a restart
        # re-arms a level that already executed and takes the partial twice.
        state.partial_levels_done = parse_partial_levels_done(
            db_row.get("partial_levels_done")
        )
        self._state[order_id] = state
        logger.info(
            "[POST_ENTRY] registered ticket=%s symbol=%s profile=%s initial_volume=%.2f "
            "breakeven_done=%s partial_levels_done=%s",
            order_id, pos.get("symbol"), profile, initial_volume,
            state.breakeven_done, sorted(state.partial_levels_done) or "none",
        )
        return state

    def _build_context(
        self, pos: Dict[str, Any], db_row: Dict[str, Any], state: TradeRuntimeState
    ) -> TradeContext:
        symbol = str(pos.get("symbol") or "")
        meta = self._symbol_meta(symbol)

        entry_price = float(pos.get("entry_price") or 0.0)
        initial_sl = float(db_row.get("expected_sl") or 0.0) or float(pos.get("sl") or 0.0)
        r_distance = abs(entry_price - initial_sl) if initial_sl else 0.0

        atr_at_entry = float(db_row.get("expected_atr") or 0.0)
        atr_now = self._current_atr(symbol) or atr_at_entry

        return TradeContext(
            order_id=str(pos.get("order_id")),
            symbol=symbol,
            direction=str(pos.get("direction") or ""),
            entry_price=entry_price,
            current_price=float(pos.get("price_current") or 0.0),
            volume=float(pos.get("volume") or 0.0),
            initial_volume=state.initial_volume or float(pos.get("volume") or 0.0),
            sl=float(pos.get("sl") or 0.0),
            tp=pos.get("tp"),
            initial_sl=initial_sl,
            r_distance=r_distance,
            atr_now=atr_now,
            atr_at_entry=atr_at_entry,
            trend_strength=float(db_row.get("expected_trend_strength") or 50.0),
            regime=str(db_row.get("expected_market_regime") or "unknown"),
            point_size=meta["point"],
            broker_stop_level_points=meta["stop_level"],
            bars_open=bars_since(pos.get("time_open"), time.time()),
            profile=state.profile,
            mfe_r=state.mfe_r,
            mae_r=state.mae_r,
            breakeven_done=state.breakeven_done,
            partial_levels_done=tuple(sorted(state.partial_levels_done)),
            extras={"db_row": db_row},
        )

    @staticmethod
    def _current_atr(symbol: str) -> float:
        try:
            from data.market.hybrid_client import get_atr_hybrid

            return float(get_atr_hybrid(symbol, timeframe="H1") or 0.0)
        except Exception as exc:
            logger.warning("[POST_ENTRY] ATR fetch failed for %s: %s", symbol, exc)
            return 0.0

    @staticmethod
    def _market_readings(ctx: TradeContext) -> Dict[str, Any]:
        """Trend/momentum readings for the Exit Score."""
        readings: Dict[str, Any] = {}
        try:
            from data.market.hybrid_client import get_indicators_hybrid

            indicators = get_indicators_hybrid(ctx.symbol, timeframe="H1") or {}
            rsi = indicators.get("rsi")
            if rsi is not None:
                # RSI doubles as a directional momentum proxy on a 0..100 scale.
                readings["momentum_score"] = float(rsi)

            trend_map = {
                "strong uptrend": 85.0, "uptrend": 65.0, "sideways": 50.0,
                "downtrend": 35.0, "strong downtrend": 15.0,
            }
            ma_trend = str(indicators.get("ma_trend") or "").lower()
            if ma_trend in trend_map:
                readings["trend_score"] = trend_map[ma_trend]
        except Exception as exc:
            logger.warning("[POST_ENTRY] readings unavailable for %s: %s", ctx.symbol, exc)
        return readings

    def _update_excursions(
        self, ctx: TradeContext, state: TradeRuntimeState, profit_usd: float
    ) -> None:
        """Track maximum favourable/adverse excursion.

        Two units are kept on purpose:

        - ``mfe_r``/``mae_r`` (R multiples) are what the layers reason in, so
          trailing calibration is comparable across symbols and position sizes.
        - ``mfe_usd``/``mae_usd`` (dollars) are what gets persisted, because the
          execution_dataset.mfe/mae columns have always held dollars and
          scripts/train_exit_model.py compares them against absolute dollar
          thresholds (``mfe > 20``, ``mfe > 10``). Writing R into those columns
          would silently change the meaning of the training label.
        """
        profit_r = ctx.profit_r
        changed = False

        if profit_r > state.mfe_r:
            state.mfe_r = profit_r
            changed = True
        if profit_r < state.mae_r:
            state.mae_r = profit_r
            changed = True

        if profit_usd > state.mfe_usd:
            state.mfe_usd = profit_usd
            changed = True
        if profit_usd < state.mae_usd:
            state.mae_usd = profit_usd
            changed = True

        if changed:
            try:
                # Dollars, matching the historical column contract.
                update_execution_mfe_mae(ctx.order_id, state.mfe_usd, state.mae_usd)
            except Exception as exc:
                logger.warning("[POST_ENTRY] mfe/mae persist failed %s: %s", ctx.order_id, exc)

    # -------------------------------------------------------------- actions
    def _risk_amount_usd(self, ctx: TradeContext) -> Optional[float]:
        try:
            from risk.r_multiple import calculate_risk_amount_usd

            if ctx.r_distance <= 0 or ctx.volume <= 0:
                return None
            return calculate_risk_amount_usd(ctx.r_distance, ctx.symbol, ctx.volume)
        except Exception:
            return None

    def _finalise_close(self, ctx: TradeContext, pos: Dict[str, Any], reasons: List[str]) -> None:
        pnl = float(pos.get("profit") or 0.0)
        record_closed_trade(ctx.order_id, pnl, self._risk_amount_usd(ctx))

        exit_reason = ";".join(reasons) if reasons else "trade_management"
        payload = {
            "order_id": ctx.order_id,
            "symbol": ctx.symbol,
            "direction": ctx.direction,
            "pnl": pnl,
            "exit_reason": exit_reason,
            "entry": ctx.entry_price,
            "exit_price": ctx.current_price,
        }
        self._bus.publish(
            TradeClosedEvent(
                order_id=ctx.order_id,
                symbol=ctx.symbol,
                direction=ctx.direction,
                pnl=pnl,
                exit_reason=exit_reason,
                decision_score=None,
                rule_score=None,
                model_score=None,
                confidence=None,
                ts=time.time(),
            ).to_event()
        )

        try:
            close_trade_db_by_order_id(order_id=ctx.order_id, pnl=pnl)
        except Exception as exc:
            logger.error("[POST_ENTRY] close_trade_db failed %s: %s", ctx.order_id, exc)

        self._perf_recorder.record_on_close(payload)
        self._ml_builder.on_trade_closed(payload)
        self._forget(ctx.order_id)

    def _forget(self, order_id: str) -> None:
        self._state.pop(str(order_id), None)
        self._orchestrator.forget_trade(str(order_id))

    # ----------------------------------------------------------------- loop
    def run_once(self) -> None:
        beat()

        positions = self._monitor.get_open_positions()
        if not positions:
            return

        live_ids = {str(p.get("order_id")) for p in positions if p.get("order_id")}
        for gone in set(self._state) - live_ids:
            logger.info("[POST_ENTRY] position closed externally ticket=%s", gone)
            self._forget(gone)

        for pos in positions:
            order_id = str(pos.get("order_id") or "")
            if not order_id:
                continue
            try:
                self._manage_one(order_id, pos)
            except Exception:
                import traceback

                logger.error(
                    "[POST_ENTRY] per-position error ticket=%s: %s",
                    order_id, traceback.format_exc(),
                )

    def _manage_one(self, order_id: str, pos: Dict[str, Any]) -> None:
        try:
            db_row = get_execution_dataset(order_id) or {}
        except Exception as exc:
            logger.warning("[POST_ENTRY] db row unavailable %s: %s", order_id, exc)
            db_row = {}

        state = self._ensure_state(pos, db_row)
        # Build once to measure the excursion, then rebuild so the layers see
        # the updated MFE/MAE in the same pass.
        self._update_excursions(
            self._build_context(pos, db_row, state),
            state,
            profit_usd=float(pos.get("profit") or 0.0),
        )
        ctx = self._build_context(pos, db_row, state)

        outcome = self._orchestrator.manage_open_trade(
            ctx,
            settings=state.settings,
            signal=self._latest_signal(ctx.symbol),
            readings=self._market_readings(ctx),
            exit_features=None,  # supplied once the exit model is enabled
            is_new_candle=True,
        )

        if not outcome.has_action:
            return

        if outcome.close_full:
            if self._executor.close_position(order_id):
                logger.info(
                    "[POST_ENTRY] closed ticket=%s reasons=%s",
                    order_id, ";".join(outcome.reasons),
                )
                self._finalise_close(ctx, pos, outcome.reasons)
            return

        if outcome.close_fraction > 0:
            volume = ctx.initial_volume * outcome.close_fraction
            if self._executor.partial_close(order_id, volume):
                # Broker confirmed. Record the level in memory AND on disk
                # before anything else can interrupt: if the process dies
                # between the close and the write, the restart would retake it.
                if outcome.partial_level_index is not None:
                    state.partial_levels_done.add(outcome.partial_level_index)
                    persisted = update_partial_levels_done(
                        order_id, state.partial_levels_done
                    )
                    if not persisted:
                        # The close already happened at the broker; we cannot
                        # undo it. Flag loudly — a restart before the next
                        # successful write would retake this level.
                        logger.error(
                            "[POST_ENTRY] partial close EXECUTED for ticket=%s level=%s "
                            "but persisting ladder state FAILED. A restart before the "
                            "next write could repeat this level.",
                            order_id, outcome.partial_level_index,
                        )
                logger.info(
                    "[POST_ENTRY] partial close ticket=%s volume=%.2f level=%s",
                    order_id, volume, outcome.partial_level_index,
                )

        if outcome.modify is not None:
            req = outcome.modify
            if self._executor.modify_sl_tp(
                order_id=req.order_id,
                symbol=req.symbol,
                direction=req.direction,
                new_sl=req.new_sl,
                new_tp=req.new_tp,
            ):
                if any("breakeven" in r for r in req.reasons):
                    state.breakeven_done = True
                    # Persist so a restart does not re-arm break-even on a
                    # trade whose stop is already there.
                    update_breakeven_done(order_id, True)
                if req.new_sl is not None:
                    self._bus.publish(
                        SLModifiedEvent(
                            ticket=order_id,
                            symbol=req.symbol,
                            direction=req.direction,
                            old_sl=ctx.sl,
                            new_sl=float(req.new_sl),
                            entry_price=ctx.entry_price,
                            reason=";".join(req.reasons) or "trade_management",
                            ts=time.time(),
                        ).to_event()
                    )

    @staticmethod
    def _latest_signal(symbol: str) -> Optional[Dict[str, Any]]:
        """Most recent decision for this symbol, for the signal-flip check."""
        try:
            from data.storage.database import get_last_decisions

            for row in get_last_decisions(limit=20) or []:
                if str(row.get("symbol")) == symbol:
                    return {
                        "direction": row.get("direction"),
                        "final_score": row.get("final_score"),
                        "ai_confidence": row.get("ai_confidence"),
                        "mtf_aligned": bool(row.get("mtf_aligned")),
                    }
        except Exception as exc:
            logger.warning("[POST_ENTRY] signal lookup failed for %s: %s", symbol, exc)
        return None

    def loop(self) -> None:
        logger.info("PostEntryManager started (interval=%.1fs)", self._interval)
        while True:
            try:
                self.run_once()
            except Exception as exc:
                logger.error("PostEntryManager loop error: %s", exc)
            time.sleep(self._interval)


def start_post_entry_manager(loop_interval_sec: Optional[float] = None) -> threading.Thread:
    mgr = PostEntryManager(loop_interval_sec=loop_interval_sec)
    thread = threading.Thread(target=mgr.loop, daemon=True, name="post_entry_manager_thread")
    thread.start()
    return thread
