from __future__ import annotations

import time
import threading

from typing import Any, Dict, Optional, List

# استيراد الإعدادات – تأكد من وجود ملف config.py
from config import POST_ENTRY_LOOP_INTERVAL_SEC

from execution.risk_management.market_regime_detector import (
    detect_market_regime,
    get_regime_settings,
)
from utils.logger import get_logger
from data.storage.database import get_execution_dataset, close_trade_db_by_order_id
from core.heartbeat import beat

from .trade_monitor import TradeMonitor
from .market_snapshot_builder import MarketSnapshotBuilder
from .position_state_machine import PositionStateMachine, PositionState
from .rules.rule_engine import RuleEngine
from .red_flags.red_flag_detector import RedFlagDetector
from .xgboost_exit_model_adapter import XGBoostExitModelAdapter
from config import ML_EXIT_ENABLED
from risk.risk_governor import get_risk_governor
from risk.r_multiple import calculate_risk_amount_usd
from .decision_fusion import DecisionFusionEngine
from .action_executor import ActionExecutor
from .event_bus import DedupEventBus
from .events import TradeClosedEvent, SLModifiedEvent
from .event_listeners.logger_listener import LoggerListener
from .event_listeners.database_listener import DatabaseListener
from .event_listeners.telegram_listener import TelegramListener
from .performance_recorder import PerformanceRecorder
from .ml_dataset_builder import MLDatasetBuilder

logger = get_logger("post_entry_manager")

# حماية وقف الخسارة الطارئ (مستقل عن النموذج)
# الوحدة: price distance (نفس وحدة SL/TP في sltp.py)
# الغرض: طبقة حماية احتياطية في حال فشل SL عند البروكر
# القيمة: 1.3× من SL الفعلي للصفقة (أوسع من SL الأساسي)
EMERGENCY_STOP_MULTIPLIER = 1.3  # أوسع من SL الأساسي
TRADE_HEALTH_EMERGENCY_THRESHOLD = 20.0

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    mt5 = None


class PostEntryManager:
    def __init__(self, loop_interval_sec: float = None) -> None:
        if loop_interval_sec is None:
            loop_interval_sec = POST_ENTRY_LOOP_INTERVAL_SEC
        self._interval = float(loop_interval_sec)

        self._trade_monitor = TradeMonitor()
        self._snapshot_builder = MarketSnapshotBuilder()
        self._state_machine = PositionStateMachine()
        self._rule_engine = RuleEngine()
        self._red_flags = RedFlagDetector()
        self._xgb_adapter = XGBoostExitModelAdapter()
        self._fusion = DecisionFusionEngine()
        self._executor = ActionExecutor()

        self._bus = DedupEventBus(ttl_sec=30.0)
        # المسجلات
        self._bus.register("SLModified", LoggerListener())
        self._bus.register("TradeClosed", LoggerListener())
        self._bus.register("TradeClosed", DatabaseListener())
        self._bus.register("TradeClosed", TelegramListener())

        self._perf_recorder = PerformanceRecorder()
        self._ml_builder = MLDatasetBuilder()

        # سجل الصفقات النشطة: ticket -> PositionState
        self._active_positions: Dict[int, PositionState] = {}
        
        # Grace period for new positions (avoid race conditions)
        # ticket -> timestamp when position was first detected
        self._new_position_grace_sec: Dict[int, float] = {}

    def _compute_risk_amount_usd(self, order_id: str, pos: Dict[str, Any], db_row: Dict[str, Any] = None) -> float:
        """Compute risk_amount_usd for a trade using the shared R-multiple module."""
        try:
            symbol = str(pos.get("symbol") or "")
            if not symbol:
                return None
            row = db_row
            if row is None:
                try:
                    from data.storage.database import get_execution_dataset
                    row = get_execution_dataset(order_id) or {}
                except Exception:
                    row = {}
            expected_entry = row.get("expected_entry")
            expected_sl = row.get("expected_sl")
            if not expected_entry or not expected_sl:
                return None
            trade_size = None
            try:
                trade_size = float(pos.get("volume") or pos.get("size") or 0)
            except Exception:
                pass
            if not trade_size or trade_size <= 0:
                try:
                    from data.storage.database import get_open_trades
                    open_trades = get_open_trades() or []
                    for t in open_trades:
                        if str(t.get("order_id", "")) == str(order_id):
                            trade_size = float(t.get("size", 0) or 0)
                            break
                except Exception:
                    pass
            if not trade_size or trade_size <= 0:
                return None
            sl_distance = abs(float(expected_entry) - float(expected_sl))
            return calculate_risk_amount_usd(sl_distance, symbol, trade_size)
        except Exception:
            return None

    def _sync_db_context(self, positions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """بناء خريطة order_id -> سجل قاعدة البيانات للصفقات المفتوحة (من MT5)."""
        ctx: Dict[str, Dict[str, Any]] = {}
        for pos in positions:
            order_id = str(pos.get("order_id") or "")
            if order_id:
                try:
                    row = get_execution_dataset(order_id)
                    if row:
                        ctx[order_id] = row
                except Exception:
                    pass
        return ctx

    def run_once(self) -> None:
        beat()  # نبض القلب من core.heartbeat

        # جلب جميع الصفقات المفتوحة من MT5 (مصدر الحقيقة)
        positions = self._trade_monitor.get_open_positions()
        if not positions:
            return

        # استخراج التذاكر الحالية من MT5
        current_tickets: set = set()
        for pos in positions:
            ticket = pos.get("order_id")
            if ticket:
                try:
                    current_tickets.add(int(ticket))
                except (ValueError, TypeError):
                    pass

        registry_tickets = set(self._active_positions.keys())
        new_tickets = current_tickets - registry_tickets
        closed_tickets = registry_tickets - current_tickets

        # معالجة الصفقات المغلقة
        for ticket in closed_tickets:
            state = self._active_positions.pop(ticket, None)
            if state:
                self._snapshot_builder.invalidate_cache(ticket)
                logger.info(f"[POST_ENTRY] Position closed ticket={ticket}")

        # تسجيل الصفقات الجديدة
        for pos in positions:
            ticket = pos.get("order_id")
            if ticket:
                try:
                    t = int(ticket)
                except (ValueError, TypeError):
                    continue
                if t in new_tickets:
                    logger.info(
                        f"[POST_ENTRY] New position detected ticket={t} symbol={pos.get('symbol')}"
                    )

        # مزامنة سياق قاعدة البيانات لكل الصفقات الحالية
        db_ctx = self._sync_db_context(positions)

        # معالجة كل صفقة
        for pos in positions:
            order_id = str(pos.get("order_id") or "")
            if not order_id:
                continue

            # -----------------------------
            # وقف الخسارة الطارئ (مستقل عن النموذج) – بناءً على SL الفعلي
            # -----------------------------
            # الغرض: طبقة حماية احتياطية في حال فشل SL عند البروكر
            # المنطق: أغلق الصفقة فقط إذا تجاوزت الخسارة SL الفعلي × EMERGENCY_STOP_MULTIPLIER
            try:
                symbol = pos.get("symbol")
                direction = pos.get("direction") or pos.get("type")
                entry_price = pos.get("entry_price") or pos.get("price_open") or pos.get("entryPrice")
                current_price = pos.get("price_current") or pos.get("price") or pos.get("price_current")
                actual_sl = pos.get("sl")  # SL الفعلي من MT5
                
                if symbol and entry_price is not None and current_price is not None and actual_sl is not None:
                    entry_price_f = float(entry_price)
                    current_price_f = float(current_price)
                    actual_sl_f = float(actual_sl)
                    
                    # حساب المسافة المالية من SL الفعلي
                    if str(direction).lower() in ["buy", "0"]:
                        sl_distance = entry_price_f - actual_sl_f  # كم يبعد SL عنEntry
                        current_loss = entry_price_f - current_price_f  # الخسارة الحالية
                    else:
                        sl_distance = actual_sl_f - entry_price_f
                        current_loss = current_price_f - entry_price_f
                    
                    # تحقق من أن SL صحيح (ليس صفر)
                    if sl_distance > 0:
                        # EMERGENCY_STOP يطلق فقط إذا تجاوزت الخسارة SL × multiplier
                        emergency_threshold = sl_distance * EMERGENCY_STOP_MULTIPLIER
                        
                        if current_loss >= emergency_threshold:
                            logger.critical(
                                f"[EMERGENCY_STOP] Closing ticket={order_id} symbol={symbol} dir={direction} "
                                f"entry={entry_price_f} current={current_price_f} sl={actual_sl_f} "
                                f"current_loss={current_loss:.5f} threshold={emergency_threshold:.5f} "
                                f"(sl_distance={sl_distance:.5f} × {EMERGENCY_STOP_MULTIPLIER})"
                            )
                            ok = self._executor.close_position(order_id)
                            if ok:
                                # Feed Risk Governor (dedup by order_id)
                                try:
                                    _risk_amt = self._compute_risk_amount_usd(order_id, pos, db_ctx.get(order_id))
                                    get_risk_governor().record_trade_close(
                                        pnl_usd=float(pos.get("profit") or 0),
                                        risk_amount_usd=_risk_amt,
                                        order_id=order_id,
                                    )
                                except Exception:
                                    pass
                                payload = {
                                    "order_id": order_id,
                                    "symbol": symbol,
                                    "direction": pos.get("direction"),
                                    "pnl": float(pos.get("profit") or 0),
                                    "exit_reason": "emergency_stop_loss",
                                    "decision_score": None,
                                    "rule_score": None,
                                    "model_score": None,
                                    "confidence": None,
                                    "entry": entry_price_f,
                                    "exit_price": current_price_f,
                                }
                                evt = TradeClosedEvent(
                                    order_id=order_id,
                                    symbol=str(symbol),
                                    direction=str(pos.get("direction") or direction),
                                    pnl=float(payload["pnl"]),
                                    exit_reason=str(payload["exit_reason"]),
                                    decision_score=payload.get("decision_score"),
                                    rule_score=payload.get("rule_score"),
                                    model_score=payload.get("model_score"),
                                    confidence=payload.get("confidence"),
                                    ts=time.time(),
                                ).to_event()
                                self._bus.publish(evt)
                                try:
                                    close_trade_db_by_order_id(order_id=order_id, pnl=float(pos.get("profit") or 0))
                                except Exception as e:
                                    logger.error(f"[EMERGENCY_STOP] close_trade_db_by_order_id failed order_id={order_id}: {e}")

                                try:
                                    ticket_int = int(order_id)
                                    self._active_positions.pop(ticket_int, None)
                                except Exception:
                                    pass

                                self._snapshot_builder.invalidate_cache(int(order_id) if str(order_id).isdigit() else order_id)
                            continue
            except Exception as e:
                import traceback
                logger.error(f"[EMERGENCY_STOP] precheck failed order_id={order_id}: {traceback.format_exc()}")


            try:
                ticket = int(order_id)
                if ticket not in self._active_positions:
                    self._active_positions[ticket] = PositionState(
                        state="NEW",
                        be_done=False,
                        partial_done=False,
                        trailing_active=False,
                        profit_locked=False,
                    )

                state = self._active_positions[ticket]

                logger.info(f"[POST_ENTRY] Monitoring ticket={ticket}")

                db_row = db_ctx.get(order_id)
                derived_state = self._state_machine.derive_state(
                    {"trade": pos}, db_row=db_row
                )
                state.state = derived_state.state
                state.be_done = derived_state.be_done

                # فحص طارئ: إذا كان trade_health (من p_win) أقل من الحد، أغلق فوراً
                try:
                    maybe_p_win = None
                    if db_row:
                        maybe_p_win = db_row.get("p_win") or db_row.get("expected_p_win")
                    if maybe_p_win is not None:
                        p_win_f = float(maybe_p_win)
                        health_score = p_win_f * 100.0
                        if health_score < TRADE_HEALTH_EMERGENCY_THRESHOLD:
                            logger.critical(
                                f"[EMERGENCY_STOP] Closing ticket={order_id} trade_health={health_score} "
                                f"(p_win={p_win_f}) threshold={TRADE_HEALTH_EMERGENCY_THRESHOLD}"
                            )
                            ok = self._executor.close_position(order_id)
                            if ok:
                                # Feed Risk Governor (dedup by order_id)
                                try:
                                    _risk_amt = self._compute_risk_amount_usd(order_id, pos, db_ctx.get(order_id))
                                    get_risk_governor().record_trade_close(
                                        pnl_usd=float(pos.get("profit") or 0),
                                        risk_amount_usd=_risk_amt,
                                        order_id=order_id,
                                    )
                                except Exception:
                                    pass
                                payload = {
                                    "order_id": order_id,
                                    "symbol": pos.get("symbol"),
                                    "direction": pos.get("direction"),
                                    "pnl": float(pos.get("profit") or 0),
                                    "exit_reason": "emergency_stop_trade_health",
                                    "decision_score": None,
                                    "rule_score": None,
                                    "model_score": None,
                                    "confidence": None,
                                    "entry": float(pos.get("entry_price") or 0),
                                    "exit_price": float(pos.get("price_current") or 0),
                                }
                                evt = TradeClosedEvent(
                                    order_id=order_id,
                                    symbol=str(payload["symbol"]),
                                    direction=str(payload["direction"]),
                                    pnl=float(payload["pnl"]),
                                    exit_reason=str(payload["exit_reason"]),
                                    decision_score=payload.get("decision_score"),
                                    rule_score=payload.get("rule_score"),
                                    model_score=payload.get("model_score"),
                                    confidence=payload.get("confidence"),
                                    ts=time.time(),
                                ).to_event()
                                self._bus.publish(evt)
                                try:
                                    close_trade_db_by_order_id(order_id=order_id, pnl=float(pos.get("profit") or 0))
                                except Exception as e:
                                    logger.error(f"[EMERGENCY_STOP] close_trade_db_by_order_id failed order_id={order_id}: {e}")
                                try:
                                    self._active_positions.pop(int(order_id), None)
                                except Exception:
                                    pass
                                continue
                except Exception as e:
                    logger.error(f"[EMERGENCY_STOP] trade_health precheck failed order_id={order_id}: {type(e).__name__}: {e}")

                # الحفاظ على القيم المتراكمة للحالات الجزئية وقفل الربح (حتى تُضاف إلى قاعدة البيانات)
                state.partial_done = state.partial_done or derived_state.partial_done
                state.trailing_active = derived_state.trailing_active
                state.profit_locked = state.profit_locked or derived_state.profit_locked

                snapshot = self._snapshot_builder.build_snapshot(pos)
                snapshot["expected_row"] = db_row or {}

                regime = detect_market_regime(
                    symbol=pos.get("symbol"),
                    atr=pos.get("atr"),
                )
                snapshot["market_regime"] = regime
                snapshot["regime_settings"] = get_regime_settings(regime)

                rule_results = self._rule_engine.evaluate(snapshot, state.state)

                # جسر: إدخال نتائج القواعد إلى RedFlagDetector
                snapshot["_rule_results_by_name"] = {rr.rule_name: rr for rr in rule_results}

                # تحديث MFE/MAE قبل RedFlagDetector
                self._xgb_adapter.update_mfe_mae(
                    position_state=state,
                    current_profit=float(pos.get("profit") or 0),
                )

                red_flags, _rf_score, _rf_meta = self._red_flags.detect(snapshot, position_state=state)

                snapshot["_red_flag_report"] = {
                    "should_consult_exit_model": _rf_meta.get("should_consult_exit_model"),
                    "flag_count": _rf_meta.get("flag_count"),
                    "severity": _rf_meta.get("severity"),
                    "triggered_flags": _rf_meta.get("triggered_flags"),
                }

                # ============================================================
                # AI/ML EXIT MODEL - DISABLED (ML_EXIT_ENABLED=False)
                # ============================================================
                # The XGBoost exit model is DISABLED until it proves
                # out-of-sample performance (AUC/accuracy clearly above chance).
                # When disabled, xgb is None so DecisionFusionEngine never
                # consults the model and never closes based on model output.
                # ============================================================
                if ML_EXIT_ENABLED:
                    xgb = self._xgb_adapter.predict(snapshot, position_state=state)
                else:
                    xgb = None
                decision = self._fusion.fuse(rule_results, red_flags, xgb, snapshot)


                if decision.decision == "ClosePosition":
                    ok = self._executor.close_position(order_id)
                    if ok:
                        # Feed Risk Governor (dedup by order_id)
                        try:
                            _risk_amt = self._compute_risk_amount_usd(order_id, pos, db_row)
                            get_risk_governor().record_trade_close(
                                pnl_usd=float(pos.get("profit") or 0),
                                risk_amount_usd=_risk_amt,
                                order_id=order_id,
                            )
                        except Exception:
                            pass
                        payload = {
                            "order_id": order_id,
                            "symbol": pos.get("symbol"),
                            "direction": pos.get("direction"),
                            "pnl": float(pos.get("profit") or 0),
                            "exit_reason": ";".join(decision.reasons) if decision.reasons else "rule_close",
                            "decision_score": decision.decision_score,
                            "rule_score": decision.rule_score,
                            "model_score": decision.model_score,
                            "confidence": decision.confidence,
                            "entry": float(pos.get("entry_price") or 0),
                            "exit_price": float(pos.get("price_current") or 0),
                        }
                        evt = TradeClosedEvent(
                            order_id=order_id,
                            symbol=str(payload["symbol"]),
                            direction=str(payload["direction"]),
                            pnl=float(payload["pnl"]),
                            exit_reason=str(payload["exit_reason"]),
                            decision_score=payload.get("decision_score"),
                            rule_score=payload.get("rule_score"),
                            model_score=payload.get("model_score"),
                            confidence=payload.get("confidence"),
                            ts=time.time(),
                        ).to_event()
                        self._bus.publish(evt)
                        try:
                            close_trade_db_by_order_id(
                                order_id=order_id,
                                pnl=float(pos.get("profit") or 0),
                            )
                        except Exception as e:
                            logger.error(f"[POST_ENTRY] close_trade_db_by_order_id failed order_id={order_id}: {e}")

                        self._perf_recorder.record_on_close(payload)
                        self._ml_builder.on_trade_closed(payload)

                        self._active_positions.pop(ticket, None)
                        self._snapshot_builder.invalidate_cache(ticket)

                elif decision.decision == "MoveSL":
                    try:
                        trade = snapshot.get("trade", {}) if isinstance(snapshot, dict) else {}

                        # جمع جميع مقترحات MoveSL من كل القاعدة في هذه الدورة
                        move_sl_proposals: List[tuple[str, Optional[float]]] = []
                        for rr in rule_results:
                            rr_name = getattr(rr, "rule_name", rr.__class__.__name__)
                            for sa in getattr(rr, "suggested_actions", []) or []:
                                if getattr(sa, "action_type", None) == "MoveSL":
                                    move_sl_proposals.append((rr_name, getattr(sa, "value", None)))

                        if not move_sl_proposals:
                            new_sl = float(trade.get("sl") or 0)
                        else:
                            # تسجيل التعارضات إن وجدت
                            proposed_values = sorted(
                                {
                                    float(v)
                                    for (_name, v) in move_sl_proposals
                                    if v is not None and str(v) != ""
                                }
                            )
                            if len(proposed_values) > 1:
                                logger.info(
                                    "[POST_ENTRY][MoveSL] conflict proposals="
                                    + ";".join(
                                        f"{name}={val}" for (name, val) in move_sl_proposals
                                    )
                                )

                        symbol = trade.get("symbol") or pos.get("symbol")
                        direction = trade.get("direction") or pos.get("direction")
                        current_sl = trade.get("sl")
                        try:
                            current_sl_val = float(current_sl) if current_sl is not None else 0.0
                        except Exception:
                            current_sl_val = 0.0

                        candidates: List[float] = []
                        for _name, v in move_sl_proposals:
                            if v is None:
                                continue
                            try:
                                fv = float(v)
                            except Exception:
                                continue

                            # تجاهل المقترحات التي تقلل الحماية (تحريك وقف الخسارة للخلف)
                            if direction in ["buy", "BUY", "Buy"]:
                                if fv > current_sl_val:
                                    candidates.append(fv)
                            elif direction in ["sell", "SELL", "Sell"]:
                                if fv < current_sl_val:
                                    candidates.append(fv)
                            else:
                                candidates.append(fv)

                        if not candidates:
                            new_sl = current_sl_val
                        else:
                            # اختيار الأكثر تحفظاً (الأقرب للسعر الحالي في الاتجاه المناسب)
                            if direction in ["buy", "BUY", "Buy"]:
                                new_sl = max(candidates)
                            else:
                                new_sl = min(candidates)

                        if new_sl is not None and new_sl > 0 and symbol and direction:
                            ok = self._executor.modify_sl(
                                order_id=order_id,
                                symbol=str(symbol),
                                direction=str(direction),
                                new_sl=float(new_sl),
                            )
                            if ok:
                                reason = "Stop Loss Modified"
                                entry_price = (
                                    trade.get("entry_price")
                                    or trade.get("entry")
                                    or pos.get("entry_price")
                                    or pos.get("price_open")
                                    or pos.get("entryPrice")
                                )
                                try:
                                    entry_price_f = float(entry_price)
                                except Exception:
                                    entry_price_f = float(pos.get("entry_price") or 0.0)

                                evt = SLModifiedEvent(
                                    ticket=str(order_id),
                                    symbol=str(symbol),
                                    direction=str(direction),
                                    old_sl=float(current_sl_val),
                                    new_sl=float(new_sl),
                                    entry_price=entry_price_f,
                                    reason=str(reason),
                                    ts=time.time(),
                                ).to_event()
                                self._bus.publish(evt)
                    except Exception:
                        pass

            except Exception as e:
                import traceback
                logger.error(f"[POST_ENTRY] per-position error ticket={order_id}: {traceback.format_exc()}")
                continue


    def loop(self) -> None:
        logger.info("PostEntryManager started")
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"PostEntryManager loop error: {e}")
            time.sleep(self._interval)


def start_post_entry_manager(loop_interval_sec: float = None) -> threading.Thread:
    mgr = PostEntryManager(loop_interval_sec=loop_interval_sec)
    t = threading.Thread(target=mgr.loop, daemon=True, name="post_entry_manager_thread")
    t.start()
    return t