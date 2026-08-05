#Trading Bot V3 - main.py

# Main entry point: connects all 5 layers

import time
import threading
from datetime import datetime

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    mt5 = None

from utils.logger import get_logger
from data.storage.database import init_db, save_trade, save_decision, get_daily_stats, get_open_trades
from data.news.fetcher import fetch_rss_news
from data.news.calendar import is_high_impact_soon
from data.news.scoring import filter_relevant_news
from data.market.hybrid_client import get_atr as get_atr_hybrid
from data.market.client import set_token as set_market_token, get_equity

from analysis.ai.deepseek import analyze_news

# Use hybrid client for data

get_atr = get_atr_hybrid
from analysis.sentiment.analyzer import analyze_sentiment
from analysis.technical.indicators import (
    get_trend_score_from_snapshot,
    get_momentum_score_from_snapshot,
    get_volatility_score_from_snapshot,
)
from data.market.market_snapshot_builder import MarketSnapshotBuilder
from core.cycle_context import make_cycle_context

from analysis.technical.regime import get_market_regime_from_snapshot

from analysis.multi_timeframe.analyzer_snapshot import get_multi_timeframe_analysis_from_snapshot

from decision.voting_engine import make_decision
from decision.signal_engine import generate_signal
from decision.confidence_engine import calculate_confidence
from risk.risk_engine import can_trade
from risk.sltp import calculate_sl_tp, calculate_sl_tp_distances
from risk.position_sizing import calculate_position_size
from risk.risk_governor import get_risk_governor
from data.market.candle_boundary import get_last_completed_candle_time
from config import TF_DECISION
from execution.quantdinger_client import login, connect_mt5, check_mt5_status
from execution.mt5_direct import open_trade

from execution.reconciliation import start_reconciliation
from execution.mt5_watchdog import start_mt5_watchdog
from execution.post_entry.post_entry_manager import start_post_entry_manager


def startup_safety_check(mt5, max_total_wait_sec: int = 20) -> bool:
    """Startup Safety Check (run once): DB ↔ MT5 consistency before Cycle 1 / before threads.

    Fail-safe behavior:
    - If MT5 IPC connection is not healthy (e.g. last_error=(-10004, 'No IPC connection'))
      after short retries, the bot stops (returns False).

    Comparison rules:
    - MT5 positions ticket ↔ DB trades.order_id.
    - No trading logic is changed; this is only a protective gate.
    """

    from data.storage.database import get_open_trades, save_trade

    logger.info("[STARTUP_SAFETY] Starting startup safety check (run-once)")

    # -----------------------------
    # (A) Wait for healthy MT5 IPC
    # -----------------------------
    start_ts = time.time()
    attempts = 0
    positions = None

    # retry loop: e.g. 5 attempts with short waits, bounded by max_total_wait_sec
    # Fail-fast rule: if last_error is NOT the known IPC issue (-10004 / 'No IPC connection'),
    # stop early to avoid masking other failures.
    while time.time() - start_ts < max_total_wait_sec:
        attempts += 1
        last_err = None
        try:
            try:
                last_err = mt5.last_error()
            except Exception:
                last_err = None

            positions = mt5.positions_get()

            if positions is None:
                logger.warning(
                    "[STARTUP_SAFETY] mt5.positions_get returned None. "
                    f"attempt={attempts} last_error={last_err}"
                )

            # requirement: specifically handle (-10004, 'No IPC connection').
            # If it's other error codes/messages, still allow bounded retries,
            # because some MT5 terminals return transient empty/None states during startup.
            # We will fail-safe only after max_total_wait_sec.
            try:
                if last_err is not None:
                    code = last_err[0]
                    msg = last_err[1] if len(last_err) > 1 else None
                    logger.warning(
                        "[STARTUP_SAFETY] mt5.positions_get None diagnostic last_error=%s"
                        % (last_err,)
                    )
                    # No special-case fail-fast here; bounded retry until timeout.
            except Exception:
                pass


            else:
                break

        except Exception as e:
            # exception while calling positions_get
            try:
                last_err = mt5.last_error()
            except Exception:
                last_err = None
            logger.warning(
                "[STARTUP_SAFETY] mt5.positions_get exception during wait. "
                f"attempt={attempts} err={e} last_error={last_err}"
            )

            # Fail-fast: only retry if last_error matches known IPC code
            try:
                if last_err is not None:
                    code = last_err[0]
                    if int(code) != -10004:
                        logger.error(
                            "[STARTUP_SAFETY] FAIL-SAFE: MT5 exception with non-IPC last_error="
                            f"{last_err}. Stopping." 
                        )
                        return False
            except Exception:
                pass

        # short backoff
        time.sleep(2)


    if positions is None:
        try:
            last_err = mt5.last_error()
        except Exception:
            last_err = None

        logger.error(
            "[STARTUP_SAFETY] FAIL-SAFE: MT5 IPC not healthy after retries. "
            f"positions_get=None attempts={attempts} last_error={last_err}. "
            "Stopping bot to avoid duplicate/unsafe trading."
        )
        return False

    # -----------------------------
    # (B) Print all MT5 open positions
    # -----------------------------
    mt5_positions = list(positions) if positions else []
    logger.info(f"[STARTUP_SAFETY] MT5 open positions count={len(mt5_positions)}")

    for p in mt5_positions:
        d = p._asdict() if hasattr(p, "_asdict") else getattr(p, "__dict__", {})
        ticket = d.get("ticket")
        symbol = d.get("symbol") or ""
        volume = d.get("volume") or 0

        # open_time may be in 'time' from MT5 namedtuple, fallback to time_open
        open_time = d.get("time_open") or d.get("time")

        logger.info(
            "[STARTUP_SAFETY][MT5] "
            f"ticket={ticket} symbol={symbol} volume={volume} open_time={open_time}"
        )

    # -----------------------------
    # (C) Load open trades from DB
    # -----------------------------
    db_open_trades = get_open_trades() or []
    logger.info(f"[STARTUP_SAFETY] DB open trades count={len(db_open_trades)}")

    # Build lookup sets by normalized types (avoid type mismatch race)
    # NOTE (per requirement #1): normalize both sides explicitly to str to prevent == / in mismatch.
    mt5_ticket_str_set = set()
    mt5_by_ticket = {}

    for p in mt5_positions:
        d = p._asdict() if hasattr(p, "_asdict") else getattr(p, "__dict__", {})
        ticket = d.get("ticket")
        if ticket is None:
            continue
        ticket_str = str(ticket)  # normalize
        mt5_ticket_str_set.add(ticket_str)
        mt5_by_ticket[ticket_str] = d

    db_order_id_str_set = set()
    db_open_by_order_id = {}
    for row in db_open_trades:
        order_id = row.get("order_id")
        if order_id is None:
            continue
        order_id_str = str(order_id)  # normalize
        db_order_id_str_set.add(order_id_str)
        db_open_by_order_id[order_id_str] = row

    # -----------------------------
    # (D) Compare and act without guessing
    # -----------------------------
    # 1) MT5 has position, DB does not -> import into DB
    missing_in_db = mt5_ticket_str_set - db_order_id_str_set
    # 2) DB has open trade, MT5 does not -> orphan warn only
    missing_in_mt5 = db_order_id_str_set - mt5_ticket_str_set

    logger.info(
        f"[STARTUP_SAFETY] missing_in_db(MT5-only)={len(missing_in_db)} "
        f"missing_in_mt5(DB-only)={len(missing_in_mt5)}"
    )

    for ticket_str in missing_in_db:
        d = mt5_by_ticket.get(ticket_str) or {}

        symbol = d.get("symbol")
        volume = d.get("volume") or 0
        price_open = d.get("price_open") or d.get("price") or 0

        # Convert MT5 type → direction ('buy'/'sell') defensively
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

        sl = d.get("sl")
        tp = d.get("tp")
        atr = None

        # NOTE (per requirement #2): save_trade() requires many args; we fill what we can.
        # If MT5 doesn't provide required SL/TP/ATR, we pass None/0 safely.
        # save_trade signature:
        # save_trade(symbol, direction, size, entry_price, sl, tp, atr,
        #             final_score, ai_score, ai_confidence, reason, order_id="", ...)
        # We set defensive placeholders for scores/confidence/final_score/reason.
        try:
            # Use safe defaults to avoid DB INSERT failures from None.
            # (We log only; we do not change any trading rules.)
            safe_size = float(volume) if volume is not None else 0.0
            safe_entry = float(price_open) if price_open is not None else 0.0
            safe_sl = float(sl) if sl is not None else 0.0
            safe_tp = float(tp) if tp is not None else 0.0
            save_trade(
                symbol=symbol,
                direction=direction,
                size=safe_size,
                entry_price=safe_entry,
                sl=safe_sl,
                tp=safe_tp,
                atr=atr,
                final_score=0.0,
                ai_score=0.0,
                ai_confidence=0.0,
                reason="STARTUP_SAFETY_IMPORT_MT5_ONLY",
                order_id=str(ticket_str),
            )
            logger.info(
                "[STARTUP_SAFETY] Imported MT5-only position into DB "
                f"ticket={ticket_str} symbol={symbol} direction={direction}"
            )
        except Exception as e:
            logger.error(
                "[STARTUP_SAFETY] FAIL-SAFE: could not import MT5-only position into DB "
                f"ticket={ticket_str} err={e}. Stopping bot to avoid duplicate trading."
            )
            return False

    # 2) DB has open trade but MT5 doesn't -> warn only
    for order_id_str in missing_in_mt5:
        row = db_open_by_order_id.get(order_id_str) or {}
        logger.warning(
            "[STARTUP_SAFETY] DB-only orphan possible (not imported into MT5): "
            f"order_id={order_id_str} symbol={row.get('symbol')} direction={row.get('direction')} "
            "(No deletion performed in startup safety check.)"
        )

    # 3) Both have it -> confirm
    synced = mt5_ticket_str_set & db_order_id_str_set
    logger.info(
        f"[STARTUP_SAFETY] Synced open positions count={len(synced)}"
    )

    logger.info("[STARTUP_SAFETY] Completed startup safety check successfully")
    return True


from feedback.adaptive_weights import run_feedback_loop
from reports.report_generator import generate_daily_report
from telegram.notifier import notify_start, notify_trade_opened, notify_trade_closed, notify_alert, notify_daily_report, send
from telegram.telegram_bot import start_telegram_bot
from config import SYMBOLS, NEWS_CHECK_INTERVAL
from analysis.features.feature_builder import build_trade_features
from data.storage.database import upsert_execution_expected
from analysis.models.xgboost_v2_inference import predict_with_v2, should_trade_v2, get_size_multiplier
from config import ENTRY_MODEL_VERSION



from analysis.models.system_orchestrator import start_daily_orchestrator_thread


logger = get_logger("main")

bot_state = {
    "trading_paused": False,
    "pause_until": None,
    "cycle_count": 0,
    "last_cycle": None,
    "start_time": datetime.now(),
}


def run_cycle():
    logger.info("=" * 50)
    bot_state["cycle_count"] += 1
    bot_state["last_cycle"] = datetime.now().strftime("%H:%M:%S")
    logger.info(f"Cycle {bot_state['cycle_count']} at {bot_state['last_cycle']}")

    if bot_state["trading_paused"]:
        if bot_state["pause_until"] and time.time() > bot_state["pause_until"]:
            bot_state["trading_paused"] = False
            bot_state["pause_until"] = None
            send("Pause ended - trading resumed")
            logger.info("Pause ended")
        else:
            logger.info("Trading paused, skipping")
            return

    # MT5 check
    if not check_mt5_status():
        logger.warning("MT5 disconnected, reconnecting...")
        if connect_mt5():
            logger.info("MT5 reconnected")
        else:
            logger.warning("MT5 still disconnected")
            return

    # ============================================================
    # Risk Governor - Independent Halt Check (Feature 3)
    # ============================================================
    # The Risk Governor is COMPLETELY SEPARATE from trade management.
    # It ONLY blocks NEW entries. It NEVER stops management/protection
    # of already-open positions (that's handled by PostEntryManager).
    # Halt state is persisted across bot restarts.
    # ============================================================
    governor = get_risk_governor()
    if governor.is_halted():
        halt_reason = governor.get_halt_reason()
        halt_sources = governor.get_halt_sources()
        # Auto-resume ONLY the mt5_disconnect source (does not clear other sources)
        if "mt5_disconnect" in halt_sources:
            logger.info("[RISK_GOVERNOR] MT5 connection restored - resuming mt5_disconnect source only")
            governor.resume_source("mt5_disconnect")
            # If other sources are still active, stay halted
            if governor.is_halted():
                logger.warning(
                    f"[RISK_GOVERNOR] Still halted by other sources: {governor.get_halt_sources()}"
                )
                return
        else:
            logger.warning(f"[RISK_GOVERNOR] New entries halted: {halt_reason} sources={halt_sources}")
            return

    # ============================================================
    # Intrabar Management (Feature 5)
    # ============================================================
    # Only generate NEW entry signals after a new candle has actually closed.
    # Between candle closes, we reuse the last cached decision for managing
    # open positions only (handled by PostEntryManager which runs continuously).
    #
    # Uses the ACTUAL last completed candle time from MT5 (broker server time),
    # NOT wall-clock. This correctly handles broker timezone offsets, weekend
    # gaps, and any timeframe.
    # ============================================================
    _last_candle_ts = bot_state.get("_last_candle_ts", 0)
    # Use the first symbol's last completed candle as the boundary reference
    current_candle_ts = get_last_completed_candle_time(SYMBOLS[0], timeframe=TF_DECISION)
    if current_candle_ts is None:
        # ============================================================
        # MT5 UNAVAILABLE - BLOCK NEW ENTRIES (Gap 3 fix)
        # ============================================================
        # Do NOT silently fall back to wall-clock. Log a clear warning
        # and block new entries until MT5 is restored.
        # Open position management (SL/TP/Trailing) continues in
        # PostEntryManager which runs independently.
        # ============================================================
        logger.error(
            "[INTRABAR] MT5 connection lost - cannot determine candle boundary. "
            "BLOCKING new entries until MT5 is restored. "
            "Open position management continues in PostEntryManager."
        )
        # Use Risk Governor halt mechanism to block new entries
        try:
            governor.halt("MT5 connection lost - candle boundary unavailable", source="mt5_disconnect")
        except Exception:
            pass
        return
    if current_candle_ts == _last_candle_ts:
        logger.info("[INTRABAR] Same candle - skipping new entry signals (management continues in PostEntryManager)")
        return
    bot_state["_last_candle_ts"] = current_candle_ts

    # ============================================================
    # فحص مزدوج للصفقات المفتوحة (MT5 + قاعدة البيانات)
    # يمنع أي تحليل أو فتح صفقات جديدة إذا كانت هناك صفقات مفتوحة
    # ============================================================
    open_positions_found = False

    # فحص MT5
    if mt5 is not None:
        try:
            mt5_positions = mt5.positions_get()
            if mt5_positions and len(mt5_positions) > 0:
                open_positions_found = True
                logger.info(f"[GUARD] Found {len(mt5_positions)} open position(s) on MT5.")
        except Exception as e:
            logger.warning(f"[GUARD] Failed to check MT5 positions: {e}")

    # فحص قاعدة البيانات
    try:
        from data.storage.database import get_open_trades
        db_open_trades = get_open_trades()
        if db_open_trades and len(db_open_trades) > 0:
            open_positions_found = True
            logger.info(f"[GUARD] Found {len(db_open_trades)} open trade(s) in database.")
    except Exception as e:
        logger.warning(f"[GUARD] Failed to check DB open trades: {e}")

    # إذا وُجدت أي صفقة مفتوحة، تخطى الدورة بالكامل
    if open_positions_found:
        logger.info("[GUARD] Open positions detected. Skipping all trading this cycle.")
        return
    # ============================================================
    
    # High impact news check
    if is_high_impact_soon(30):
        logger.warning("High impact news incoming - pausing")
        notify_alert("High impact news in <30min - cycle skipped")
        return

    # Fetch data
    news = fetch_rss_news()
    if not news:
        logger.warning("No news fetched")
        return

    equity = get_equity()
    logger.info(f"Equity: {equity:.2f}$ | News: {len(news)}")
    news = filter_relevant_news(news)

    try:
        _news_fetched = news is not None and len(news) > 0
    except Exception:
        _news_fetched = False

    # Build one shared immutable market snapshot for this cycle
    snapshot_builder = MarketSnapshotBuilder()
    snapshot = snapshot_builder.build(SYMBOLS)
    cycle_context = make_cycle_context(snapshot=snapshot, cycle_id=bot_state["cycle_count"])

    from data.news.fetcher import filter_news_for_symbol
    for symbol in SYMBOLS:
        logger.info(f"--- {symbol} ---")



        try:
            symbol_news = filter_news_for_symbol(news, symbol)
            if not symbol_news:
                logger.info(f"{symbol}: no pair-specific news -> skipping DeepSeek analysis")
                continue

            ai = analyze_news(symbol_news, symbol, cycle_context.snapshot)
            signal = generate_signal(ai)

            sentiment = analyze_sentiment(news, symbol)

            # MTF/regime/features must also become snapshot-only in a later step.
            mtf = get_multi_timeframe_analysis_from_snapshot(
                cycle_context.snapshot, symbol
            )

            # trend/momentum/volatility are already derived from snapshot-only MTF inputs
            trend_score, trend_dir = mtf.h4_score, mtf.h4_direction
            momentum = (mtf.h1_score, mtf.h1_direction)
            volatility_score = get_volatility_score_from_snapshot(
                cycle_context.snapshot, symbol
            )


            regime = get_market_regime_from_snapshot(
                cycle_context.snapshot, symbol
            )

            _indicators_fetched = all([
                trend_score is not None,
                trend_dir is not None,
                momentum is not None,
                volatility_score is not None,
                regime is not None,
                mtf is not None,
            ])

            # runtime-derived only (no extra flags/variables)
            _live_data = True
            _cached = False
            try:
                _fallback_like = (
                    str(trend_dir).lower() == "neutral"
                    and float(trend_score) == 40
                    and float(volatility_score) == 50
                )
                if _fallback_like:
                    _live_data = False
            except Exception:
                pass

            logger.info(
                f"[VERIFY][DATA_TRUTH] symbol={symbol} timeframe=H4/H1/M15 "
                f"news_fetched={_news_fetched} indicators_fetched={_indicators_fetched} "
                f"live_data={_live_data} cached={_cached}"
            )

            sent_score_val = sentiment.score if sentiment.direction != "neutral" else 40

            decision = make_decision(
                symbol=symbol,
                ai_analysis=ai.__dict__,
                trend_data={"h4_score": trend_score, "h4_direction": trend_dir},
                momentum_data=momentum,
                volatility_score=volatility_score,
                sentiment_score_val=sent_score_val,
                mtf_data=mtf.__dict__,
            )

            final_score = decision["final_score"]
            direction = decision["direction"]
            ai_confidence = decision["ai_confidence"]

            scores = {
                "final": final_score,
                "ai": decision["ai_score"],
                "trend": decision["trend_score"],
                "momentum": decision["momentum_score"],
                "sentiment": decision["sentiment_score"],
                "volatility": decision["volatility_score"],
            }

            confidence = calculate_confidence(
                ai_confidence=ai_confidence,
                mtf_aligned=mtf.aligned,
                trend_direction=trend_dir,
                ai_bias=ai.bias,
                regime=regime,
            )

            save_decision(
                symbol,
                direction,
                scores,
                ai_confidence,
                confidence,
                mtf.aligned,
                regime,
                ai.reason,
                "DECIDED",
            )

            # Signal validity check
            signal_is_valid = (direction != "NEUTRAL" and signal.get("is_valid") is True)
            risk_passed, risk_reason = can_trade(symbol, direction, final_score, ai_confidence, equity)

            atr = get_atr(symbol)

            # ============================================================
            # NEW: Calculate SL/TP as DISTANCES (not absolute prices)
            # This eliminates price discrepancy between QuantDinger and MT5
            # ============================================================
            try:
                sl_distance, tp_distance = calculate_sl_tp_distances(
                    symbol, atr, regime, account_equity=equity
                )
                sl_tp_calculated = sl_distance is not None and tp_distance is not None
            except Exception as e:
                logger.error(f"[SLTP] Failed to calculate SL/TP distances for {symbol}: {e}")
                continue
            # Use real consecutive losses from daily stats so risk-based sizing reacts to losing streaks
            stats_for_sizing = get_daily_stats()
            consecutive_losses = stats_for_sizing.get("consecutive_losses", 0)

            position_size = calculate_position_size(
                equity,
                sl_distance,
                symbol,
                consecutive_losses=consecutive_losses,
                score=final_score,
            )


            # XGBoost gate
            spread = 0.0
            # XGBoost inputs from snapshot only (no direct market fetch)
            snapshot_h1 = cycle_context.snapshot.get(symbol, "H1") or {}
            rsi = float(snapshot_h1.get("rsi", 50.0))
            macd = float(snapshot_h1.get("macd", 0.0))


            if ENTRY_MODEL_VERSION == "v2":
                from analysis.entry_v2.inference import predict_with_entry_v2

                v2_result = predict_with_entry_v2(
                    rsi=rsi,
                    atr=atr,
                    macd=macd,
                    trend_strength=mtf.strength if isinstance(mtf.strength, (int, float)) else 0.0,
                    trend_score=trend_score,
                    momentum_score=momentum[0] if isinstance(momentum, tuple) else momentum,
                    volatility_score=volatility_score,
                    market_regime=regime.get("regime", "UNKNOWN") if isinstance(regime, dict) else str(regime),
                    direction=direction,
                )
            else:
                v2_result = predict_with_v2(
                    rsi=rsi,
                    atr=atr,
                    macd=macd,
                    trend_strength=mtf.strength if isinstance(mtf.strength, (int, float)) else 0.0,
                    trend_score=trend_score,
                    momentum_score=momentum[0] if isinstance(momentum, tuple) else momentum,
                    volatility_score=volatility_score,
                    market_regime=regime.get("regime", "UNKNOWN") if isinstance(regime, dict) else str(regime),
                    direction=direction,
                )

            xgboost_p_win = v2_result["p_win"]
            model_available = v2_result["available"]


            reject_reason = None
            final_decision_valid = False

            if not model_available:
                reject_reason = "model_available=False"
                final_decision_valid = False
            else:
                threshold = None
                if ENTRY_MODEL_VERSION == "v2":
                    try:
                        from analysis.entry_v2.inference import get_entry_threshold

                        threshold = get_entry_threshold()
                    except Exception:
                        threshold = None

                if threshold is None:
                    threshold = 0.60

                if not should_trade_v2(xgboost_p_win, threshold=threshold):
                    reject_reason = f"p_win<threshold (p_win={xgboost_p_win:.3f} threshold={threshold})"
                    final_decision_valid = False

                else:
                    size_multiplier = get_size_multiplier(xgboost_p_win)
                    if size_multiplier <= 0:
                        reject_reason = "size_multiplier<=0"
                        final_decision_valid = False
                    else:
                        # position_size adjusted by multiplier
                        final_decision_valid = bool(
                            signal_is_valid
                            and risk_passed
                            and sl_tp_calculated
                            and position_size is not None
                            and position_size > 0
                            and size_multiplier > 0
                        )

            logger.info(
                f"[VERIFY][DECISION_TRUTH] bias={ai.bias} score={final_score} confidence={confidence} "
                f"vote={direction} risk_passed={risk_passed} sl_tp_calculated={sl_tp_calculated} "
                f"position_size={position_size} xgboost_p_win={xgboost_p_win} "
                f"final_decision_valid={final_decision_valid} reject_reason={reject_reason or risk_reason}"
            )

            if not final_decision_valid:
                continue

            # unchanged rest of flow
            features = build_trade_features(
                symbol=symbol,
                market_data={"atr": atr, "spread": spread},
                indicators={
                    "rsi": rsi,
                    "atr": atr,
                    "macd": None,
                    "trend_strength": mtf.strength,
                    "momentum_score": decision.get("momentum_score", None),
                    "volatility_score": decision.get("volatility_score", None),
                },
                ai_analysis={
                    "impact_score": ai.get("impact_score", getattr(ai, "impact_score", None)),
                    "news_impact_score": ai.get("news_impact_score", None) if isinstance(ai, dict) else None,
                    "ai_score": decision.get("ai_score", None),
                    "confidence": ai_confidence,
                    "final_score": final_score,
                    "direction": direction,
                } if isinstance(ai, dict) else {
                    "impact_score": getattr(ai, "impact_score", None),
                    "news_impact_score": getattr(ai, "news_impact_score", None),
                    "ai_score": decision.get("ai_score", None),
                    "confidence": ai_confidence,
                    "final_score": final_score,
                    "direction": direction,
                },
                sentiment={"sentiment_score": decision.get("sentiment_score", sent_score_val)},
                regime={
                    "market_regime": regime.get("market_regime", regime.get("regime", None)) if isinstance(regime, dict) else None,
                    "trend_strength": mtf.strength,
                } if isinstance(regime, dict) else {"market_regime": None, "trend_strength": mtf.strength},
                mtf_data={
                    "aligned": mtf.aligned,
                    "strength": mtf.strength,
                    "momentum_score": decision.get("momentum_score", None),
                    "volatility_score": decision.get("volatility_score", None),
                },
            )

            # compute multiplier again using same p_win & logic (no threshold change)
            size_multiplier = get_size_multiplier(xgboost_p_win)
            base_size = position_size
            position_size = round(position_size * size_multiplier, 2)

            # ============================================================
            # Send distances to open_trade - it will calculate final SL/TP
            # from live MT5 price to eliminate price discrepancy
            # ============================================================
            result = open_trade(symbol, direction, position_size, sl_distance, tp_distance, ai.reason[:80])
            logger.info(f"[VERIFY][EXECUTION_RESULT] symbol={symbol} open_trade_result={result}")

            if result is not None and isinstance(result, dict) and result.get("status") == "success" and result.get("order_id"):

                order_id = str(result.get("order_id", "") or result.get("id", "") or result.get("ticket", ""))
                entry_price = float(result.get("price", 0) or 0)
                
                # Recalculate SL/TP based on actual execution price for DB storage
                # This ensures DB records match what was actually sent to MT5
                sl, tp = calculate_sl_tp(symbol, entry_price, direction, atr, account_equity=equity)

                expected_payload = {
                    "expected_rsi": features.get("rsi", None),
                    "expected_macd": features.get("macd", None),
                    "expected_session": features.get("session", None),
                    "expected_atr": features.get("atr", None),
                    "expected_trend_strength": features.get("trend_strength", None),
                    "expected_momentum_score": (features.get("momentum_score", None) * 100) if features.get("momentum_score", None) is not None else float(decision.get("momentum_score", 0) or 0),
                    "expected_volatility_score": (features.get("volatility_score", None) * 100) if features.get("volatility_score", None) is not None else float(volatility_score or 0),
                    "expected_market_regime": features.get("market_regime", None)
                    or (regime if isinstance(regime, str) else regime.get("regime", "UNKNOWN") if isinstance(regime, dict) else "UNKNOWN"),
                    "expected_ai_score": (features.get("ai_score", None) * 100) if features.get("ai_score", None) is not None else float(decision.get("ai_score", 0) or 0),
                    "expected_ai_confidence": features.get("expected_confidence", ai_confidence),
                    "expected_final_score": features.get("expected_final_score", final_score),
                    "expected_trend_score": (features.get("trend_strength", None) * 100) if features.get("trend_strength", None) is not None else float(trend_score or 0),
                    "expected_sentiment_score": (features.get("sentiment_score", None) * 100) if features.get("sentiment_score", None) is not None else float(sent_score_val or 0),
                    "expected_news_impact_score": features.get("news_impact_score", None),
                    "expected_entry": entry_price,
                    "expected_spread": features.get("spread", None),
                    "expected_final_score": features.get("expected_final_score", final_score),
                }

                required_non_null = [
                    "expected_rsi",
                    "expected_atr",
                    "expected_trend_strength",
                    "expected_momentum_score",
                    "expected_volatility_score",
                    "expected_market_regime",
                    "expected_ai_score",
                    "expected_ai_confidence",
                    "expected_final_score",
                    "expected_trend_score",
                    "expected_sentiment_score",
                    "expected_entry",
                ]

                missing = [k for k in required_non_null if expected_payload.get(k) is None]
                if missing:
                    continue

                upsert_execution_expected(
                    order_id=order_id,
                    symbol=symbol,
                    direction=direction,
                    expected_entry=expected_payload["expected_entry"],
                    expected_final_score=expected_payload["expected_final_score"],
                    expected_ai_score=expected_payload["expected_ai_score"],
                    expected_ai_confidence=expected_payload["expected_ai_confidence"],
                    expected_trend_score=expected_payload["expected_trend_score"],
                    expected_momentum_score=expected_payload["expected_momentum_score"],
                    expected_sentiment_score=expected_payload["expected_sentiment_score"],
                    expected_volatility_score=expected_payload["expected_volatility_score"],

                    expected_rsi=expected_payload["expected_rsi"],
                    expected_macd=expected_payload["expected_macd"],
                    expected_session=expected_payload["expected_session"],
                    expected_spread=expected_payload["expected_spread"],
                    expected_atr=expected_payload["expected_atr"],
                    expected_trend_strength=expected_payload["expected_trend_strength"],
                    expected_market_regime=expected_payload["expected_market_regime"],
                    expected_news_impact_score=expected_payload["expected_news_impact_score"],

                    expected_indicators_json=None,
                    strategy="V3",
                )

                save_trade(
                    symbol,
                    direction,
                    position_size,
                    entry_price,
                    sl,
                    tp,
                    atr,
                    final_score,
                    decision["ai_score"],
                    ai_confidence,
                    ai.reason[:100],
                    order_id=order_id,
                )

                notify_trade_opened(
                    symbol,
                    direction,
                    position_size,
                    entry_price,
                    sl,
                    tp,
                    final_score,
                    confidence,
                    ai.reason[:80],
                )

        except Exception:
            continue

    try:
        run_feedback_loop()
    except Exception:
        pass


def main():
    logger.info("Trading Bot V3 starting...")

    init_db()
    token = login()
    if not token:
        logger.error("Failed to login to QuantDinger")
        return

    set_market_token(token)

    # Explicitly initialize MT5 IPC early (before startup_safety_check).
    # NOTE: this is the missing step responsible for positions_get() returning None.
    if mt5 is None:
        logger.error("[STARTUP_SAFETY] mt5 module is None. Stopping bot.")
        return

    try:
        if not mt5.terminal_info():
            if not mt5.initialize():
                logger.error(f"[STARTUP_SAFETY] mt5.initialize() failed last_error={mt5.last_error()}")
                return
    except Exception as e:
        try:
            le = mt5.last_error()
        except Exception:
            le = None
        logger.error(f"[STARTUP_SAFETY] mt5.initialize() exception err={e} last_error={le}. Stopping bot.")
        return

    try:
        if mt5.terminal_info() is None:
            logger.error("[STARTUP_SAFETY] mt5.terminal_info() is None after initialize. Stopping bot.")
            return
    except Exception:
        pass

    if not connect_mt5():
        logger.warning("MT5 connection failed at startup")
    else:
        logger.info("MT5 connected")

    # Startup safety check (run once) BEFORE any thread and BEFORE Cycle 1
    ok = startup_safety_check(mt5)
    if not ok:
        logger.error("[STARTUP_SAFETY] Startup safety check failed. Stopping bot.")
        return


    # Start daemon threads
    start_telegram_bot(bot_state)
    start_mt5_watchdog()
    start_reconciliation()


    # Post-entry manager replaces all post-entry trade decision logic.
    from config import POST_ENTRY_LOOP_INTERVAL_SEC
    start_post_entry_manager(loop_interval_sec=POST_ENTRY_LOOP_INTERVAL_SEC)



    notify_start()

    while True:
        try:
            run_cycle()

            if bot_state["cycle_count"] % 48 == 0:
                notify_daily_report(get_daily_stats())

            logger.info(f"Sleeping {NEWS_CHECK_INTERVAL}s...")
            time.sleep(NEWS_CHECK_INTERVAL)

        except KeyboardInterrupt:
            send("Bot stopped by user")
            break

        except Exception as e:
            logger.error(f"Main loop: {e}")
            notify_alert(f"Main loop error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()

