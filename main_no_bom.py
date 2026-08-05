#Trading Bot V3 - main.py

# Main entry point: connects all 5 layers

import time
import threading
from datetime import datetime

from utils.logger import get_logger
from data.storage.database import init_db, save_trade, save_decision, get_daily_stats, get_open_trades
from data.news.fetcher import fetch_rss_news
from data.news.calendar import is_high_impact_soon
from data.news.scoring import filter_relevant_news
from data.market.hybrid_client import get_atr as get_atr_hybrid
from data.market.client import set_token as set_market_token, get_equity, get_indicators
from analysis.ai.deepseek import analyze_news

# Use hybrid client for data

get_atr = get_atr_hybrid
from analysis.sentiment.analyzer import analyze_sentiment
from analysis.technical.indicators import get_trend_score, get_momentum_score, get_volatility_score
from analysis.technical.regime import get_market_regime
from analysis.multi_timeframe.analyzer import get_multi_timeframe_analysis
from decision.voting_engine import make_decision
from decision.signal_engine import generate_signal
from decision.confidence_engine import calculate_confidence
from risk.risk_engine import can_trade
from risk.sltp import calculate_sl_tp
from risk.position_sizing import calculate_position_size
from execution.quantdinger_client import login, open_trade, connect_mt5, check_mt5_status
from execution.reconciliation import start_reconciliation
from execution.mt5_watchdog import start_mt5_watchdog
from feedback.adaptive_weights import run_feedback_loop
from reports.report_generator import generate_daily_report
from telegram.notifier import notify_start, notify_trade_opened, notify_trade_closed, notify_alert, notify_daily_report, send
from telegram.telegram_bot import start_telegram_bot
from config import SYMBOLS, NEWS_CHECK_INTERVAL
from analysis.features.feature_builder import build_trade_features
from data.storage.database import upsert_execution_expected
from analysis.models.xgboost_v2_inference import predict_with_v2, should_trade_v2, get_size_multiplier

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

    if not check_mt5_status():
        logger.warning("MT5 disconnected, reconnecting...")
        if connect_mt5():
            logger.info("MT5 reconnected")
        else:
            logger.warning("MT5 still disconnected")
            return

    if is_high_impact_soon(30):
        logger.warning("High impact news incoming - pausing")
        notify_alert("High impact news in <30min - cycle skipped")
        return

    news = fetch_rss_news()
    if not news:
        logger.warning("No news fetched")
        return

    equity = get_equity()
    logger.info(f"Equity: {equity:.2f}$ | News: {len(news)}")
    news = filter_relevant_news(news)

    try:
        _news_fetched = (news is not None and len(news) > 0)
    except Exception:
        _news_fetched = False

    from data.news.fetcher import filter_news_for_symbol

    for symbol in SYMBOLS:
        logger.info(f"--- {symbol} ---")

        try:
            symbol_news = filter_news_for_symbol(news, symbol)
            if not symbol_news:
                logger.info(f"{symbol}: no pair-specific news -> skipping DeepSeek analysis")
                continue

            ai = analyze_news(symbol_news, symbol)
            signal = generate_signal(ai)

            sentiment = analyze_sentiment(news, symbol)
            mtf = get_multi_timeframe_analysis(symbol)
            trend_score, trend_dir = get_trend_score(symbol)
            momentum = get_momentum_score(symbol)
            volatility_score = get_volatility_score(symbol)
            regime = get_market_regime(symbol)

            _indicators_fetched = all([
                trend_score is not None,
                trend_dir is not None,
                momentum is not None,
                volatility_score is not None,
                regime is not None,
                mtf is not None,
            ])

            _live_data = True
            _cached = False
            try:
                _fallback_like = (str(trend_dir).lower() == "neutral" and float(trend_score) == 40 and float(volatility_score) == 50)
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

            save_decision(symbol, direction, scores, ai_confidence, confidence, mtf.aligned, regime, ai.reason, "DECIDED")

            signal_is_valid = (direction != "NEUTRAL" and signal.get("is_valid") is True)
            ok_risk, risk_reason = can_trade(symbol, direction, final_score, ai_confidence, equity)

            atr = get_atr(symbol)
            entry_price = 0.0
            from data.market.hybrid_client import get_price_hybrid

            entry_price = get_price_hybrid(symbol)
            sl, tp = (None, None)
            try:
                sl, tp = calculate_sl_tp(symbol, entry_price, direction, atr)
            except Exception:
                sl, tp = (None, None)

            sl_tp_calculated = (sl is not None and tp is not None)
            sl_distance = abs(entry_price - sl) if sl_tp_calculated else 0

            position_size = calculate_position_size(equity, sl_distance, symbol, consecutive_losses=0, score=final_score)
            position_size_ok = (position_size is not None and position_size > 0)

            spread = 0.0
            indicators_data = get_indicators(symbol)
            rsi = float(indicators_data.get("rsi", 50.0))
            macd = float(indicators_data.get("macd", 0.0))

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

            p_win = v2_result["p_win"]
            model_available = v2_result["available"]

            size_multiplier = 1.0
            xgboost_valid = False
            reject_reason = None

            if not model_available:
                xgboost_valid = False
                reject_reason = "model_available=False"
            else:
                threshold = 0.60
                if not should_trade_v2(p_win, threshold=threshold):
                    xgboost_valid = False
                    reject_reason = f"p_win<threshold (p_win={p_win:.3f} threshold={threshold})"
                else:
                    size_multiplier = get_size_multiplier(p_win)
                    if size_multiplier <= 0:
                        xgboost_valid = False
                        reject_reason = "size_multiplier<=0"
                    else:
                        xgboost_valid = True

            final_decision_valid = bool(signal_is_valid and ok_risk and sl_tp_calculated and position_size_ok and xgboost_valid)

            logger.info(
                f"[VERIFY][DECISION_TRUTH] bias={ai.bias} score={final_score} confidence={confidence} vote={direction} "
                f"risk_passed={ok_risk} sl_tp_calculated={sl_tp_calculated} position_size={position_size} "
                f"xgboost_p_win={p_win} final_decision_valid={final_decision_valid} reject_reason={reject_reason or risk_reason}"
            )

            if not final_decision_valid:
                continue

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

            base_size = position_size
            position_size = round(position_size * size_multiplier, 2)

            result = open_trade(symbol, direction, position_size, sl, tp, ai.reason[:80])
            if result is not None and isinstance(result, dict) and result.get("status") == "success" and result.get("order_id"):
                order_id = str(result.get("order_id", "") or result.get("id", "") or result.get("ticket", ""))
                entry_price = float(result.get("price", 0) or 0)
                sl, tp = calculate_sl_tp(symbol, entry_price, direction, atr)

                expected_payload = {
                    "expected_rsi": features.get("rsi", None),
                    "expected_macd": features.get("macd", None),
                    "expected_session": features.get("session", None),
                    "expected_atr": features.get("atr", None),
                    "expected_trend_strength": features.get("trend_strength", None),
                    "expected_momentum_score": (features.get("momentum_score", None) * 100) if features.get("momentum_score", None) is not None else float(decision.get("momentum_score", 0) or 0),
                    "expected_volatility_score": (features.get("volatility_score", None) * 100) if features.get("volatility_score", None) is not None else float(volatility_score or 0),
                    "expected_market_regime": features.get("market_regime", None) or (
                        regime if isinstance(regime, str) else regime.get("regime", "UNKNOWN") if isinstance(regime, dict) else "UNKNOWN"
                    ),
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

                trade_id = save_trade(
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

                notify_trade_opened(symbol, direction, position_size, entry_price, sl, tp, final_score, confidence, ai.reason[:80])

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

    if not connect_mt5():
        logger.warning("MT5 connection failed at startup")
    else:
        logger.info("MT5 connected")

    start_telegram_bot(bot_state)
    start_mt5_watchdog()
    start_reconciliation()

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

