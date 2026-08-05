# Trading Bot V3 - feedback/adaptive_weights.py
# Adapts decision weights based on component accuracy + symbol performance

import sqlite3
from utils.logger import get_logger
from data.storage.database import (
    DB_FILE, get_weights, save_weights, get_recent_trades,
    get_symbol_performance, get_best_worst_symbols, save_performance
)
from config import WEIGHT_LEARNING_RATE, MIN_TRADES_TO_LEARN, WEIGHT_SMOOTHING, SYMBOLS

logger = get_logger("adaptive_weights")


def get_component_accuracy(symbol: str, limit: int = 20) -> float:
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""SELECT pnl, direction FROM trades
            WHERE symbol=? AND status='closed'
            ORDER BY closed_at DESC LIMIT ?""", (symbol, limit))
        rows = c.fetchall()
        conn.close()

        if len(rows) < 3:
            return 0.5

        correct = sum(1 for row in rows if row[0] > 0)
        return correct / len(rows)
    except Exception as e:
        logger.error(f"Accuracy error: {e}")
        return 0.5


def update_weights(symbol: str):
    try:
        recent = get_recent_trades(symbol, MIN_TRADES_TO_LEARN)
        if len(recent) < MIN_TRADES_TO_LEARN:
            return

        current = get_weights(symbol)
        new_weights = current.copy()

        # تحليل الأداء الحالي
        perf = get_symbol_performance(symbol)
        win_rate = perf["win_rate"] / 100
        profit_factor = perf["profit_factor"]

        # حساب دقة كل مكون
        components = ["ai", "trend", "momentum", "sentiment"]
        accuracies = {comp: get_component_accuracy(symbol) for comp in components}

        total_accuracy = sum(accuracies.values())
        if total_accuracy > 0:
            for comp in components:
                target = accuracies[comp] / total_accuracy

                # لو الأداء ضعيف → تعديل أسرع
                if win_rate < 0.4:
                    smoothing = WEIGHT_SMOOTHING * 0.8
                elif win_rate > 0.6:
                    smoothing = WEIGHT_SMOOTHING * 1.1
                else:
                    smoothing = WEIGHT_SMOOTHING

                smoothing = max(0.5, min(0.95, smoothing))
                new_weights[comp] = (
                    smoothing * current[comp] +
                    (1 - smoothing) * target
                )

        # تطبيع الأوزان
        total = sum(new_weights.values())
        if total > 0:
            for key in new_weights:
                new_weights[key] = round(new_weights[key] / total, 3)

        save_weights(symbol, new_weights)

        # حفظ الأداء في DB
        save_performance(
            symbol=symbol,
            batch_size=len(recent),
            win_rate=perf["win_rate"],
            profit_factor=perf["profit_factor"],
            avg_win=perf["avg_win"],
            avg_loss=perf["avg_loss"],
            sharpe=0.0
        )

        logger.info(f"{symbol} weights updated: {new_weights} | WR={perf['win_rate']}% PF={perf['profit_factor']}")

    except Exception as e:
        logger.error(f"Weight update error: {e}")


def log_performance_summary():
    """طباعة ملخص أداء كل الأزواج"""
    try:
        bw = get_best_worst_symbols()
        if bw["best"]:
            b = bw["best"]
            logger.info(f"🏆 أفضل زوج: {b['symbol']} | WR={b['win_rate']}% | PnL={b['total_pnl']}$")
        if bw["worst"]:
            w = bw["worst"]
            logger.info(f"⚠️ أسوأ زوج: {w['symbol']} | WR={w['win_rate']}% | PnL={w['total_pnl']}$")
    except Exception as e:
        logger.error(f"Performance summary error: {e}")


def run_feedback_loop():
    for symbol in SYMBOLS:
        update_weights(symbol)
    log_performance_summary()