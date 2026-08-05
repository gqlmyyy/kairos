# Trading Bot V3 - feedback/performance.py
# Performance analytics: win rate, profit factor, sharpe

from utils.logger import get_logger
from data.storage.database import get_recent_trades, save_performance
from config import FEEDBACK_BATCH_SIZE

logger = get_logger("performance")

def calculate_metrics(trades: list) -> dict:
    if not trades:
        return {"win_rate": 0, "profit_factor": 0, "avg_win": 0, "avg_loss": 0, "sharpe": 0, "total_pnl": 0, "count": 0}
    
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) < 0]
    
    total_trades = len(trades)
    win_rate = len(wins) / total_trades if total_trades else 0
    
    total_profit = sum(t.get("pnl", 0) for t in wins)
    total_loss = abs(sum(t.get("pnl", 0) for t in losses))
    profit_factor = total_profit / total_loss if total_loss else 999
    
    avg_win = total_profit / len(wins) if wins else 0
    avg_loss = total_loss / len(losses) if losses else 0
    
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    
    pnls = [t.get("pnl", 0) for t in trades]
    avg_pnl = sum(pnls) / len(pnls) if pnls else 0
    variance = sum((p - avg_pnl) ** 2 for p in pnls) / len(pnls) if pnls else 0
    std_dev = variance ** 0.5
    sharpe = (avg_pnl / std_dev) if std_dev else 0
    
    return {
        "win_rate": round(win_rate, 3),
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "sharpe": round(sharpe, 2),
        "total_pnl": round(total_pnl, 2),
        "count": total_trades
    }

def analyze_performance(symbol: str) -> dict:
    trades = get_recent_trades(symbol, FEEDBACK_BATCH_SIZE)
    metrics = calculate_metrics(trades)
    logger.info(f"Performance {symbol}: WR={metrics['win_rate']:.1%} PF={metrics['profit_factor']:.2f} Sharpe={metrics['sharpe']:.2f}")
    
    save_performance(symbol, len(trades), metrics["win_rate"],
                     metrics["profit_factor"], metrics["avg_win"],
                     metrics["avg_loss"], metrics["sharpe"])
    return metrics

