# Trading Bot V3 - feedback/learning.py
# Learning from closed trades - records results for future weight optimization

from utils.logger import get_logger
from data.storage.database import get_recent_trades, save_performance
from feedback.performance import calculate_metrics
from config import SYMBOLS, MIN_TRADES_TO_LEARN

logger = get_logger("learning")

def learn_from_history(symbol: str):
    """Analyze past trades and record what worked"""
    trades = get_recent_trades(symbol, 50)
    if len(trades) < MIN_TRADES_TO_LEARN:
        logger.info(f"{symbol}: Not enough trades ({len(trades)}) to learn")
        return None
    
    metrics = calculate_metrics(trades)
    
    # Log insights
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) < 0]
    
    if wins:
        best_scores = [t.get("final_score", 0) for t in wins]
        logger.info(f"{symbol}: Avg winning score: {sum(best_scores)/len(best_scores):.1f}")
    
    if losses:
        worst_scores = [t.get("final_score", 0) for t in losses]
        logger.info(f"{symbol}: Avg losing score: {sum(worst_scores)/len(worst_scores):.1f}")
    
    save_performance(symbol, len(trades), metrics["win_rate"],
                     metrics["profit_factor"], metrics["avg_win"],
                     metrics["avg_loss"], metrics["sharpe"])
    
    return metrics

def run_learning_cycle():
    """Run learning for all symbols"""
    for symbol in SYMBOLS:
        learn_from_history(symbol)

