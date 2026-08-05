# Trading Bot V3 - reports/report_generator.py
# Daily/weekly performance reports

from utils.logger import get_logger
from data.storage.database import get_daily_stats, get_recent_trades
from config import SYMBOLS

logger = get_logger("reports")

def generate_daily_report() -> str:
    stats = get_daily_stats()
    pnl = stats.get("total_pnl", 0)
    emoji = chr(0x1F4C8) if pnl >= 0 else chr(0x1F4C9)
    
    winrate = 0
    if stats.get("total_trades", 0) > 0:
        winrate = (stats.get("winning_trades", 0) / stats["total_trades"]) * 100
    
    report = f"""
{emoji} <b>Daily Report</b>
========================
P&amp;L: <b>{pnl:+.2f}$</b>
Trades: <b>{stats.get('total_trades', 0)}</b>
Wins: <b>{stats.get('winning_trades', 0)}</b>
Losses: <b>{stats.get('losing_trades', 0)}</b>
Win Rate: <b>{winrate:.1f}%</b>
Consecutive Losses: <b>{stats.get('consecutive_losses', 0)}</b>
Best: <b>{stats.get('best_symbol', 'N/A')}</b>
Worst: <b>{stats.get('worst_symbol', 'N/A')}</b>
"""
    return report

def generate_weekly_report() -> str:
    lines = ["<b>Weekly Performance</b>", "========================"]
    total_pnl = 0
    total_trades = 0
    
    for symbol in SYMBOLS:
        trades = get_recent_trades(symbol, 100)
        pnl = sum(t.get("pnl", 0) for t in trades)
        wins = len([t for t in trades if t.get("pnl", 0) > 0])
        n = len(trades)
        wr = (wins / n * 100) if n else 0
        total_pnl += pnl
        total_trades += n
        lines.append(f"{symbol}: {n} trades | {pnl:+.2f}$ | WR {wr:.0f}%")
    
    lines.append(f"\nTotal: {total_trades} trades | {total_pnl:+.2f}$")
    return "\n".join(lines)

def get_best_symbols() -> list:
    results = []
    for symbol in SYMBOLS:
        trades = get_recent_trades(symbol, 50)
        if trades:
            pnl = sum(t.get("pnl", 0) for t in trades)
            results.append((symbol, pnl))
    return sorted(results, key=lambda x: x[1], reverse=True)

