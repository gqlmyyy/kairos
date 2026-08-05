#!/usr/bin/env python3
"""
Fetch market data from local SQLite database as fallback
"""

import sqlite3
from datetime import datetime, timedelta
from utils.logger import get_logger

logger = get_logger("db_market_client")

DB_PATH = "trading_bot_v3.db"

def get_candles_from_db(symbol: str, timeframe: str = "H4", count: int = 100) -> list:
    """
    Attempt to get historical candle data from the local database
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if there's a market_data or candles table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE '%candle%' OR name LIKE '%ohlc%'
        """)
        tables = cursor.fetchall()
        
        if not tables:
            logger.warning(f"No candle tables found in {DB_PATH}")
            return []
        
        table_name = tables[0]['name']
        logger.info(f"Found table: {table_name}")
        
        # Get recent candles
        query = f"""
            SELECT * FROM {table_name}
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        cursor.execute(query, (symbol, timeframe, count))
        candles = cursor.fetchall()
        
        logger.info(f"Retrieved {len(candles)} candles for {symbol} {timeframe} from DB")
        
        # Convert to dict format
        result = []
        for row in candles:
            result.append({
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
                "time": row.get("timestamp", "")
            })
        
        conn.close()
        return list(reversed(result))  # Reverse to get ascending order
        
    except Exception as e:
        logger.error(f"Database fetch error: {e}")
        return []


def check_available_symbols_in_db() -> list:
    """List all symbols available in the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT symbol FROM (
                SELECT symbol FROM trades
                UNION
                SELECT symbol FROM execution_dataset
            )
        """)
        
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        logger.info(f"Available symbols in DB: {symbols}")
        return symbols
        
    except Exception as e:
        logger.error(f"Database query error: {e}")
        return []
