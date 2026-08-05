#!/usr/bin/env python3
"""
Check what data is available in trading_bot_v3.db
"""

import sqlite3
from datetime import datetime

DB_PATH = "trading_bot_v3.db"

def inspect_database():
    """Inspect the database structure and content"""
    
    print("\n" + "="*60)
    print("Database Inspection: trading_bot_v3.db")
    print("="*60)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. List all tables
        print("\n[1] Available Tables:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if not tables:
            print("  ⚠️  No tables found!")
            return
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  • {table_name}: {count} rows")
        
        # 2. Check for market data tables
        print("\n[2] Market Data Tables:")
        market_patterns = ['candle', 'ohlc', 'kline', 'market', 'price']
        has_market_data = False
        
        for pattern in market_patterns:
            cursor.execute(f"""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE '%{pattern}%'
            """)
            result = cursor.fetchall()
            if result:
                has_market_data = True
                for table in result:
                    print(f"  ✓ Found: {table[0]}")
        
        if not has_market_data:
            print("  ✗ No market data tables found")
        
        # 3. Check trades table
        print("\n[3] Trades Data:")
        cursor.execute("SELECT COUNT(*) FROM trades")
        trade_count = cursor.fetchone()[0]
        print(f"  • Total trades: {trade_count}")
        
        if trade_count > 0:
            cursor.execute("""
                SELECT DISTINCT symbol FROM trades
            """)
            symbols = [row[0] for row in cursor.fetchall()]
            print(f"  • Symbols: {', '.join(symbols)}")
            
            cursor.execute("""
                SELECT symbol, COUNT(*) as count FROM trades
                GROUP BY symbol
            """)
            for symbol, count in cursor.fetchall():
                print(f"    - {symbol}: {count}")
        
        # 4. Check execution_dataset table
        print("\n[4] Execution Dataset:")
        try:
            cursor.execute("SELECT COUNT(*) FROM execution_dataset")
            exec_count = cursor.fetchone()[0]
            print(f"  • Total records: {exec_count}")
            
            if exec_count > 0:
                cursor.execute("""
                    SELECT DISTINCT symbol FROM execution_dataset
                """)
                symbols = [row[0] for row in cursor.fetchall()]
                print(f"  • Symbols: {', '.join(symbols)}")
        except Exception as e:
            print(f"  ⚠️  Error reading execution_dataset: {e}")
        
        # 5. Check decisions table
        print("\n[5] Decisions Data:")
        try:
            cursor.execute("SELECT COUNT(*) FROM decisions")
            dec_count = cursor.fetchone()[0]
            print(f"  • Total decisions: {dec_count}")
        except Exception as e:
            print(f"  ⚠️  Error reading decisions: {e}")
        
        # 6. Check daily_stats table
        print("\n[6] Daily Stats:")
        try:
            cursor.execute("SELECT COUNT(*) FROM daily_stats")
            stats_count = cursor.fetchone()[0]
            print(f"  • Total stats: {stats_count}")
            
            if stats_count > 0:
                cursor.execute("""
                    SELECT date, SUM(pnl) as total_pnl, COUNT(*) as trades
                    FROM daily_stats
                    GROUP BY date
                    ORDER BY date DESC
                    LIMIT 5
                """)
                print("  • Last 5 days:")
                for date, pnl, trades in cursor.fetchall():
                    print(f"    - {date}: PnL=${pnl}, Trades={trades}")
        except Exception as e:
            print(f"  ⚠️  Error reading daily_stats: {e}")
        
        # 7. Schema details
        print("\n[7] Recent Table Schemas:")
        for table in ['trades', 'execution_dataset', 'decisions'][:1]:
            try:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                if columns:
                    print(f"\n  {table}:")
                    for col in columns:
                        col_name, col_type = col[1], col[2]
                        print(f"    • {col_name}: {col_type}")
            except:
                pass
        
        print("\n" + "="*60)
        print("Inspection Complete")
        print("="*60 + "\n")
        
        conn.close()
        
    except sqlite3.OperationalError as e:
        print(f"\n✗ Database error: {e}")
        print("  The database might be locked or corrupted")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")

if __name__ == "__main__":
    inspect_database()
