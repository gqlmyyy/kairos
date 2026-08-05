#!/usr/bin/env python3
"""
Inspect execution_dataset schema and sample data
"""

import sqlite3
import json

DB_PATH = "trading_bot_v3.db"

def inspect_execution_dataset():
    """Examine the execution_dataset table in detail"""
    
    print("\n" + "="*70)
    print("Execution Dataset Inspection")
    print("="*70)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 1. Schema
        print("\n[1] Table Schema:")
        cursor.execute("PRAGMA table_info(execution_dataset)")
        columns = cursor.fetchall()
        
        for col in columns:
            col_name, col_type = col[1], col[2]
            print(f"  • {col_name:30} {col_type}")
        
        # 2. Statistics by Symbol
        print("\n[2] Data Count by Symbol:")
        cursor.execute("""
            SELECT symbol, COUNT(*) as count FROM execution_dataset
            GROUP BY symbol
            ORDER BY count DESC
        """)
        for row in cursor.fetchall():
            print(f"  • {row['symbol']:10} {row['count']:5} records")
        
        # 3. Sample Data
        print("\n[3] Sample Record (Latest):")
        cursor.execute("""
            SELECT * FROM execution_dataset
            ORDER BY dataset_updated_at DESC
            LIMIT 1
        """)
        sample = cursor.fetchone()
        
        if sample:
            print(f"  • Order ID: {sample['order_id']}")
            print(f"  • Symbol: {sample['symbol']}")
            print(f"  • Expected Entry: {sample['expected_entry']}")
            print(f"  • Expected RSI: {sample['expected_rsi']}")
            print(f"  • Expected ATR: {sample['expected_atr']}")
            print(f"  • Expected Score: {sample['expected_final_score']}")
            print(f"  • Actual Entry: {sample['actual_entry']}")
            print(f"  • Status: {sample['status']}")
        
        # 4. Market regimes found
        print("\n[4] Market Regimes:")
        cursor.execute("""
            SELECT DISTINCT expected_market_regime FROM execution_dataset
            WHERE expected_market_regime IS NOT NULL
        """)
        regimes = [row[0] for row in cursor.fetchall()]
        print(f"  {', '.join(regimes) if regimes else 'None found'}")
        
        # 5. Price Range by Symbol
        print("\n[5] Price Information from Latest Records:")
        cursor.execute("""
            SELECT symbol, 
                   MIN(expected_entry) as min_price,
                   MAX(expected_entry) as max_price,
                   AVG(expected_entry) as avg_price
            FROM execution_dataset
            WHERE expected_entry > 0
            GROUP BY symbol
        """)
        
        for row in cursor.fetchall():
            symbol, min_p, max_p, avg_p = row
            if symbol:
                print(f"  • {symbol:10} Min: {min_p:8.4f} Max: {max_p:8.4f} Avg: {avg_p:8.4f}")
        
        # 6. Latest data by symbol
        print("\n[6] Latest Records by Symbol:")
        cursor.execute("""
            SELECT symbol, order_id, expected_entry, dataset_updated_at 
            FROM execution_dataset
            WHERE dataset_updated_at IS NOT NULL
            ORDER BY symbol, dataset_updated_at DESC
            LIMIT 9
        """)
        
        current_symbol = None
        for row in cursor.fetchall():
            if row[0] != current_symbol:
                current_symbol = row[0]
                print(f"\n  {current_symbol}:")
            print(f"    • {row[3]} - Price: {row[2]}")
        
        # 7. Key metrics
        print("\n[7] Available Indicators in Data:")
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN expected_rsi IS NOT NULL THEN 1 END) as rsi_count,
                COUNT(CASE WHEN expected_atr IS NOT NULL THEN 1 END) as atr_count,
                COUNT(CASE WHEN expected_macd IS NOT NULL THEN 1 END) as macd_count,
                COUNT(CASE WHEN expected_trend_strength IS NOT NULL THEN 1 END) as trend_count
            FROM execution_dataset
        """)
        row = cursor.fetchone()
        print(f"  • Expected RSI data:     {row[0]} records")
        print(f"  • Expected ATR data:     {row[1]} records")
        print(f"  • Expected MACD data:    {row[2]} records")
        print(f"  • Expected Trend data:   {row[3]} records")
        
        print("\n" + "="*70)
        print("✅ Inspection Complete")
        print("="*70 + "\n")
        
        conn.close()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    inspect_execution_dataset()
