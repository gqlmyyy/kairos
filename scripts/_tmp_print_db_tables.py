from __future__ import annotations

import sqlite3

DB_FILE = "trading_bot_v3.db"


def main() -> None:
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print("TABLE_COUNT", len(tables))
    print("TABLES_START")
    for t in sorted(tables):
        print(t)
    print("TABLES_END")

    candles = [t for t in tables if any(k in t.lower() for k in ["candle", "ohlc", "kline", "market_data", "price"])]
    print("CANDLES_CANDIDATES_START")
    for t in sorted(candles):
        print(t)
    print("CANDLES_CANDIDATES_END")

    con.close()


if __name__ == "__main__":
    main()
