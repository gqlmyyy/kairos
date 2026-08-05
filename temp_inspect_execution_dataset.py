import sqlite3
db_path = "trading_bot_v3.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

print("tables:", [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()])

rows = c.execute("PRAGMA table_info(execution_dataset)").fetchall()
cols = [r[1] for r in rows]
print("has_breakeven_done", "breakeven_done" in cols)
print("has_trailing_done", "trailing_done" in cols)
print("columns_count", len(cols))
print("columns", cols)
conn.close()
