import sqlite3
from config import DB_FILE

print("DB_FILE=", DB_FILE)

conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='execution_dataset'")
r = c.fetchone()
print("execution_dataset exists?", r is not None)

c.execute("PRAGMA table_info(execution_dataset)")
cols = [row[1] for row in c.fetchall()]
print("columns_count", len(cols))

checks = [
    "expected_rsi",
    "expected_atr",
    "expected_ai_score",
    "expected_momentum_score",
    "actual_pnl",
    "slippage",
    "execution_quality_score",
    "expected_entry",
]
print("has columns:", {k: (k in cols) for k in checks})

total = c.execute("SELECT COUNT(*) FROM execution_dataset").fetchone()[0]

def safe_count(col):
    if col not in cols:
        return None
    return c.execute(f"SELECT COUNT(*) FROM execution_dataset WHERE {col} IS NOT NULL").fetchone()[0]

exp_entry = safe_count("expected_entry")
act_pnl = safe_count("actual_pnl")

print("total_rows", total)
print("expected_entry_not_null", exp_entry)
print("actual_pnl_not_null", act_pnl)

conn.close()
