import sqlite3
conn = sqlite3.connect("data/trading_bot_v3.db")
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
print(c.fetchall())
conn.close()
