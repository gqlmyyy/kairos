import sqlite3
conn = sqlite3.connect('trading_bot_v3.db')
c = conn.cursor()
c.execute("SELECT CASE WHEN actual_pnl > 0 THEN 'WIN' ELSE 'LOSS' END as result, COUNT(*) FROM execution_dataset WHERE order_id LIKE 'HIST_%' GROUP BY result")
for row in c.fetchall():
    print(row)
conn.close()
