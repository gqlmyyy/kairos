import sqlite3
conn = sqlite3.connect('data/trading_bot_v3.db')
c = conn.cursor()
for col in [('breakeven_done','INTEGER DEFAULT 0'), ('trailing_done','INTEGER DEFAULT 0')]:
    try:
        c.execute(f"ALTER TABLE execution_dataset ADD COLUMN {col[0]} {col[1]};")
        print(f"تمت إضافة العمود {col[0]}")
    except sqlite3.OperationalError as e:
        if 'duplicate' in str(e).lower():
            print(f"العمود {col[0]} موجود مسبقاً")
        else:
            raise e
conn.commit()
c.execute("PRAGMA table_info(execution_dataset)")
print([row[1] for row in c.fetchall()])
conn.close()
