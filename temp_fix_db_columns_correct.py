import sqlite3

conn = sqlite3.connect('trading_bot_v3.db')
c = conn.cursor()

for col in [('breakeven_done','INTEGER DEFAULT 0'), ('trailing_done','INTEGER DEFAULT 0')]:
    try:
        c.execute(f"ALTER TABLE execution_dataset ADD COLUMN {col[0]} {col[1]};")
        print(f"تمت إضافة العمود {col[0]}")
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if 'duplicate' in msg or 'already exists' in msg:
            print(f"العمود {col[0]} موجود مسبقاً")
        elif 'no such table' in msg:
            raise e
        else:
            # SQLite can also throw "duplicate column name" depending on version
            if 'column' in msg and 'exists' in msg:
                print(f"العمود {col[0]} موجود مسبقاً")
            else:
                raise e

conn.commit()

rows = c.execute("PRAGMA table_info(execution_dataset)").fetchall()
cols = [r[1] for r in rows]
print("columns:", cols)
print("has_breakeven_done:", 'breakeven_done' in cols)
print("has_trailing_done:", 'trailing_done' in cols)

conn.close()
