#!/usr/bin/env python3
"""
Commands Reference - الأوامر الأساسية والاستخدام السريع
"""

print("""
╔══════════════════════════════════════════════════════════════════╗
║           Trading Bot V3 - Commands Reference                    ║
║               الأوامر الأساسية والمراجع السريعة                ║
╚══════════════════════════════════════════════════════════════════╝

📖 DOCUMENTATION (الوثائق)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. START_HERE.md ⭐ (30 seconds read)
   الخطوات الأساسية للبدء السريع
   👉 اقرأ هذا أولاً!

2. FINAL_SOLUTION.md (5 minutes read)
   الحل النهائي الشامل مع كل التفاصيل

3. HYBRID_CLIENT_GUIDE.md (3 minutes read)
   شرح الـ Hybrid Client والاستخدام

4. QUICK_FIX.md (2 minutes read)
   حل سريع للمشاكل الشائعة

5. API_FIX_GUIDE_AR.md (بالعربية)
   دليل شامل بالعربية

6. FILES_SUMMARY.md
   قائمة كاملة بجميع الملفات الجديدة


🔍 DIAGNOSTIC COMMANDS (أوامر التشخيص)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# فحص قاعدة البيانات العامة
python inspect_database.py

# فحص execution_dataset بالتفصيل
python inspect_execution_dataset.py

# تشخيص شامل للـ API والاتصال
python diagnose_api.py

# فحص الـ hybrid client
python -c "
from data.market.hybrid_client import get_indicators_hybrid
from config import SYMBOLS
for s in SYMBOLS:
    indicators = get_indicators_hybrid(s)
    print(f'{s}: RSI={indicators[\"rsi\"]}, MACD={indicators[\"macd\"]}, ATR={indicators[\"atr\"]}')
"


💡 QUICK TEST (اختبار سريع)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# اختبر الـ hybrid client مباشرة
python -c "
from data.market.hybrid_client import get_indicators_hybrid
indicators = get_indicators_hybrid('EURUSD')
print(f'✓ Data: {indicators}')
"

# اختبر البيانات من execution_dataset
python -c "
from data.market.db_fallback_client import get_latest_indicators_from_db
from config import SYMBOLS
for s in SYMBOLS:
    data = get_latest_indicators_from_db(s)
    if data:
        print(f'{s}: ✓ Found {len(data)} indicators')
"

# اختبر التحديثات الجديدة
python test_api_improvements.py


🚀 RUN COMMANDS (تشغيل البوت)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# تشغيل البوت الرئيسي
python main.py

# تشغيل مع logging كامل
python main.py 2>&1 | tee logs/bot_run.log

# تشغيل مع معالجة الأخطاء
python -u main.py


📊 DATABASE QUERIES (استعلامات قاعدة البيانات)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# عد الصفوف في execution_dataset
python -c "
import sqlite3
conn = sqlite3.connect('trading_bot_v3.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM execution_dataset')
print(f'Total records: {cursor.fetchone()[0]}')
cursor.execute('SELECT symbol, COUNT(*) FROM execution_dataset GROUP BY symbol')
for s, count in cursor.fetchall():
    print(f'{s}: {count}')
conn.close()
"

# آخر البيانات المتاحة
python -c "
import sqlite3
conn = sqlite3.connect('trading_bot_v3.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute('SELECT symbol, expected_rsi, expected_atr, expected_entry FROM execution_dataset ORDER BY dataset_updated_at DESC LIMIT 3')
for row in cursor.fetchall():
    print(f'{row[0]}: RSI={row[1]}, ATR={row[2]}, Price={row[3]}')
conn.close()
"


✅ VERIFICATION CHECKLIST (قائمة التحقق)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

□ فحص قاعدة البيانات      → python inspect_database.py
□ فحص البيانات التفصيلية  → python inspect_execution_dataset.py
□ فحص الـ API والاتصال   → python diagnose_api.py
□ اختبر الـ hybrid client → python -c "..."
□ جميع الملفات موجودة؟    → ls *.md | grep -i solution
□ البوت جاهز للتشغيل؟     → python main.py


📁 IMPORTANT FILES (الملفات المهمة)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Core Hybrid Client:
  ✓ data/market/hybrid_client.py
  ✓ data/market/db_fallback_client.py

Enhanced Original:
  ✓ data/market/client.py (with fallback + caching)

Diagnostic Tools:
  ✓ diagnose_api.py
  ✓ inspect_database.py
  ✓ inspect_execution_dataset.py

Documentation:
  ✓ START_HERE.md              ← اقرأ أولاً!
  ✓ FINAL_SOLUTION.md
  ✓ HYBRID_CLIENT_GUIDE.md
  ✓ FILES_SUMMARY.md


🎯 NEXT STEPS (الخطوات التالية)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣  اقرأ START_HERE.md
2️⃣  اختبر: python inspect_execution_dataset.py
3️⃣  استخدم hybrid_client في البوت:
    
    from data.market.hybrid_client import get_indicators_hybrid as get_indicators

4️⃣  شغّل البوت:
    
    python main.py

5️⃣  راقب السجلات في logs/


💬 QUICK HELP (مساعدة سريعة)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: أين أقرأ الحل؟
A: START_HERE.md ثم FINAL_SOLUTION.md

Q: كيف أستخدم البيانات الجديدة؟
A: ادخل hybrid_client في main.py

Q: هل النظام آمن؟
A: نعم ✓ هناك 3 مستويات fallback

Q: كم من البيانات متوفرة؟
A: 630 صف تاريخي (EURUSD 217, XAUUSD 210, GBPUSD 203)

Q: ماذا لو فشل الـ hybrid client؟
A: سيستخدم FALLBACK_INDICATORS الآمنة

═══════════════════════════════════════════════════════════════════

Ready? Let's go! 🚀

Start with: python inspect_execution_dataset.py
Then read: START_HERE.md

═══════════════════════════════════════════════════════════════════
""")
