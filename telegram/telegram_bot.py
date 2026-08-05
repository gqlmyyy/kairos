# Trading Bot V3 - telegram/telegram_bot.py
# Full Telegram bot with all required commands - Arabic Support

import threading
import time
import requests
from datetime import datetime
from utils.logger import get_logger
from data.storage.database import get_daily_stats, get_open_trades, get_last_decisions, get_weights
from execution.mt5_direct import get_open_positions_mt5, close_trade_mt5, close_all_trades_mt5, check_mt5_status_mt5
from telegram.notifier import send
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, QUANTDINGER_URL, SYMBOLS, INITIAL_WEIGHTS

logger = get_logger("telegram_bot")
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

bot_state = None

def set_state(state):
   global bot_state
   bot_state = state

def get_updates(offset=None):
   try:
       r = requests.get(f"{BASE_URL}/getUpdates",
                      params={"timeout": 30, "offset": offset}, timeout=35)
       data = r.json()
       return data.get("result", []) if data else []
   except Exception as e:
       logger.error(f"Updates error: {e}")
       return []

def reply(chat_id, text):
   try:
       requests.post(f"{BASE_URL}/sendMessage",
                    json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
   except Exception as e:
       logger.error(f"Reply error: {e}")

def translate_direction(d):
   d = str(d).lower()
   if d in ["buy", "bullish"]:
       return "شراء 📈"
   elif d in ["sell", "bearish"]:
       return "بيع 📉"
   return "محايد ↔️"

def translate_bias(b):
   b = str(b).lower()
   if b == "bullish":
       return "صاعد 📈"
   elif b == "bearish":
       return "هابط 📉"
   return "محايد ↔️"

def translate_regime(r):
   r = str(r).upper()
   if r == "TRENDING":
       return "اتجاه قوي 🚀"
   elif r == "RANGING":
       return "متذبذب ↔️"
   elif r == "VOLATILE":
       return "متقلب ⚡"
   return r

def cmd_status():
   global bot_state
   uptime = datetime.now() - bot_state["start_time"]
   h = int(uptime.total_seconds() // 3600)
   m = int((uptime.total_seconds() % 3600) // 60)
   stats = get_daily_stats()
   mt5 = "✅" if check_mt5_status_mt5() else "❌"
   try:
       r = requests.get(f"{QUANTDINGER_URL}/api/health", timeout=5)
       qd_status = "✅" if r.status_code == 200 else "❌"
   except:
       qd_status = "❌"
   icon = "⏸" if bot_state["trading_paused"] else "✅"
   return f"""🤖 <b>بوت التداول V3</b>
========================
⏱ وقت التشغيل: <b>{h}س {m}د</b>
🔄 الدورات: <b>{bot_state['cycle_count']}</b>
🕐 آخر دورة: <b>{bot_state['last_cycle'] or 'لا يوجد'}</b>

QuantDinger: {qd_status}
MT5: {mt5}
التداول: {icon}

📊 <b>اليوم:</b>
الصفقات: <b>{stats.get('total_trades', 0)}</b>
الربح/الخسارة: <b>{stats.get('total_pnl', 0):+.2f}$</b>"""

def cmd_positions():
   positions = get_open_positions_mt5()
   if not positions:
       return "📭 لا توجد صفقات مفتوحة."
   lines = [f"📋 <b>الصفقات المفتوحة ({len(positions)})</b>", "========================"]
   for p in positions:
       sym = p.get("symbol", "?")
       dir_ = p.get("direction", p.get("type", "?"))
       vol = p.get("size", p.get("volume", 0))
       entry = p.get("price", p.get("price_open", 0))
       profit = float(p.get("profit", p.get("pnl", 0)))
       arrow = "🟢" if str(dir_).lower() in ["buy", "0"] else "🔴"
       pnl_icon = "💰" if profit >= 0 else "📉"
       tid = p.get("id", p.get("ticket", ""))
       lines.append(f"{arrow} <b>{sym}</b> | {translate_direction(dir_)}")
       lines.append(f"   التذكرة: {tid} | الحجم: {vol} | الدخول: {entry}")
       lines.append(f"   {pnl_icon} الربح/الخسارة: <b>{profit:+.2f}$</b>")
   return "\n".join(lines)

def cmd_balance():
   from execution.quantdinger_client import get_equity as qd_equity
   from execution.quantdinger_client import get_headers
   equity = qd_equity()
   info = {"balance": 0, "margin": 0, "free_margin": 0, "equity": 0}
   try:
       r = requests.get(f"{QUANTDINGER_URL}/api/mt5/account", headers=get_headers(), timeout=5)
       data = r.json()
       if data.get("success"):
           info = data
   except:
       pass
   return f"""💼 <b>معلومات الحساب</b>
==================
💰 الرصيد الفعلي: <b>{float(info.get('equity', equity)):.2f}$</b>
🏦 الرصيد: <b>{float(info.get('balance', 0)):.2f}$</b>
📊 الهامش المستخدم: <b>{float(info.get('margin', 0)):.2f}$</b>
✅ الهامش المتاح: <b>{float(info.get('margin_free', 0)):.2f}$</b>"""

def cmd_why():
   decisions = get_last_decisions(5)
   if not decisions:
       return "📭 لا توجد قرارات حديثة."
   lines = ["🧠 <b>آخر القرارات</b>", "==================="]
   for d in decisions:
       s = d.get("symbol", d[0] if isinstance(d, (list, tuple)) else "?")
       dir_ = d.get("direction", d[1] if isinstance(d, (list, tuple)) else "?")
       score = d.get("final_score", d[2] if isinstance(d, (list, tuple)) else 0)
       reason = d.get("reason", d[3] if isinstance(d, (list, tuple)) else "")
       action = d.get("action", d[4] if isinstance(d, (list, tuple)) else "")
       action_ar = "تخطي" if action == "SKIP" else "تداول" if action == "TRADE" else action
       lines.append(f"<b>{s}</b>: {translate_direction(dir_)} | النتيجة: {score} [{action_ar}]")
       lines.append(f"  📝 {reason[:80]}")
   return "\n".join(lines)

def cmd_report():
   stats = get_daily_stats()
   pnl = stats.get("total_pnl", 0)
   wr = 0
   if stats.get("total_trades", 0):
       wr = stats.get("winning_trades", 0) / stats["total_trades"] * 100
   emoji = "📈" if pnl >= 0 else "📉"
   return f"""{emoji} <b>التقرير اليومي</b>
==================
💰 الربح/الخسارة: <b>{pnl:+.2f}$</b>
📊 الصفقات: <b>{stats.get('total_trades', 0)}</b>
✅ الرابحة: <b>{stats.get('winning_trades', 0)}</b>
❌ الخاسرة: <b>{stats.get('losing_trades', 0)}</b>
🎯 نسبة الفوز: <b>{wr:.1f}%</b>"""

def cmd_analyze(symbol=None):
   if not symbol:
       symbols = SYMBOLS
   else:
       symbols = [symbol]
   from analysis.ai.deepseek import analyze_news
   from data.news.fetcher import fetch_rss_news, filter_news_for_symbol

   # snapshot-only indicators/regime (same as main.run_cycle)
   from analysis.technical.indicators import (
       get_trend_score_from_snapshot,
       get_momentum_score_from_snapshot,
       get_volatility_score_from_snapshot,
   )
   from analysis.technical.regime import get_market_regime_from_snapshot
   from data.market.market_snapshot_builder import MarketSnapshotBuilder

   news = fetch_rss_news()

   # Build a single shared snapshot for all requested symbols, exactly like
   # main.run_cycle() does.
   try:
       snapshot_builder = MarketSnapshotBuilder()
       snapshot = snapshot_builder.build(symbols)
   except Exception as e:
       logger.error(f"cmd_analyze: snapshot build failed: {e}")
       snapshot = None

   lines = ["🔍 <b>تحليل الأزواج</b>", "==================="]
   for sym in symbols:
       symbol_news = filter_news_for_symbol(news, sym)
       ai = analyze_news(symbol_news, sym, snapshot)

       if snapshot is None:
           trend_score, trend_dir = 40, "neutral"
           regime = "UNKNOWN"
           logger.warning(
               f"cmd_analyze: using fallback values for {sym} because snapshot build failed"
           )
       else:
           # main.py uses: trend_score, trend_dir = mtf.h4_score, mtf.h4_direction
           # but for telegram we explicitly call the snapshot-based helpers.
           trend_data = get_trend_score_from_snapshot(snapshot, sym)
           trend_score, trend_dir = trend_data if isinstance(trend_data, tuple) else (trend_data, "neutral")

           regime = get_market_regime_from_snapshot(snapshot, sym)

       strength = "قوي" if ai.confidence >= 0.75 else "متوسط" if ai.confidence >= 0.60 else "ضعيف"
       lines.append(f"\n💱 <b>{sym}</b>")
       lines.append(f"  🤖 AI: {translate_bias(ai.bias)} | النتيجة: {ai.impact_score} | الثقة: {ai.confidence:.0%} ({strength})")
       lines.append(f"  📈 الاتجاه: {translate_bias(trend_dir)} | النتيجة: {trend_score}")
       lines.append(f"  🌊 النظام: {translate_regime(regime)}")
       if ai.reason:
           lines.append(f"  📝 {ai.reason[:100]}")

   return "\n".join(lines)

def cmd_news():
   from data.news.fetcher import fetch_rss_news
   news = fetch_rss_news()
   if not news:
       return "📭 لا توجد أخبار."
   lines = ["📰 <b>آخر الأخبار</b>", "=================="]
   for n in news[:8]:
       impact = "🔴" if n.is_high_impact else "⚪"
       lines.append(f"{impact} [{n.source}] {n.headline[:70]}")
   return "\n".join(lines)

def cmd_emergency():
   closed = close_all_trades_mt5()
   bot_state["trading_paused"] = True
   return f"🚨 <b>إغلاق طارئ</b>\nتم إغلاق {closed} صفقة. التداول متوقف."

def cmd_stop():
   bot_state["trading_paused"] = True
   return "⏹ تم إيقاف التداول."

def cmd_start_trading():
   bot_state["trading_paused"] = False
   bot_state["pause_until"] = None
   return "▶️ تم استئناف التداول."

def cmd_pause(args):
   if args:
       try:
           h = int(args[0].replace("h", ""))
           bot_state["pause_until"] = time.time() + h * 3600
           bot_state["trading_paused"] = True
           return f"⏸ تم الإيقاف المؤقت لمدة {h} ساعة."
       except:
           pass
   bot_state["trading_paused"] = True
   return "⏸ تم الإيقاف المؤقت (حتى الاستئناف اليدوي)."

def cmd_weights():
   lines = ["⚖️ <b>أوزان القرار</b>", "==================="]
   for sym in SYMBOLS:
       w = get_weights(sym)
       lines.append(f"<b>{sym}</b>: AI={w['ai']:.2f} | اتجاه={w['trend']:.2f} | زخم={w['momentum']:.2f}")
   return "\n".join(lines)

def cmd_close(args):
   """إغلاق صفقة معينة"""
   if not args:
       return "❌ استخدم: /close TICKET_NUMBER"
   ticket = args[0]
   success = close_trade_mt5(ticket)
   if success:
       logger.info(f"[Telegram] /close success ticket={ticket}")
       return f"✅ تم إغلاق الصفقة {ticket}"
   logger.info(f"[Telegram] /close failed ticket={ticket}")
   return f"❌ فشل إغلاق الصفقة {ticket}"

def cmd_performance():
    from data.storage.database import get_all_symbols_performance
    perfs = get_all_symbols_performance()
    if not perfs or all(p["total"] == 0 for p in perfs):
        return "📊 لا توجد بيانات أداء بعد."
    lines = ["📊 <b>تحليل الأداء</b>", "=================="]
    for p in perfs:
        if p["total"] == 0:
            continue
        emoji = "🏆" if p["total_pnl"] > 0 else "⚠️"
        lines.append(f"{emoji} <b>{p['symbol']}</b>")
        lines.append(f"   الصفقات: {p['total']} | الفوز: {p['win_rate']}%")
        lines.append(f"   PnL: {p['total_pnl']:+.2f}$ | PF: {p['profit_factor']}")
        lines.append(f"   أفضل: {p['best_trade']:+.2f}$ | أسوأ: {p['worst_trade']:+.2f}$")
    return "\n".join(lines)

def cmd_dataset_status():
   """Bootstrap-only dataset status.

   Dataset metrics based on execution_dataset rows:
   - trainable_rows: rows where at least one actual label exists (actual_pnl or execution_quality_score)
   - rejected_bootstrap_rows: placeholder count based on NULL expected fields (no reliable rejected table exists in current DB schema)

   NOTE: This command must NOT include live trades. It reads only execution_dataset.
   """
   from data.storage.database import get_conn
   conn = get_conn()
   c = conn.cursor()

   # trainable: label available for ML training
   c.execute("""
      SELECT COUNT(*) FROM execution_dataset
      WHERE status IN ('closed','open')
        AND (actual_pnl IS NOT NULL OR execution_quality_score IS NOT NULL)
   """)
   trainable = int(c.fetchone()[0] or 0)

   # rejected: rows where expected business snapshot is missing (proxy).
   c.execute("""
      SELECT COUNT(*) FROM execution_dataset
      WHERE status IN ('closed','open')
        AND expected_entry IS NULL
   """)
   rejected = int(c.fetchone()[0] or 0)

   conn.close()

   total = trainable + rejected
   dataset_quality = (trainable / total) if total > 0 else 0.0

   return (
      f"📦 Dataset Status\n"
      f"Dataset Quality: {dataset_quality:.2f}\n"
      f"Trainable: {trainable}\n"
      f"Rejected: {rejected}\n"
      f"JSON: {{\"trainable_rows\": {trainable}, \"rejected_bootstrap_rows\": {rejected}, \"dataset_quality\": {dataset_quality:.4f}}}"
   )

def cmd_help():
   return """📖 <b>قائمة الأوامر</b>
========================
/status - حالة النظام
/positions - الصفقات المفتوحة
/balance - رصيد الحساب
/report - التقرير اليومي
/why - آخر القرارات
/analyze - تحليل جميع الأزواج
/ai EURUSD - تحليل زوج معين
/performance - تحليل أداء الأزواج
/news - آخر الأخبار
/weights - أوزان القرار
/dataset_status - جودة dataset (bootstrap only)
/close 12345 - إغلاق صفقة
/stop - إيقاف التداول
/start - استئناف التداول
/pause 2h - إيقاف مؤقت لساعتين
/emergency - إغلاق كل شيء! 🚨"""

COMMANDS = {
   "/status": cmd_status,
   "/positions": cmd_positions,
   "/balance": cmd_balance,
   "/why": cmd_why,
   "/report": cmd_report,
   "/weights": cmd_weights,
   "/dataset_status": cmd_dataset_status,
   "/stop": cmd_stop,
   "/start": cmd_start_trading,
   "/emergency": cmd_emergency,
   "/news": cmd_news,
   "/performance": cmd_performance,
   "/help": cmd_help,
}

def handle(text, chat_id):
   global bot_state
   if str(chat_id) != str(TELEGRAM_CHAT_ID):
       reply(chat_id, "⛔ غير مصرح.")
       return
   parts = text.strip().split()
   cmd = parts[0].lower()
   args = parts[1:] if len(parts) > 1 else []
   if cmd in COMMANDS:
       reply(chat_id, COMMANDS[cmd]())
   elif cmd == "/pause":
       reply(chat_id, cmd_pause(args))
   elif cmd == "/close":
       reply(chat_id, cmd_close(args))
   elif cmd in ["/analyze", "/ai"]:
       sym = args[0].upper() if args else None
       reply(chat_id, "🔍 جاري التحليل...")
       threading.Thread(target=lambda: reply(chat_id, cmd_analyze(sym)), daemon=True).start()
   else:
       reply(chat_id, "❓ أمر غير معروف. /help")

def polling_loop():
   offset = None
   logger.info("Telegram polling started")
   while True:
       try:
           updates = get_updates(offset)
           for u in updates:
               offset = u["update_id"] + 1
               msg = u.get("message", {})
               text = msg.get("text", "")
               chat_id = msg.get("chat", {}).get("id")
               if text and text.startswith("/"):
                   handle(text, chat_id)
       except Exception as e:
           logger.error(f"Polling error: {e}")
           time.sleep(5)

def heartbeat_loop():
   while True:
       time.sleep(900)
       try:
           stats = get_daily_stats()
           trades = get_open_positions_mt5()
           pnl = stats.get("total_pnl", 0)
           if bot_state["pause_until"] and time.time() > bot_state["pause_until"]:
               bot_state["trading_paused"] = False
               bot_state["pause_until"] = None
               send("▶️ انتهى الإيقاف المؤقت - استُؤنف التداول.")
           paused = "⏸" if bot_state["trading_paused"] else "✅"
           pnl_icon = "💰" if pnl >= 0 else "📉"
           send(f"""💓 <b>نبضة V3</b>
الدورات: <b>{bot_state['cycle_count']}</b>
التداول: {paused}
الصفقات المفتوحة: <b>{len(trades)}</b>
{pnl_icon} الربح/الخسارة: <b>{pnl:+.2f}$</b>""")
       except Exception as e:
           logger.error(f"Heartbeat error: {e}")

def start_telegram_bot(state):
   set_state(state)
   threading.Thread(target=polling_loop, daemon=True).start()
   threading.Thread(target=heartbeat_loop, daemon=True).start()
   logger.info("Telegram bot V3 started")