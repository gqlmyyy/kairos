# Trading Bot V3 - telegram/notifier.py

import requests
from utils.logger import get_logger
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = get_logger("notifier")
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

def send(text: str):
    try:
        requests.post(f"{BASE_URL}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def notify_alert(msg: str):
    send(f"⚠️ <b>تنبيه</b>\n{msg}")

def notify_status(msg: str):
    send(f"ℹ️ {msg}")

def notify_start():
    send(f"""🚀 <b>Trading Bot V3 Started</b>
======================
Pairs: EURUSD, XAUUSD, GBPUSD, USDJPY
Decision: Voting System
SL/TP: ATR-based
Risk: Equity-based
Drawdown: 5/10/20% tiers
Correlation: Enabled
Reconciliation: 60s
Multi-TF: H4/H1/M15
Feedback: Adaptive Weights

Type /help for commands""")

def notify_trade_opened(symbol, direction, size, entry, sl, tp, score, confidence, reason):
    emoji = "🟢" if direction == "BUY" else "🔴"
    dir_ar = "شراء" if direction == "BUY" else "بيع"
    send(f"""{emoji} <b>تم فتح صفقة</b>
==================
الزوج: <b>{symbol}</b>
الاتجاه: <b>{dir_ar}</b>
الحجم: <b>{size}</b>
الدخول: <b>{entry}</b>
وقف الخسارة: <b>{sl}</b>
جني الأرباح: <b>{tp}</b>
النتيجة: <b>{score}/100</b>
الثقة: <b>{confidence:.0%}</b>
السبب: {reason[:80]}""")

def notify_trade_closed(symbol, direction, pnl, reason="", size=0, entry=0, exit_price=0):
    emoji = "💰" if pnl > 0 else "📉"
    result = "ربح" if pnl > 0 else "خسارة"
    dir_ar = "شراء" if direction in ["BUY", "buy"] else "بيع"
    send(f"""{emoji} <b>تم إغلاق صفقة</b>
==================
الزوج: <b>{symbol}</b>
الاتجاه: <b>{dir_ar}</b>
الحجم: <b>{size}</b>
الدخول: <b>{entry}</b>
الخروج: <b>{exit_price}</b>
{result}: <b>{abs(pnl):.2f}$</b>
السبب: {reason}""")

def notify_daily_report(stats):
    pnl = stats.get("total_pnl", 0)
    emoji = "📈" if pnl >= 0 else "📉"
    wr = (stats.get("winning_trades", 0) / stats["total_trades"] * 100) if stats.get("total_trades", 0) else 0
    send(f"""{emoji} <b>التقرير اليومي</b>
==================
الربح/الخسارة: <b>{pnl:+.2f}$</b>
الصفقات: <b>{stats.get('total_trades', 0)}</b>
نسبة الفوز: <b>{wr:.1f}%</b>
الخسائر المتتالية: <b>{stats.get('consecutive_losses', 0)}</b>""")