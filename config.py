# Trading Bot V3 - config.py
# Central configuration with environment variables

from dotenv import load_dotenv  # noqa: F401
import os

load_dotenv()



# ==============================
# API Keys
# ==============================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-2922e9853c654120b9b89844295efdc5")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d8l1ia9r01qut1f8gd80d8l1ia9r01qut1f8gd8g")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8543232380:AAHqo6L7Lntf2C3Tu5Dfo4U2bWn5os_XPk8")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6697592398")

# ==============================
# MT5
# ==============================
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "110609311"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "*7DsDjJu")
MT5_SERVER = os.getenv("MT5_SERVER", "MetaQuotes-Demo")
MT5_PATH = os.getenv("MT5_PATH", r"C:\\Program Files\\MetaTrader 5\\terminal64.exe")

# ==============================
# Trading Pairs
# ==============================
SYMBOLS = ["EURUSD", "XAUUSD", "GBPUSD"]

# Entry model selection
# v1: legacy entry inference (existing)
# v2: independent entry_v2 architecture (self-contained)
ENTRY_MODEL_VERSION = os.getenv("ENTRY_MODEL_VERSION", "v1")


# ==============================
# Timeframes
# ==============================
TF_TREND = "H4"        # Trend direction
TF_DECISION = "H1"     # Entry decision
TF_TIMING = "M15"      # Entry timing

# ==============================
# Risk Management
# ==============================
BASE_RISK_PERCENT = 0.005       # 0.5% base risk per trade
MAX_DAILY_LOSS = 0.03           # 3% max daily loss → STOP
MAX_DAILY_LOSS_USD = 500        # أقصى خسارة يومية بالدولار
MAX_DRAWDOWN_HALT = 0.05        # 5% daily DD → stop trading
ACCOUNT_DRAWDOWN_HALF = 0.10    # 10% account DD → half risk
ACCOUNT_DRAWDOWN_STOP = 0.20    # 20% account DD → full stop
MAX_OPEN_TRADES = 3
STOP_AFTER_LOSSES = 10
# Hard ceiling on a single trade's risk, as a fraction of equity.
#
# BASE_RISK_PERCENT above is the *target* budget and scales with signal
# strength. This is the absolute limit that target may never be rounded past.
#
# It matters because broker lot sizes are discrete: when the risk-correct size
# falls below the minimum lot, the choice is to take the minimum lot (risking
# more than budgeted) or skip the trade. Overshooting the soft budget slightly
# is acceptable; overshooting this ceiling is not, and the trade is refused.
#
# Sized from a real incident: a $99.40 account taking XAUUSD at the 0.01
# minimum lot with a correct 1.5xATR stop risks $71 — 71% of the account.
MAX_RISK_PER_TRADE_PCT = 0.02
MAX_SL_PIPS = 100
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.5

# ==============================
# Decision Voting Weights
# ==============================
INITIAL_WEIGHTS = {
    "ai":         0.30,
    "trend":      0.30,
    "momentum":   0.20,
    "sentiment":  0.10,
    "volatility": 0.10,
}
MIN_SCORE = 45
AI_MIN_CONFIDENCE = 0.6
SIGNAL_MIN_CONFIDENCE = 0.6

# ==============================
# News Scoring
# ==============================
NEWS_SOURCE_WEIGHTS = {
    "reuters": 1.0, "bloomberg": 1.0, "forexlive": 0.8,
    "dailyfx": 0.8, "fxstreet": 0.7, "kitco": 0.7,
    "bbc": 0.6, "investing": 0.6, "finnhub": 0.6,
    "default": 0.5,
}
NEWS_DECAY_HOURS = 8
NEWS_CHECK_INTERVAL = 1800  # 30 min

# ==============================
# Execution
# ==============================
MAX_RETRIES = 3
RETRY_DELAY = 5
ORDER_TIMEOUT = 30

# ==============================
# MT5 Order Filling Mode
# ==============================
# Empty = let mt5_direct probe the symbol's supported filling modes.
MT5_ORDER_TYPE_FILLING = os.getenv("MT5_ORDER_TYPE_FILLING", "")


# ==============================
# Feedback / Learning
# ==============================
WEIGHT_LEARNING_RATE = 0.05
MIN_TRADES_TO_LEARN = 10
WEIGHT_SMOOTHING = 0.9
FEEDBACK_BATCH_SIZE = 20

# ==============================
# Post-entry manager loop (used for synchronization/guards)
# ==============================
POST_ENTRY_LOOP_INTERVAL_SEC = 5.0

# ==============================
# Reconciliation
# ==============================
RECONCILIATION_INTERVAL = 60  # every 60 seconds

# ==============================
# Time / timezone normalization
# ==============================
# MT5 position timestamps (e.g., `position.time`) are broker server time and may be
# offset from UTC. Some brokers use UTC+2 / UTC+3 depending on DST.
# This offset is used to convert MT5 broker-local times to UTC for all time deltas.
# IMPORTANT: if broker DST changes, update this value accordingly.
BROKER_UTC_OFFSET_HOURS = float(os.getenv("BROKER_UTC_OFFSET_HOURS", "3"))


# ============================================================================
# AI/ML EXIT MODEL - DISABLED (Feature Flag)
# ============================================================================
# The XGBoost exit model is DISABLED until it proves out-of-sample performance
# (AUC and accuracy clearly above random chance). Do NOT re-enable without
# validating on a held-out dataset.
#
#   - ML_EXIT_ENABLED=False  -> the AI/ML exit model NEVER affects exit decisions.
#   - The code is kept intact but all call paths are gated by this flag.
# ============================================================================
ML_EXIT_ENABLED = False

# ==============================
# ATR-based Risk Management (regime-aware)
# ==============================
# SL/TP multipliers are adjusted per market regime (taken from get_regime_settings).
# high_volatility -> wider SL, mean_reversion -> tighter SL, trend -> extended TP.
ATR_SL_BASE_MULTIPLIER = 1.5
ATR_TP_BASE_MULTIPLIER = 2.5

# ==============================
# Risk Governor (independent halt)
# ==============================
# Maximum cumulative loss in R units before halting new entries.
RISK_GOVERNOR_MAX_LOSS_R = 6.0
# Persist halt state across bot restarts.
RISK_GOVERNOR_PERSIST = True
# Consecutive losing trades before new entries are halted.
# Imported by risk/risk_governor.py — it previously lived in the "Equity Guard"
# section, which was removed with the old trade-management generation. Losing it
# broke the governor's import entirely, which silently disabled risk halting.
MAX_CONSECUTIVE_LOSSES = 3

# ==============================
# Market Regime Detection
# ==============================
MARKET_REGIME_ENABLED = True
REGIME_ADX_THRESHOLD = 25
REGIME_LOW_ADX_THRESHOLD = 20



# ==============================
# Watchdog
# ==============================
WATCHDOG_INTERVAL = 120  # every 2 minutes
WATCHDOG_FAIL_LIMIT = 3

# ==============================
# Correlation
# ==============================
MAX_CORRELATION = 0.80
CORRELATION_LOOKBACK = 100

# ==============================
# Correlation Protection (risk)
# ==============================
CORRELATION_PROTECTION_ENABLED = True
CORRELATION_ACTION = "block"  # "block" | "close_old"
# Pairs that are treated as correlated (used by new correlation_protection module if needed)
CORRELATED_PAIRS = [("EURUSD", "GBPUSD"), ("XAUUSD", "XAGUSD")]


# ==============================
# Logging
# ==============================
# File handler rotates daily at midnight and keeps LOG_RETENTION_DAYS backups.
LOG_LEVEL_FILE = os.getenv("LOG_LEVEL_FILE", "INFO")
LOG_LEVEL_CONSOLE = os.getenv("LOG_LEVEL_CONSOLE", "INFO")
LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "14"))

# Per-module overrides. The post-entry hot loop runs every few seconds, so its
# chatty components are pinned to WARNING to keep the daily log readable.
LOG_LEVEL_PER_MODULE = {
    "tm.orchestrator": "INFO",
    "tm.exit_score": "WARNING",
    "tm.adaptive_trailing": "WARNING",
    "tm.trade_age": "WARNING",
    "tm.partial_tp": "WARNING",
    "tm.breakeven": "WARNING",
    "tm.min_modify": "WARNING",
    "post_entry_trade_monitor": "WARNING",
    "mt5_client": "INFO",
    "mt5_session": "INFO",
    "hybrid_market_client": "INFO",
}

# ==============================
# Performance DB
# ==============================
DB_FILE = "trading_bot_v3.db"

PIP_VALUES = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001,
    "USDJPY": 0.01, "XAUUSD": 0.1,
    "USDCAD": 0.0001, "AUDUSD": 0.0001,
    "NZDUSD": 0.0001, "USDCHF": 0.0001,
}

