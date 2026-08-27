# Trading Bot V3 - config.py
# Central configuration with environment variables

from dotenv import load_dotenv  # noqa: F401
import os

load_dotenv()


# ==============================
# Environment readers
# ==============================
# `os.getenv(name, default)` returns the default only when the variable is
# *absent*. A present-but-empty `MT5_LOGIN=` returns "", and `int("")` raised a
# bare ValueError at import time — before any logging existed, from a module
# every entry point imports, with a traceback that did not name the variable.
# An empty `.env` line is the normal outcome of copying `.env.example`, so this
# took the whole bot down rather than degrading one feature.
#
# These readers treat empty/whitespace-only as absent, and turn a malformed
# value into an error that says which variable is wrong.


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(
            f"config: environment variable {name}={raw!r} is not a valid "
            f"integer (expected something like {default})"
        ) from None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        raise ValueError(
            f"config: environment variable {name}={raw!r} is not a valid "
            f"number (expected something like {default})"
        ) from None


# ==============================
# API Keys
# ==============================

DEEPSEEK_API_KEY = _env_str("DEEPSEEK_API_KEY", "")
FINNHUB_API_KEY = _env_str("FINNHUB_API_KEY", "")
TELEGRAM_BOT_TOKEN = _env_str("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = _env_str("TELEGRAM_CHAT_ID", "")

# ==============================
# MT5
# ==============================
MT5_LOGIN = _env_int("MT5_LOGIN", 0)
MT5_PASSWORD = _env_str("MT5_PASSWORD", "")
MT5_SERVER = _env_str("MT5_SERVER", "")
MT5_PATH = _env_str("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")

# ==============================
# Trading Pairs
# ==============================
SYMBOLS = ["EURUSD", "XAUUSD", "GBPUSD"]

# Entry model selection.
#
# v1 is the only supported value. The "v2" path routed inference through
# analysis/entry_v2, which is quarantined: its dataset has proven look-ahead,
# its entry price resolves to an EMA, its H4 indicators are computed on an H1
# grid, and it has no direction column (ENTRY_PIPELINE_AUDIT.md).
#
# That path could never have produced a tradeable number anyway — it sends 10
# placeholder scalars to a 65-feature artifact, so the contract check blocks
# every call. Failing loudly at startup is better than a bot that runs all day
# and silently refuses every signal.
#
# 'research' routes the entry gate to analysis/research/live_gate.py: the
# xgbooost research models, resolved from models/research/registry.json by
# (symbol, timeframe) and pinned to generation research_v2. It never reads
# models/entry/entry_model.json.
#
# That gate requires two independent conditions — the registry status its own
# module defines, and a separate activation record — and no shipped model
# satisfies either today. Selecting 'research' therefore blocks every signal
# with a reason naming what is missing. That is the intended fail-closed
# state, not a misconfiguration; see analysis/research/live_gate.py and
# analysis/research/production_gate.py.
ENTRY_MODEL_VERSION = _env_str("ENTRY_MODEL_VERSION", "v1")

SUPPORTED_ENTRY_MODEL_VERSIONS = ("v1", "research")

if ENTRY_MODEL_VERSION not in SUPPORTED_ENTRY_MODEL_VERSIONS:
    raise ValueError(
        f"ENTRY_MODEL_VERSION={ENTRY_MODEL_VERSION!r} is not supported. "
        f"Available: {', '.join(SUPPORTED_ENTRY_MODEL_VERSIONS)}. The 'v2' entry "
        f"path is quarantined pending a rebuilt data pipeline — see "
        f"ENTRY_PIPELINE_AUDIT.md."
    )


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
MT5_ORDER_TYPE_FILLING = _env_str("MT5_ORDER_TYPE_FILLING", "")


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
BROKER_UTC_OFFSET_HOURS = _env_float("BROKER_UTC_OFFSET_HOURS", 3.0)


# ==============================
# Trade-management settings live in trade_management/tm_config.py
# ==============================
# ML_EXIT_ENABLED, ATR_SL_BASE_MULTIPLIER, ATR_TP_BASE_MULTIPLIER and
# MAX_SL_PIPS used to be defined here as well. Nothing read those copies —
# every consumer resolves them through tm_config — so two values with the same
# name could disagree, and editing the one here silently did nothing.
# Single source of truth: trade_management/tm_config.py.

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
# Signal feature calibration
# ==============================
# These three blocks exist because the corresponding features were frozen
# constants in production (KNOWN_ISSUES #13). Every threshold here is read by
# BOTH the live path and the training pipeline, so the two cannot disagree.
#
# --- Sideways band (fixes market_regime) ---
# `ma_trend` used to return "sideways" only when `price == ma20` exactly, a
# float equality that never occurs, so the H4 trend direction was never
# "neutral" and the regime was always TRENDING. Price is now treated as flat
# when it sits within this multiple of ATR of MA20 — an ATR-relative band, so
# it means the same thing on EURUSD as on XAUUSD.
MA_TREND_FLAT_ATR_MULT = 0.25

# --- Volatility buckets (fixes volatility_score) ---
# `get_volatility_score_from_snapshot` reads a "volatility" key that no code
# path emitted, so every lookup fell through to the neutral default of 55.
#
# The measure is current ATR against the *median ATR of the same symbol over the
# indicator window*: "is this symbol more volatile than it usually is".
# An absolute ATR% threshold cannot work here — measured on live data, EURUSD H4
# runs at ~0.14% of price while XAUUSD runs at ~0.96%, so any fixed cut pins each
# symbol to one bucket forever, which is the same frozen-feature bug in a new
# place. A self-relative ratio is scale-free and comparable across instruments.
VOLATILITY_RATIO_VERY_HIGH = 1.50   # >= 1.5x its own typical ATR
VOLATILITY_RATIO_HIGH = 1.20
VOLATILITY_RATIO_LOW = 0.80         # <  0.8x

# --- Trend strength encoding (fixes trend_strength) ---
# MultiTimeframeData.strength is a string ("weak"/"moderate"/"strong"). main.py
# guarded it with isinstance(..., (int, float)), which never passed, so the
# model always received 0.0. These are the numeric values the string maps to.
TREND_STRENGTH_VALUES = {
    "weak": 25.0,
    "moderate": 60.0,
    "strong": 100.0,
}
TREND_STRENGTH_DEFAULT = 0.0  # unknown/absent — distinct from a real "weak"



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
LOG_LEVEL_FILE = _env_str("LOG_LEVEL_FILE", "INFO")
LOG_LEVEL_CONSOLE = _env_str("LOG_LEVEL_CONSOLE", "INFO")
LOG_RETENTION_DAYS = _env_int("LOG_RETENTION_DAYS", 14)

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

