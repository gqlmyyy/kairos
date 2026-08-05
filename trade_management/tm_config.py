"""Single source of truth for every trade-management number.

No layer may hard-code a threshold, weight, bar count or R multiple. If a value
influences a decision it belongs here.

Values are grouped per layer. Environment variables override the defaults so a
paper/shadow deployment can be tuned without editing code.
"""

from __future__ import annotations

import os


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


def _b(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# =============================================================================
# LAYER 1 - Baseline protection
# =============================================================================

# --- Initial protection (ATR-based SL/TP at entry) ---
ATR_SL_BASE_MULTIPLIER = _f("TM_ATR_SL_BASE_MULTIPLIER", 1.5)
ATR_TP_BASE_MULTIPLIER = _f("TM_ATR_TP_BASE_MULTIPLIER", 2.5)
MAX_SL_PIPS = _i("TM_MAX_SL_PIPS", 100)

# Regime adjustment applied on top of the base multipliers: (sl_factor, tp_factor)
REGIME_SLTP_FACTORS = {
    "high_volatility": (1.3, 1.0),
    "volatile": (1.3, 1.0),
    "trending": (1.0, 1.3),
    "trend": (1.0, 1.3),
    "weak_trend": (0.9, 1.1),
    "mean_reversion": (0.8, 0.8),
    "ranging": (0.8, 0.8),
    "sideways": (0.8, 0.8),
    "normal": (1.0, 1.0),
    "unknown": (1.0, 1.0),
}

# --- Break-even ---
BREAKEVEN_ENABLED = _b("TM_BREAKEVEN_ENABLED", True)
# Profit (in R) required before SL is pulled to entry.
BREAKEVEN_TRIGGER_R = _f("TM_BREAKEVEN_TRIGGER_R", 1.0)
# Small cushion beyond entry so break-even covers commission/spread, in ATR units.
BREAKEVEN_OFFSET_ATR = _f("TM_BREAKEVEN_OFFSET_ATR", 0.05)

# --- Minimum modify distance (final filter before any SL/TP write) ---
MIN_MODIFY_ENABLED = _b("TM_MIN_MODIFY_ENABLED", True)
# An SL/TP change is only sent to the broker when it moves at least this many
# points. Prevents modify-spam on a loop that ticks every few seconds.
MIN_MODIFY_DISTANCE_POINTS = _f("TM_MIN_MODIFY_DISTANCE_POINTS", 15.0)
# Fallback point size when the broker does not expose symbol_info.point.
DEFAULT_POINT_SIZE = _f("TM_DEFAULT_POINT_SIZE", 0.00001)
# Never send an SL closer to price than the broker's stop level (points).
MIN_BROKER_STOP_BUFFER_POINTS = _f("TM_MIN_BROKER_STOP_BUFFER_POINTS", 5.0)

# --- Intrabar ---
# New entries only after a candle closes; management runs every loop pass.
INTRABAR_ENTRY_TIMEFRAME = os.getenv("TM_INTRABAR_ENTRY_TIMEFRAME", "H1")


# =============================================================================
# LAYER 2 - Unified exit
# =============================================================================

# --- Hard override: signal flip ---
SIGNAL_FLIP_ENABLED = _b("TM_SIGNAL_FLIP_ENABLED", True)
# The opposing signal must be fully confirmed before it can override everything.
SIGNAL_FLIP_MIN_SCORE = _f("TM_SIGNAL_FLIP_MIN_SCORE", 45.0)
SIGNAL_FLIP_MIN_CONFIDENCE = _f("TM_SIGNAL_FLIP_MIN_CONFIDENCE", 0.60)
# Flip must also be aligned across timeframes to count as "confirmed".
SIGNAL_FLIP_REQUIRE_MTF_ALIGNED = _b("TM_SIGNAL_FLIP_REQUIRE_MTF_ALIGNED", True)

# --- Soft exit: unified Exit Score ---
EXIT_SCORE_THRESHOLD = _f("TM_EXIT_SCORE_THRESHOLD", 0.75)

# Component weights.
#
# The original design allocated 15% to a "volume weakness" component. There is
# no volume field that is present on every candle regardless of data path:
# QuantDinger's indicator endpoint returns rsi/atr/macd/ma_trend/close only,
# while the MT5 fallback exposes tick_volume. Mixing the two would make the
# component's meaning depend on which source happened to answer.
#
# That 15% is therefore redistributed proportionally across the remaining three
# components (40/25/20 -> 47.06/29.41/23.53, preserving their relative sizes).
#
# TO RE-ENABLE VOLUME: once a volume field is available on every candle from
# every data path, set EXIT_WEIGHT_VOLUME_WEAKNESS to 0.15 and restore the other
# three to 0.40 / 0.25 / 0.20. The scorer already normalises weights, so no
# other change is required beyond supplying `volume_weakness` in the inputs.
EXIT_WEIGHT_PROBABILITY = _f("TM_EXIT_WEIGHT_PROBABILITY", 0.4706)
EXIT_WEIGHT_TREND_REVERSAL = _f("TM_EXIT_WEIGHT_TREND_REVERSAL", 0.2941)
EXIT_WEIGHT_MOMENTUM_WEAKNESS = _f("TM_EXIT_WEIGHT_MOMENTUM_WEAKNESS", 0.2353)
EXIT_WEIGHT_VOLUME_WEAKNESS = _f("TM_EXIT_WEIGHT_VOLUME_WEAKNESS", 0.0)

# --- Probability qualification ---
# A single probability reading is never a strong signal on its own. It only
# contributes at full strength when one of these is true.
PROB_MIN_CONSECUTIVE_DECLINES = _i("TM_PROB_MIN_CONSECUTIVE_DECLINES", 2)
PROB_DROP_FROM_ENTRY_PCT = _f("TM_PROB_DROP_FROM_ENTRY_PCT", 0.30)
# Weight multiplier applied when the probability reading is NOT yet qualified.
PROB_UNQUALIFIED_DAMPING = _f("TM_PROB_UNQUALIFIED_DAMPING", 0.25)

# --- Exit model (ML) ---
# Production stays False. Shadow mode computes and logs the probability without
# ever letting it influence a decision.
ML_EXIT_ENABLED = _b("TM_ML_EXIT_ENABLED", False)
ML_EXIT_SHADOW_MODE = _b("TM_ML_EXIT_SHADOW_MODE", False)
ML_EXIT_MODEL_PATH = os.getenv("TM_ML_EXIT_MODEL_PATH", "models/exit/exit_model.json")


# =============================================================================
# LAYER 3 - Adaptive trailing (also the target-extension mechanism)
# =============================================================================

ADAPTIVE_TRAILING_ENABLED = _b("TM_ADAPTIVE_TRAILING_ENABLED", True)
# Trailing only arms once the trade has this much open profit, in R.
TRAILING_ACTIVATE_R = _f("TM_TRAILING_ACTIVATE_R", 1.0)

# Distance = ATR_now * base * trend_factor * volatility_factor, clamped.
TRAILING_BASE_ATR_MULTIPLIER = _f("TM_TRAILING_BASE_ATR_MULTIPLIER", 2.0)
TRAILING_MIN_ATR_MULTIPLIER = _f("TM_TRAILING_MIN_ATR_MULTIPLIER", 0.8)
TRAILING_MAX_ATR_MULTIPLIER = _f("TM_TRAILING_MAX_ATR_MULTIPLIER", 4.0)

# Trend strength (0..100) maps linearly to a widening factor.
# strength <= LOW  -> TIGHT factor ; strength >= HIGH -> WIDE factor
TRAILING_TREND_LOW = _f("TM_TRAILING_TREND_LOW", 25.0)
TRAILING_TREND_HIGH = _f("TM_TRAILING_TREND_HIGH", 70.0)
TRAILING_TREND_TIGHT_FACTOR = _f("TM_TRAILING_TREND_TIGHT_FACTOR", 0.7)
TRAILING_TREND_WIDE_FACTOR = _f("TM_TRAILING_TREND_WIDE_FACTOR", 1.5)

# Volatility expansion/contraction relative to ATR at entry.
# ratio = atr_now / atr_at_entry, clamped to this band then applied directly.
TRAILING_VOL_RATIO_MIN = _f("TM_TRAILING_VOL_RATIO_MIN", 0.6)
TRAILING_VOL_RATIO_MAX = _f("TM_TRAILING_VOL_RATIO_MAX", 1.8)

# MAE/MFE calibration: the normal pullback from peak (as a fraction of MFE)
# before a retreat should be treated as a reversal. Used only to tune trailing
# sensitivity — it never emits an exit decision of its own.
MFE_PULLBACK_TOLERANCE = _f("TM_MFE_PULLBACK_TOLERANCE", 0.40)
MFE_CALIBRATION_MIN_SAMPLES = _i("TM_MFE_CALIBRATION_MIN_SAMPLES", 20)
# How strongly the historical pullback statistic may stretch the trail distance.
MFE_CALIBRATION_MAX_ADJUST = _f("TM_MFE_CALIBRATION_MAX_ADJUST", 0.30)


# =============================================================================
# LAYER 4 - Trade age management (Time Stop is an internal condition here)
# =============================================================================

TRADE_AGE_ENABLED = _b("TM_TRADE_AGE_ENABLED", True)

# Phase boundaries in closed candles since entry.
AGE_PHASE_SETTLE_MAX_BARS = _i("TM_AGE_PHASE_SETTLE_MAX_BARS", 5)     # 0-5   no big changes
AGE_PHASE_TRAIL_MAX_BARS = _i("TM_AGE_PHASE_TRAIL_MAX_BARS", 12)      # 6-12  trailing ramps in
AGE_PHASE_TIGHTEN_MAX_BARS = _i("TM_AGE_PHASE_TIGHTEN_MAX_BARS", 15)  # 12-15 tighten

# Trailing multiplier scaling per phase (applied on top of Layer 3's distance).
AGE_SETTLE_TRAIL_SCALE = _f("TM_AGE_SETTLE_TRAIL_SCALE", 1.0)
AGE_TRAIL_SCALE_START = _f("TM_AGE_TRAIL_SCALE_START", 1.0)
AGE_TRAIL_SCALE_END = _f("TM_AGE_TRAIL_SCALE_END", 0.8)
AGE_TIGHTEN_SCALE_START = _f("TM_AGE_TIGHTEN_SCALE_START", 0.8)
AGE_TIGHTEN_SCALE_END = _f("TM_AGE_TIGHTEN_SCALE_END", 0.5)

# Time-stop condition: past this many bars with less than this profit -> close.
TIME_STOP_MIN_BARS = _i("TM_TIME_STOP_MIN_BARS", 10)
TIME_STOP_MAX_BARS = _i("TM_TIME_STOP_MAX_BARS", 15)
TIME_STOP_MIN_PROFIT_R = _f("TM_TIME_STOP_MIN_PROFIT_R", 0.3)


# =============================================================================
# LAYER 5 - Partial take profit
# =============================================================================

PARTIAL_TP_ENABLED = _b("TM_PARTIAL_TP_ENABLED", True)

# Ladder: (profit_in_R, fraction_of_ORIGINAL_volume_to_close)
# +1R is break-even only (handled by Layer 1) and closes nothing.
PARTIAL_TP_LADDER = (
    (_f("TM_PARTIAL_L1_R", 2.0), _f("TM_PARTIAL_L1_FRACTION", 0.30)),
    (_f("TM_PARTIAL_L2_R", 3.0), _f("TM_PARTIAL_L2_FRACTION", 0.30)),
)
# Broker minimum lot; a partial smaller than this is skipped rather than sent.
MIN_PARTIAL_VOLUME = _f("TM_MIN_PARTIAL_VOLUME", 0.01)
# Never close so much that the remainder falls below the minimum lot.
MIN_REMAINING_VOLUME = _f("TM_MIN_REMAINING_VOLUME", 0.01)


# =============================================================================
# LAYER 6 - Trade profile (config selector, not decision logic)
# =============================================================================

DEFAULT_PROFILE = os.getenv("TM_DEFAULT_PROFILE", "trend")

# Per-profile overrides applied to layers 2-5 at entry time, once.
# Any key omitted keeps the module-level default above.
PROFILE_OVERRIDES = {
    "trend": {
        # Wide trailing, no fixed target — the trail is the exit.
        "TRAILING_BASE_ATR_MULTIPLIER": 2.5,
        "TRAILING_MAX_ATR_MULTIPLIER": 5.0,
        "USE_FIXED_TP": False,
        "EXIT_SCORE_THRESHOLD": 0.80,
        "TIME_STOP_MAX_BARS": 20,
    },
    "breakout": {
        "TRAILING_BASE_ATR_MULTIPLIER": 2.0,
        "TRAILING_MAX_ATR_MULTIPLIER": 4.0,
        "USE_FIXED_TP": False,
        "EXIT_SCORE_THRESHOLD": 0.75,
        "TIME_STOP_MAX_BARS": 15,
    },
    "mean_reversion": {
        # Tight trailing, fixed target, quick exit.
        "TRAILING_BASE_ATR_MULTIPLIER": 1.2,
        "TRAILING_MAX_ATR_MULTIPLIER": 2.0,
        "USE_FIXED_TP": True,
        "EXIT_SCORE_THRESHOLD": 0.65,
        "TIME_STOP_MAX_BARS": 10,
        "ATR_TP_BASE_MULTIPLIER": 1.8,
    },
    "range": {
        # Short targets.
        "TRAILING_BASE_ATR_MULTIPLIER": 1.0,
        "TRAILING_MAX_ATR_MULTIPLIER": 1.8,
        "USE_FIXED_TP": True,
        "EXIT_SCORE_THRESHOLD": 0.70,
        "TIME_STOP_MAX_BARS": 8,
        "ATR_TP_BASE_MULTIPLIER": 1.5,
    },
}

# Default when a profile does not specify it.
USE_FIXED_TP = _b("TM_USE_FIXED_TP", True)

VALID_PROFILES = tuple(PROFILE_OVERRIDES.keys())
