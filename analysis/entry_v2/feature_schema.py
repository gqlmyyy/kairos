from __future__ import annotations

"""analysis/entry_v2/feature_schema.py

Feature schema for Entry v2 (feature generation ONLY).

This module defines:
- canonical feature names
- a helper to validate feature list (no duplicates)

The feature list is deterministic and must be identical between:
- feature_engineering.py output columns
- training/inference later (not part of this task)
"""

from typing import List


# Canonical feature ordering
FEATURE_COLUMNS: List[str] = [
    # Timeframe-specific indicators
    # RSI(14), ATR(14), ADX(14), MACD, EMA/SMA, Momentum, CCI, Stochastic, Bollinger Width

    # H4
    "h4_rsi_14",
    "h4_atr_14",
    "h4_adx_14",
    "h4_macd",
    "h4_macd_signal",
    "h4_macd_hist",
    "h4_ema_12",
    "h4_ema_26",
    "h4_ema_50",
    "h4_sma_20",
    "h4_sma_50",
    "h4_sma_200",
    "h4_momentum",
    "h4_cci_20",
    "h4_stoch_k_14",
    "h4_stoch_d_14",
    "h4_bollinger_width_20",

    # H1
    "h1_rsi_14",
    "h1_atr_14",
    "h1_adx_14",
    "h1_macd",
    "h1_macd_signal",
    "h1_macd_hist",
    "h1_ema_12",
    "h1_ema_26",
    "h1_ema_50",
    "h1_sma_20",
    "h1_sma_50",
    "h1_sma_200",
    "h1_momentum",
    "h1_cci_20",
    "h1_stoch_k_14",
    "h1_stoch_d_14",
    "h1_bollinger_width_20",





    # Lag features (1/2/3) for selected base indicators
    # RSI/ATR/ADX/MACD/Momentum

    "lag1_rsi_14",
    "lag2_rsi_14",
    "lag3_rsi_14",
    "lag1_atr_14",
    "lag2_atr_14",
    "lag3_atr_14",
    "lag1_adx_14",
    "lag2_adx_14",
    "lag3_adx_14",
    "lag1_macd",
    "lag2_macd",
    "lag3_macd",
    "lag1_momentum",
    "lag2_momentum",
    "lag3_momentum",

    # Delta features (t - t-1)
    "delta_rsi_14",
    "delta_atr_14",
    "delta_adx_14",
    "delta_macd",
    "delta_momentum",

    # Interaction features
    "rsi_x_adx",
    "atr_x_trend",
    "macd_x_momentum",
    "rsi_x_atr",
    "trend_x_session",
    "volatility_x_momentum",

    # Trend agreement features across timeframes (simple categorical agreements encoded as float)
    "trend_agree_h4_h1",

    # Session/time features
    "day_of_week",
    "hour",
    "session_encoded",

    # Symbol encoding
    "symbol_encoded",

    # NOTE: No non-numeric/to-documentation columns here (e.g., label_reason) or identifiers (symbol).
]


def validate_feature_columns(columns: List[str] | None = None) -> None:
    cols = columns or FEATURE_COLUMNS
    if not cols:
        raise ValueError("feature columns empty")
    if len(cols) != len(set(cols)):
        dup = [c for c in cols if cols.count(c) > 1]
        raise RuntimeError(f"Duplicated feature columns found: {sorted(set(dup))[:10]}")

