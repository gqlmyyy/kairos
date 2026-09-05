"""Price-action features. All causal: rolling windows end at the current row."""
from __future__ import annotations

import numpy as np
import pandas as pd


def returns(close: pd.Series, periods: list[int]) -> pd.DataFrame:
    return pd.DataFrame({f"return_{p}": close.pct_change(periods=p) for p in periods}, index=close.index)


def normalized_returns(close: pd.Series, atr_series: pd.Series) -> pd.Series:
    return close.diff() / atr_series.replace(0.0, np.nan)


def candle_range(high: pd.Series, low: pd.Series) -> pd.Series:
    return high - low


def body_size(open_: pd.Series, close: pd.Series) -> pd.Series:
    return (close - open_).abs()


def upper_wick(open_: pd.Series, high: pd.Series, close: pd.Series) -> pd.Series:
    return high - pd.concat([open_, close], axis=1).max(axis=1)


def lower_wick(open_: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat([open_, close], axis=1).min(axis=1) - low


def distance_from_recent_high(close: pd.Series, high: pd.Series, lookback: int) -> pd.Series:
    rolling_high = high.rolling(window=lookback, min_periods=lookback).max()
    return (close - rolling_high) / close.replace(0.0, np.nan)


def distance_from_recent_low(close: pd.Series, low: pd.Series, lookback: int) -> pd.Series:
    rolling_low = low.rolling(window=lookback, min_periods=lookback).min()
    return (close - rolling_low) / close.replace(0.0, np.nan)


def rolling_volatility(close: pd.Series, lookback: int) -> pd.Series:
    return close.pct_change().rolling(window=lookback, min_periods=lookback).std()


# ---------------------------------------------------------------------------
# Extended price action (feature_schema 1.1.0). Appended below the originals,
# which are left exactly as they were -- the ATR-normalised variants are
# additions, not replacements.
# ---------------------------------------------------------------------------

def normalize_by_atr(series: pd.Series, atr_series: pd.Series) -> pd.Series:
    """Express a price-unit quantity in ATR units, so the value does not
    depend on the instrument's absolute price level (a 5-dollar range means
    something very different for XAUUSD than for EURUSD)."""
    return series / atr_series.replace(0.0, np.nan)


def return_acceleration(fast_return: pd.Series, slow_return: pd.Series) -> pd.Series:
    """Change in momentum: how much the shorter-horizon return exceeds the
    longer-horizon one. Both inputs are already causal pct_change series."""
    return fast_return - slow_return


def return_in_atr(close: pd.Series, period: int, atr_series: pd.Series) -> pd.Series:
    """Price change over `period` bars measured in ATR units, rather than
    as a percentage. Uses only rows <= i (`diff` looks backwards)."""
    return close.diff(periods=period) / atr_series.replace(0.0, np.nan)


def daily_structure(timestamp_session_tz: pd.Series, high: pd.Series, low: pd.Series,
                    close: pd.Series, atr_series: pd.Series) -> pd.DataFrame:
    """Intraday position relative to the CURRENT day's range so far.

    Causality is the whole point here: the day's high/low are running
    (`cummax`/`cummin` within the day), so at row i they reflect only bars
    up to and including i. The day's FINAL high/low -- which would be
    look-ahead -- are never used. Day boundaries come from the project's
    existing `session_timezone`, not from UTC or the machine's locale.

    `position_in_day_range` falls back to 0.5 (mid-range) on the degenerate
    first-bar-of-day case where high == low, rather than emitting NaN, which
    would drop the first bar of every trading day from the dataset.
    """
    day_key = timestamp_session_tz.dt.normalize()
    day_high = high.groupby(day_key).cummax()
    day_low = low.groupby(day_key).cummin()
    day_range = day_high - day_low
    safe_close = close.replace(0.0, np.nan)
    span = day_range.replace(0.0, np.nan)
    return pd.DataFrame({
        "distance_from_day_high": (close - day_high) / safe_close,
        "distance_from_day_low": (close - day_low) / safe_close,
        "day_range": day_range,
        "day_range_atr": day_range / atr_series.replace(0.0, np.nan),
        "position_in_day_range": ((close - day_low) / span).fillna(0.5),
    }, index=close.index)


# ---------------------------------------------------------------------------
# Market-data features (Phase 3). `real_volume` is deliberately absent: it is
# identically zero across every dataset in this repository (verified directly
# against data/raw/mt5 during the Phase 3 audit), so it can never be a
# feature. `spread` IS present and mostly populated, so a minimal, normalized
# spread block is provided -- see the reliability notes in the Phase 3 report
# (EURUSD H1/H4 carry many literal-zero spreads; they are kept as real
# observations, never interpolated away).
# ---------------------------------------------------------------------------

def spread_relative(spread: pd.Series, lookback: int,
                    zero_median_policy: str = "UNIT_WHEN_ALSO_ZERO") -> pd.Series:
    """Current spread divided by its trailing median -- dimensionless, so it
    compares instruments and spread regimes. Causal: the median window ends
    at the current row (inclusive).

    ZERO-MEDIAN REPAIR (research rebuild). Several feeds in this project
    report a literal 0 spread for long stretches -- measured share of
    zero-spread bars: EURUSD 50-74%, GBPUSD 18-48%, XAUUSD 5-13%. Whenever the
    trailing median is 0 the original formula divided by NaN, and because the
    dataset builder drops any row with a single non-finite feature, ONE
    optional spread column silently deleted the majority of EURUSD's history
    in multi-month contiguous blocks. That is not a data problem and not a
    modelling choice -- it is an unintended row filter, and it is the origin
    of the island-shaped eligible population documented in
    data/reports/research_rebuild/.

    Policies:
      UNIT_WHEN_ALSO_ZERO (default) -- median 0 AND spread 0 means the current
          spread IS the typical spread, so the ratio is 1.0 by definition.
          Median 0 with a positive spread stays NaN: that ratio is genuinely
          unbounded and must not be invented.
      STRICT -- the pre-repair behaviour (always NaN on a zero median), kept
          so the frozen Phase 5/6/7 artifacts remain reproducible.
    """
    if zero_median_policy not in ("UNIT_WHEN_ALSO_ZERO", "STRICT"):
        raise ValueError(f"unknown zero_median_policy: {zero_median_policy!r}")
    med = spread.rolling(window=lookback, min_periods=lookback).median()
    out = spread / med.replace(0.0, np.nan)
    if zero_median_policy == "UNIT_WHEN_ALSO_ZERO":
        both_zero = (med == 0.0) & (spread == 0.0)
        out = out.mask(both_zero, 1.0)
    return out


def spread_mean(spread: pd.Series, lookback: int) -> pd.Series:
    """Trailing mean spread over `lookback` bars, causal."""
    return spread.rolling(window=lookback, min_periods=lookback).mean()


def atr_percent(atr_series: pd.Series, close: pd.Series) -> pd.Series:
    """ATR normalized by price level (a.k.a. ATR%), dimensionless, so
    volatility compares across instruments. Causal by construction."""
    return atr_series / close.replace(0.0, np.nan)
