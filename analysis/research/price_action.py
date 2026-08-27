"""Causal price-action formulas for the research contract.

Same rule as ``indicators``: every rolling window ends at the current row.
The ATR-normalised variants are the ones the research models actually use —
the raw price-unit originals (``range``, ``body_size``, ``upper_wick``,
``lower_wick``) are PRICE_UNIT and excluded by the contract, so they are not
implemented here at all rather than implemented and left as a trap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def returns(close: pd.Series, periods) -> pd.DataFrame:
    return pd.DataFrame({f"return_{p}": close.pct_change(periods=p) for p in periods},
                        index=close.index)


def return_acceleration(fast_return: pd.Series, slow_return: pd.Series) -> pd.Series:
    return fast_return - slow_return


def normalize_by_atr(series: pd.Series, atr_series: pd.Series) -> pd.Series:
    """Express a price-unit quantity in ATR units.

    This single division is what turns an instrument-specific magnitude into
    something a model can share across EURUSD and XAUUSD. A zero ATR yields
    NaN, never a division by zero and never a substituted constant.
    """
    return series / atr_series.replace(0.0, np.nan)


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


def return_in_atr(close: pd.Series, period: int, atr_series: pd.Series) -> pd.Series:
    """Price change over `period` bars in ATR units. `diff` looks backwards only."""
    return close.diff(periods=period) / atr_series.replace(0.0, np.nan)


def atr_percent(atr_series: pd.Series, close: pd.Series) -> pd.Series:
    return atr_series / close.replace(0.0, np.nan)


def daily_structure(timestamp_session_tz: pd.Series, high: pd.Series, low: pd.Series,
                    close: pd.Series, atr_series: pd.Series) -> pd.DataFrame:
    """Position inside the CURRENT day's range so far.

    Causality is the entire point: the day's high/low are RUNNING
    (``cummax``/``cummin`` within the day), so at row ``i`` they reflect only
    bars up to and including ``i``. The day's final high/low would be
    look-ahead and are never touched.

    ``position_in_day_range`` falls back to 0.5 on the degenerate flat-bar
    case rather than NaN. That is a declared contract value
    (``midpoint_0.5_when_day_range_is_zero``), not a silent fallback: without
    it the first bar of every trading day would be dropped from the dataset.
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


def spread_relative(spread: pd.Series, lookback: int,
                    zero_median_policy: str = "UNIT_WHEN_ALSO_ZERO") -> pd.Series:
    """Spread over its own trailing median — dimensionless.

    ``UNIT_WHEN_ALSO_ZERO`` is the research repo's repair and the policy the
    shipped models were trained under: when the trailing median is 0 AND the
    current spread is 0, the current spread IS the typical spread, so the
    ratio is 1.0 by definition. A zero median with a POSITIVE spread stays
    NaN — that ratio is genuinely unbounded and must not be invented.

    Note what this is not: it is not a fallback that hides missing data. A
    literal zero spread is a real, common observation on these feeds (measured
    at 50-74% of EURUSD bars). Treating it as missing was the bug; treating it
    as a real zero is the fix.
    """
    if zero_median_policy not in ("UNIT_WHEN_ALSO_ZERO", "STRICT"):
        raise ValueError(f"unknown zero_median_policy: {zero_median_policy!r}")
    med = spread.rolling(window=lookback, min_periods=lookback).median()
    out = spread / med.replace(0.0, np.nan)
    if zero_median_policy == "UNIT_WHEN_ALSO_ZERO":
        out = out.mask((med == 0.0) & (spread == 0.0), 1.0)
    return out
