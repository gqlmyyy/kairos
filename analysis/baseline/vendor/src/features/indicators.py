"""Baseline technical indicators. Every function here is causal by
construction: it only uses `.rolling()`/`.ewm()`/`.shift(positive)` over rows
up to and including the current index, never a negative shift or any
look-ahead. See FEATURE_CONTRACT.md for exact formulas."""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int, method: str = "wilder") -> pd.Series:
    tr = true_range(high, low, close)
    if method == "sma":
        return tr.rolling(window=period, min_periods=period).mean()
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
    return out


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def directional_movement(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di


def trend_strength(close: pd.Series, ema_fast_period: int, ema_slow_period: int) -> pd.Series:
    ema_f = ema(close, ema_fast_period)
    ema_s = ema(close, ema_slow_period)
    return (ema_f - ema_s) / close.replace(0.0, np.nan)


def trend_score(close: pd.Series, lookback: int) -> pd.Series:
    """Normalized linear-regression slope of close over the trailing window."""
    x = np.arange(lookback, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(window: np.ndarray) -> float:
        y = window
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        return cov / x_var if x_var > 0 else 0.0

    slope = close.rolling(window=lookback, min_periods=lookback).apply(_slope, raw=True)
    return slope / close.replace(0.0, np.nan)


def momentum_score(close: pd.Series, lookback: int) -> pd.Series:
    return close.pct_change(periods=lookback)


def volatility_score(atr_series: pd.Series, lookback: int) -> pd.Series:
    rolling_mean_atr = atr_series.rolling(window=lookback, min_periods=lookback).mean()
    return atr_series / rolling_mean_atr.replace(0.0, np.nan)


def market_regime(adx_series: pd.Series, trend_threshold: float) -> pd.Series:
    """1.0 = TRENDING (ADX >= threshold), 0.0 = RANGING."""
    return (adx_series >= trend_threshold).astype(float)


def session_indicator(timestamp_session_tz: pd.Series, sessions_cfg: dict) -> pd.DataFrame:
    """One-hot session membership columns computed purely from the row's own
    timestamp (session_timezone), with no dependency on other rows."""
    hour_frac = timestamp_session_tz.dt.hour + timestamp_session_tz.dt.minute / 60.0
    out = {}
    for name, (start_str, end_str) in sessions_cfg.items():
        sh, sm = (int(x) for x in start_str.split(":"))
        eh, em = (int(x) for x in end_str.split(":"))
        start = sh + sm / 60.0
        end = eh + em / 60.0
        if start <= end:
            mask = (hour_frac >= start) & (hour_frac < end)
        else:
            mask = (hour_frac >= start) | (hour_frac < end)
        out[f"session_{name}"] = mask.astype(float)
    return pd.DataFrame(out, index=timestamp_session_tz.index)


def direction_indicator(close: pd.Series, ema_period: int) -> pd.Series:
    """+1 close above EMA (bullish bias), -1 below, 0 exactly on it."""
    ema_ = ema(close, ema_period)
    return np.sign(close - ema_)


# ---------------------------------------------------------------------------
# Extended indicators (feature_schema 1.1.0). Appended, never reordering the
# functions above -- existing feature positions must stay stable.
# ---------------------------------------------------------------------------

def bollinger_bands(close: pd.Series, period: int, num_std: float) -> pd.DataFrame:
    """Classic Bollinger Bands, all causal (rolling windows end at row i).

    `bb_width` is normalised by the middle band so it is comparable across
    instruments and price levels; `bb_percent_b` places close inside the
    band as 0..1 (outside the band is <0 or >1, which is meaningful and is
    deliberately not clipped).
    """
    middle = close.rolling(window=period, min_periods=period).mean()
    # ddof=0 (population sd) is the standard Bollinger convention.
    sd = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * sd
    lower = middle - num_std * sd
    span = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": (upper - lower) / middle.replace(0.0, np.nan),
        "bb_percent_b": (close - lower) / span,
    }, index=close.index)


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int, d_period: int, smooth: int) -> pd.DataFrame:
    """Slow stochastic oscillator, causal.

    raw %K = 100 * (close - LL(k_period)) / (HH(k_period) - LL(k_period))
    %K     = SMA(raw %K, smooth)      <- "slow" %K
    %D     = SMA(%K, d_period)
    """
    lowest = low.rolling(window=k_period, min_periods=k_period).min()
    highest = high.rolling(window=k_period, min_periods=k_period).max()
    span = (highest - lowest).replace(0.0, np.nan)
    raw_k = 100.0 * (close - lowest) / span
    k = raw_k.rolling(window=smooth, min_periods=smooth).mean()
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return pd.DataFrame({"stochastic_k": k, "stochastic_d": d}, index=close.index)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Commodity Channel Index, causal.

    CCI = (TP - SMA(TP, n)) / (0.015 * mean_absolute_deviation(TP, n)),
    TP = (high + low + close) / 3. The 0.015 constant is Lambert's original
    scaling and is intentionally not configurable.
    """
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda w: np.abs(w - w.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad.replace(0.0, np.nan))


# ---------------------------------------------------------------------------
# Phase 3 baseline completions (feature_schema 1.2.0). Appended below every
# pre-existing function -- nothing above may move, for the same column-order
# reason documented in engine.py.
# ---------------------------------------------------------------------------

def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average, causal trailing window ending at the current row."""
    return close.rolling(window=period, min_periods=period).mean()


def log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    """Log return over `periods` bars: ln(close_i / close_{i-periods}).

    Time-additive (multi-bar log returns sum), unlike pct_change. Uses only
    rows <= i. A non-positive close (impossible for these instruments, but a
    corrupt row must never fabricate inf) yields NaN rather than +/-inf."""
    prev = close.shift(periods=periods)
    ratio = close / prev.replace(0.0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.log(ratio)
    return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Session-clock features (Phase 3). Pure functions of the row's OWN timestamp,
# expressed in the project's session_timezone. They never touch OHLC and can
# therefore never look ahead.
# ---------------------------------------------------------------------------

def hour_of_day(timestamp_session_tz: pd.Series) -> pd.Series:
    """UTC-normalized hour-of-day in [0, 23] in the session timezone."""
    return timestamp_session_tz.dt.hour.astype("float64")


def day_of_week(timestamp_session_tz: pd.Series) -> pd.Series:
    """Monday=0 .. Sunday=6 in the session timezone."""
    return timestamp_session_tz.dt.dayofweek.astype("float64")


def minute_of_day(timestamp_session_tz: pd.Series) -> pd.Series:
    """Minutes since local midnight in the session timezone, in [0, 1439]."""
    tod = timestamp_session_tz.dt.hour * 60 + timestamp_session_tz.dt.minute
    return tod.astype("float64")


def minutes_from_session_open(
    timestamp_session_tz: pd.Series, trading_day_open_utc: str
) -> pd.Series:
    """Minutes elapsed since this instrument's trading day opened.

    `trading_day_open_utc` is the DOMINANT empirically observed daily open
    for THIS symbol (Phase 2 observed-trading-hours evidence: FX majors open
    00:00 UTC, XAUUSD opens 01:00 UTC). It is a fixed-UTC approximation: the
    DST-split days documented in the Phase 2 audit shift the true open by an
    hour, which this deliberately accepts and the Phase 3 report records as a
    limitation -- inventing a second DST engine for one feature was judged
    worse than an honest, documented approximation.

    Defined for every bar via wrap-around modulo 1440 (a bar before the open
    belongs to the previous trading day's cycle)."""
    oh, om = (int(x) for x in trading_day_open_utc.split(":"))
    open_minutes = oh * 60 + om
    mod = (minute_of_day(timestamp_session_tz) - open_minutes) % 1440
    return mod.astype("float64")
