"""Causal indicator formulas for the research contract.

Causal by construction: every function reads only ``.rolling()`` / ``.ewm()``
/ ``.shift(positive)`` windows that END at the current row. There is no
negative shift and no centred window anywhere in this module, so a value at
row ``i`` cannot depend on row ``i+1``. ``tests/test_research_causality.py``
proves it by mutating the future and asserting nothing upstream moves.

These are transcriptions of the research repo's frozen formulas
(``src/features/indicators.py``), not re-derivations. Where a choice exists —
Wilder vs simple smoothing, ``ddof=0`` vs ``ddof=1``, how a zero denominator
is handled — the research choice is reproduced exactly, because parity with
the artifact that was trained is the only thing that makes a prediction mean
anything. ``tests/test_research_golden_parity.py`` pins it bit-for-bit.

This is deliberately NOT the same arithmetic as
``analysis/features/live_parity_features.py``, which mirrors the legacy live
path (simple-average RSI, SMA-based MACD). Two different models, two
different vocabularies; mapping one onto the other by name would be exactly
the units/scale error the integration exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int,
        method: str = "wilder") -> pd.Series:
    """Wilder ATR: EWM of True Range with alpha = 1/period, adjust=False."""
    tr = true_range(high, low, close)
    if method == "sma":
        return tr.rolling(window=period, min_periods=period).mean()
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI.

    The two degenerate branches are part of the contract, not defensive
    padding: an all-gain window is RSI 100, and a completely flat window is
    RSI 50 rather than NaN. Both are reproduced from the research formula.
    """
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


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period, min_periods=period).mean()


def directional_movement(high: pd.Series, low: pd.Series, close: pd.Series,
                         period: int) -> tuple:
    """Wilder's directional movement system -> (adx, plus_di, minus_di)."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=high.index)
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = (100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
               / atr_.replace(0.0, np.nan))
    minus_di = (100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
                / atr_.replace(0.0, np.nan))
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di


def trend_strength(close: pd.Series, ema_fast_period: int, ema_slow_period: int) -> pd.Series:
    """(EMA_fast - EMA_slow) / close.

    Note the division by close: this is what makes it scale-free, and it is
    the whole difference between this and a raw MACD line. The legacy KAIROS
    `trend_strength` is an entirely different quantity — a bucketed encoding
    of an MTF agreement string — and the two must never be substituted for
    each other.
    """
    return (ema(close, ema_fast_period) - ema(close, ema_slow_period)) / close.replace(0.0, np.nan)


def trend_score(close: pd.Series, lookback: int) -> pd.Series:
    """OLS slope of close over the trailing window, divided by close."""
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
    """close.pct_change(lookback).

    Unlike the legacy KAIROS `momentum_score` — a three-bucket re-encoding of
    RSI, which the legacy contract itself flags as redundant capacity rather
    than a second dimension — this is a plain multi-bar return and carries
    information RSI does not.
    """
    return close.pct_change(periods=lookback)


def volatility_score(atr_series: pd.Series, lookback: int) -> pd.Series:
    rolling_mean_atr = atr_series.rolling(window=lookback, min_periods=lookback).mean()
    return atr_series / rolling_mean_atr.replace(0.0, np.nan)


def market_regime(adx_series: pd.Series, trend_threshold: float) -> pd.Series:
    """1.0 = TRENDING (ADX >= threshold), 0.0 = RANGING.

    Two states, not the legacy path's four. A model trained on this binary
    cannot read the legacy 4-value REGIME_ENCODING, which is why the two
    contracts keep separate `market_regime` columns.
    """
    return (adx_series >= trend_threshold).astype(float)


def direction_indicator(close: pd.Series, ema_period: int) -> pd.Series:
    """+1 above the EMA, -1 below, 0 exactly on it."""
    return np.sign(close - ema(close, ema_period))


def bollinger_bands(close: pd.Series, period: int, num_std: float) -> pd.DataFrame:
    middle = close.rolling(window=period, min_periods=period).mean()
    # ddof=0 (population sd) is the Bollinger convention and what the research
    # datasets were built with. ddof=1 would shift every band slightly and
    # break golden parity.
    sd = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * sd
    lower = middle - num_std * sd
    span = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": (upper - lower) / middle.replace(0.0, np.nan),
        # Deliberately NOT clipped: <0 or >1 means price left the band, which
        # is information, and clipping would destroy it.
        "bb_percent_b": (close - lower) / span,
    }, index=close.index)


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int, d_period: int, smooth: int) -> pd.DataFrame:
    """Slow stochastic: %K is the smoothed raw %K, %D is the SMA of %K."""
    lowest = low.rolling(window=k_period, min_periods=k_period).min()
    highest = high.rolling(window=k_period, min_periods=k_period).max()
    span = (highest - lowest).replace(0.0, np.nan)
    raw_k = 100.0 * (close - lowest) / span
    k = raw_k.rolling(window=smooth, min_periods=smooth).mean()
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return pd.DataFrame({"stochastic_k": k, "stochastic_d": d}, index=close.index)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Commodity Channel Index. Lambert's 0.015 is fixed, not configurable."""
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda w: np.abs(w - w.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad.replace(0.0, np.nan))


def log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    """ln(close_i / close_{i-periods}).

    Time-additive, unlike pct_change. A non-positive close -- impossible for
    these instruments, but a corrupt row must never fabricate an infinity --
    yields NaN rather than +/-inf.
    """
    ratio = close / close.shift(periods=periods).replace(0.0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.log(ratio)
    return out.replace([np.inf, -np.inf], np.nan)


# --- session clock ----------------------------------------------------------
# Pure functions of the row's OWN timestamp. They never touch OHLC, so they
# cannot look ahead under any circumstances.

def session_indicator(timestamp_session_tz: pd.Series, sessions_cfg: dict) -> pd.DataFrame:
    hour_frac = timestamp_session_tz.dt.hour + timestamp_session_tz.dt.minute / 60.0
    out = {}
    for name, (start_str, end_str) in sessions_cfg.items():
        sh, sm = (int(x) for x in start_str.split(":"))
        eh, em = (int(x) for x in end_str.split(":"))
        start, end = sh + sm / 60.0, eh + em / 60.0
        # A window that wraps midnight is a union, not an empty intersection.
        mask = ((hour_frac >= start) & (hour_frac < end)) if start <= end \
            else ((hour_frac >= start) | (hour_frac < end))
        out[f"session_{name}"] = mask.astype(float)
    return pd.DataFrame(out, index=timestamp_session_tz.index)


def hour_of_day(timestamp_session_tz: pd.Series) -> pd.Series:
    return timestamp_session_tz.dt.hour.astype("float64")


def day_of_week(timestamp_session_tz: pd.Series) -> pd.Series:
    return timestamp_session_tz.dt.dayofweek.astype("float64")


def minute_of_day(timestamp_session_tz: pd.Series) -> pd.Series:
    return (timestamp_session_tz.dt.hour * 60 + timestamp_session_tz.dt.minute).astype("float64")


def minutes_from_session_open(timestamp_session_tz: pd.Series,
                              trading_day_open_utc: str) -> pd.Series:
    """Minutes since this instrument's trading day opened, mod 1440.

    A fixed-UTC approximation: on DST-split days the true open shifts by an
    hour. Inherited from the research contract deliberately and stated rather
    than silently corrected — a second DST engine on the KAIROS side would be
    a new divergence, and divergence is what this integration is preventing.
    """
    oh, om = (int(x) for x in trading_day_open_utc.split(":"))
    return ((minute_of_day(timestamp_session_tz) - (oh * 60 + om)) % 1440).astype("float64")
