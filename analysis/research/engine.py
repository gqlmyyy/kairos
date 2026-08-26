"""The canonical feature engine: OHLC candles -> the model's feature vector.

One calculator, every consumer
------------------------------
Offline inference, replay, golden-parity tests and any future live path all
call :func:`build_feature_frame`. There is no second implementation and no
"live shortcut" — the shortcut is what let the legacy path send ten numbers
to a sixty-five-feature model for months.

MTF causality
-------------
Every timeframe's frame carries ``close_time = timestamp + timeframe_minutes``:
the instant that candle's information becomes knowable. Context timeframes are
joined with ``merge_asof(..., direction="backward", allow_exact_matches=True)``
on ``close_time``, which selects the last context candle whose OWN close_time
is at or before the entry row's close_time — strictly a CLOSED candle, never
one still forming.

That is the property the legacy ``entry_v2`` dataset violated, where an H4
feature was read from a candle that would not close for another three hours.
``tests/test_research_causality.py`` proves the property here by mutating
future candles and asserting no earlier feature moves.

Only COARSER timeframes are ever context. A finer timeframe would need
intrabar data KAIROS does not store, and — more importantly — the models do
not ask for one: the M15 contracts request H1 and H4 only, and no shipped
model requests M30. Nothing here adds a timeframe a model did not ask for.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.research import contract as C
from analysis.research import indicators as ind
from analysis.research import price_action as pa

TIMEFRAME_MINUTES: Dict[str, int] = {"M15": 15, "M30": 30, "H1": 60, "H4": 240}

#: Raw candle columns the engine consumes. `spread` is required by the
#: research contract; a source that lacks it cannot produce `spread_relative`
#: and that is reported as UNAVAILABLE rather than filled in.
CANDLE_COLUMNS = ("timestamp", "open", "high", "low", "close", "spread")

#: Columns the engine emits that are NOT model features. They carry the row's
#: identity and warm-up position; a model must never receive them.
META_COLUMNS = ("timestamp", "close_time", "bar_index")


def bar_index_column(timeframe: str, entry_timeframe: str) -> str:
    """Where this timeframe's bar counter lives in a merged frame."""
    return "bar_index" if timeframe == entry_timeframe else f"{timeframe}_bar_index"


class EngineError(Exception):
    """The engine cannot honour the contract with the input it was given."""


def timeframe_minutes(timeframe: str) -> int:
    try:
        return TIMEFRAME_MINUTES[timeframe]
    except KeyError:
        raise EngineError(
            f"unknown timeframe {timeframe!r}; known: {sorted(TIMEFRAME_MINUTES)}") from None


def compute_timeframe_features(
    candles: pd.DataFrame,
    timeframe: str,
    symbol: str,
    *,
    has_spread: bool = True,
) -> pd.DataFrame:
    """Every per-timeframe feature the contract can produce, on one timeframe.

    Returns a frame carrying ``timestamp``, ``close_time`` and one column per
    feature. Producing the full library rather than only the requested subset
    is deliberate: the model's own feature list does the selecting, later and
    in one place, so the engine has no way to accidentally reorder it.

    ``has_spread=False`` omits the spread-derived columns entirely instead of
    emitting a fabricated value. An absent column is then detectable; a column
    full of invented zeros is not.
    """
    required = [c for c in CANDLE_COLUMNS if c != "spread" or has_spread]
    missing = [c for c in required if c not in candles.columns]
    if missing:
        raise EngineError(f"{symbol}/{timeframe}: candle frame is missing {missing}")

    df = candles.sort_values("timestamp").reset_index(drop=True)
    if df["timestamp"].duplicated().any():
        dupes = df.loc[df["timestamp"].duplicated(), "timestamp"].head(3).tolist()
        raise EngineError(
            f"{symbol}/{timeframe}: duplicate timestamps {dupes}; a candle series with "
            f"two bars at one instant has no defined feature value")

    out = pd.DataFrame({"timestamp": df["timestamp"]})
    out["close_time"] = df["timestamp"] + pd.Timedelta(minutes=timeframe_minutes(timeframe))
    # How many bars of THIS timeframe precede this row. Not a feature: it is
    # what lets a consumer tell "warm-up is not finished yet" (MISSING) apart
    # from "the formula produced NaN on complete data" (INVALID). Collapsing
    # those two is how a warm-up row acquires a fabricated value.
    out["bar_index"] = range(len(df))

    o, h, l, c = (df["open"].astype(float), df["high"].astype(float),
                  df["low"].astype(float), df["close"].astype(float))

    # One ATR, computed once, shared by every ATR-normalised feature. Two
    # ATRs with the same nominal period but different smoothing would make
    # `range_atr` and `body_atr` silently incomparable.
    atr14 = ind.atr(h, l, c, C.ATR_PERIOD, method="wilder")

    # --- momentum / oscillators --------------------------------------------
    out["rsi"] = ind.rsi(c, C.RSI_PERIOD)
    out["momentum_score"] = ind.momentum_score(c, C.MOMENTUM_LOOKBACK)
    stoch = ind.stochastic(h, l, c, C.STOCH_K, C.STOCH_D, C.STOCH_SMOOTH)
    out["stochastic_k"] = stoch["stochastic_k"]
    out["stochastic_d"] = stoch["stochastic_d"]
    out[f"cci_{C.CCI_PERIOD}"] = ind.cci(h, l, c, C.CCI_PERIOD)

    # --- returns ------------------------------------------------------------
    r = pa.returns(c, C.RETURN_PERIODS)
    for p in C.RETURN_PERIODS:
        out[f"return_{p}"] = r[f"return_{p}"]
    for a, b in zip(C.RETURN_PERIODS, C.RETURN_PERIODS[1:]):
        out[f"return_acceleration_{a}_{b}"] = pa.return_acceleration(
            r[f"return_{a}"], r[f"return_{b}"])

    # --- trend --------------------------------------------------------------
    out["trend_strength"] = ind.trend_strength(c, C.EMA_FAST, C.EMA_SLOW)
    out["trend_score"] = ind.trend_score(c, C.TREND_SCORE_LOOKBACK)
    out["direction"] = ind.direction_indicator(c, C.DIRECTION_EMA_PERIOD)
    adx, plus_di, minus_di = ind.directional_movement(h, l, c, C.ADX_PERIOD)
    out["adx"], out["plus_di"], out["minus_di"] = adx, plus_di, minus_di
    out["market_regime"] = ind.market_regime(adx, C.ADX_TREND_THRESHOLD)

    ema_fast, ema_slow = ind.ema(c, C.EMA_FAST), ind.ema(c, C.EMA_SLOW)
    out["distance_close_ema20_atr"] = pa.normalize_by_atr(c - ema_fast, atr14)
    out["distance_close_ema50_atr"] = pa.normalize_by_atr(c - ema_slow, atr14)
    out["ema20_ema50_distance_atr"] = pa.normalize_by_atr(ema_fast - ema_slow, atr14)
    out["close_to_sma20_atr"] = pa.normalize_by_atr(c - ind.sma(c, C.SMA_PERIOD), atr14)

    # --- volatility ---------------------------------------------------------
    out["volatility_score"] = ind.volatility_score(atr14, C.VOLATILITY_LOOKBACK)
    out["atr_pct"] = pa.atr_percent(atr14, c)
    out["rolling_volatility"] = pa.rolling_volatility(c, C.ROLLING_VOL_LOOKBACK)
    bb = ind.bollinger_bands(c, C.BB_PERIOD, C.BB_STD)
    out["bb_width"], out["bb_percent_b"] = bb["bb_width"], bb["bb_percent_b"]

    # --- structure ----------------------------------------------------------
    out["distance_from_recent_high"] = pa.distance_from_recent_high(c, h, C.RECENT_LOOKBACK)
    out["distance_from_recent_low"] = pa.distance_from_recent_low(c, l, C.RECENT_LOOKBACK)

    # --- ATR-normalised candle geometry -------------------------------------
    out["range_atr"] = pa.normalize_by_atr(pa.candle_range(h, l), atr14)
    out["body_atr"] = pa.normalize_by_atr(pa.body_size(o, c), atr14)
    out["upper_wick_atr"] = pa.normalize_by_atr(pa.upper_wick(o, h, c), atr14)
    out["lower_wick_atr"] = pa.normalize_by_atr(pa.lower_wick(o, l, c), atr14)

    # --- day structure and session clock ------------------------------------
    ts_session = _session_timestamps(df["timestamp"])
    day = pa.daily_structure(ts_session, h, l, c, atr14)
    for col in ("distance_from_day_high", "distance_from_day_low",
                "day_range_atr", "position_in_day_range"):
        out[col] = day[col]

    sessions = ind.session_indicator(ts_session, C.SESSION_WINDOWS)
    for col in sessions.columns:
        out[col] = sessions[col]
    out["hour_of_day"] = ind.hour_of_day(ts_session)
    out["day_of_week"] = ind.day_of_week(ts_session)
    out["minute_of_day"] = ind.minute_of_day(ts_session)
    out["minutes_from_session_open"] = ind.minutes_from_session_open(
        ts_session, C.TRADING_DAY_OPEN_UTC.get(symbol, C.DEFAULT_TRADING_DAY_OPEN_UTC))

    # --- spread -------------------------------------------------------------
    if has_spread:
        out["spread_relative"] = pa.spread_relative(
            df["spread"].astype(float), C.SPREAD_RELATIVE_LOOKBACK,
            zero_median_policy="UNIT_WHEN_ALSO_ZERO")

    return out


def _session_timestamps(timestamp: pd.Series) -> pd.Series:
    """Timestamps in the contract's session timezone, tz-aware.

    A naive timestamp is read as UTC rather than as machine-local time: the
    machine's locale is not part of the contract and must never reach a
    feature value. This is one of the things that would differ between a
    Windows box and this Linux container if it were left implicit.
    """
    ts = pd.to_datetime(timestamp, utc=True)
    return ts.dt.tz_convert(C.SESSION_TIMEZONE)


def align_multi_timeframe(
    entry: pd.DataFrame,
    context: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """As-of merge each context timeframe onto the entry frame by close_time.

    ``direction="backward"`` with ``allow_exact_matches=True`` picks the last
    context candle whose own close_time is <= this row's close_time. A context
    candle that has not closed yet has a LATER close_time and is therefore
    unreachable — the causality guarantee is structural, not a check that
    could be forgotten.
    """
    merged = entry.sort_values("close_time").reset_index(drop=True)
    for tf, ctx in context.items():
        ctx_sorted = ctx.sort_values("close_time").reset_index(drop=True)
        feature_cols = [c for c in ctx_sorted.columns if c not in ("timestamp", "close_time")]
        # `bar_index` travels with the context frame so a consumer can tell how
        # many CLOSED context bars stood behind this entry row.
        small = ctx_sorted[["close_time"] + feature_cols].rename(
            columns={c: f"{tf}_{c}" for c in feature_cols})
        merged = pd.merge_asof(
            merged, small, on="close_time",
            direction="backward", allow_exact_matches=True,
        )
    return merged


def add_alignment_features(merged: pd.DataFrame, entry_timeframe: str,
                           context_timeframes: Sequence[str]) -> pd.DataFrame:
    """Cross-timeframe agreement features, computed after the merge.

    These read only already-merged columns, so they inherit the merge's
    "last CLOSED candle only" causality rather than re-deriving it. Trend
    state reuses the contract's existing ``direction`` rather than inventing
    a competing notion of trend.
    """
    tfs = [entry_timeframe] + list(context_timeframes)
    cols = {tf: (C.MTF_TREND_SOURCE if tf == entry_timeframe
                 else f"{tf}_{C.MTF_TREND_SOURCE}") for tf in tfs}
    available = [tf for tf in tfs if cols[tf] in merged.columns]
    if len(available) < 2:
        return merged

    new: Dict[str, pd.Series] = {}
    trend = {}
    for tf in available:
        series = merged[cols[tf]].astype(float)
        trend[tf] = series
        new[f"{tf}_trend_state"] = series

    # Mean, not sum, so the range does not depend on how many context
    # timeframes happen to be configured.
    new["trend_alignment_score"] = pd.concat(
        [trend[tf] for tf in available], axis=1).mean(axis=1)

    for a, b in zip(available, available[1:]):
        agree = ((np.sign(trend[a]) == np.sign(trend[b])) & (trend[a] != 0)).astype(float)
        # A NaN on either side is "unknown", not "disagree".
        new[f"{a}_{b}_trend_agreement"] = agree.where(trend[a].notna() & trend[b].notna())

    first = trend[available[0]]
    full, notna = (first != 0), first.notna()
    for tf in available[1:]:
        full &= (np.sign(trend[tf]) == np.sign(first))
        notna &= trend[tf].notna()
    new["_".join(available) + "_full_alignment"] = full.astype(float).where(notna)

    collision = [k for k in new if k in merged.columns]
    if collision:
        raise EngineError(
            f"alignment features {collision} collide with aligned data; renaming is "
            f"required rather than overwriting a merged column")
    return pd.concat([merged, pd.DataFrame(new, index=merged.index)], axis=1)


def build_feature_frame(
    symbol: str,
    entry_timeframe: str,
    candles_by_timeframe: Dict[str, pd.DataFrame],
    context_timeframes: Sequence[str],
    *,
    has_spread: bool = True,
) -> pd.DataFrame:
    """The full canonical feature frame for one symbol / entry timeframe.

    ``candles_by_timeframe`` must carry the entry timeframe and every context
    timeframe. Nothing is derived that was not supplied: if a caller wants an
    M30 stack it must supply M30 candles, and no shipped model asks for one.
    """
    for tf in [entry_timeframe, *context_timeframes]:
        if tf not in candles_by_timeframe:
            raise EngineError(
                f"{symbol}/{entry_timeframe}: no candles supplied for timeframe {tf!r}; "
                f"the contract requires it")

    entry = compute_timeframe_features(
        candles_by_timeframe[entry_timeframe], entry_timeframe, symbol,
        has_spread=has_spread)
    context = {
        tf: compute_timeframe_features(candles_by_timeframe[tf], tf, symbol,
                                       has_spread=has_spread)
        for tf in context_timeframes
    }
    merged = align_multi_timeframe(entry, context)
    return add_alignment_features(merged, entry_timeframe, context_timeframes)
