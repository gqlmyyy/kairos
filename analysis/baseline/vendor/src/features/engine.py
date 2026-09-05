"""Feature Engine: turns a single-timeframe canonical OHLCV dataframe into a
feature dataframe plus its deterministic FeatureSpec list (Section 11-13).

Causality rule (Section 12): every feature at row i uses only rows <= i of
this same timeframe. A row's *information* is not usable until its candle has
closed, so every feature frame also carries `close_time = timestamp + tf_minutes`,
which is what the MTF aligner (src/features/mtf_alignment.py) merges on.
"""
from __future__ import annotations

import pandas as pd

from src.features import indicators as ind
from src.features import price_action as pa
from src.features import structure as struct_
from src.features.feature_schema import (
    AVAILABILITY_CLOSE_TIME,
    NULL_DEFINED_FROM_FIRST_BAR,
    NULL_NAN_UNTIL_WARMUP,
    SOURCE_OHLC,
    SOURCE_SPREAD,
    SOURCE_TIMESTAMP,
    FeatureSpec,
)


def compute_timeframe_features(
    df: pd.DataFrame,
    timeframe: str,
    timeframe_minutes: int,
    features_cfg: dict,
    session_timezone: str,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    version = features_cfg.get("feature_schema_version", "1.0.0")
    out = df[["timestamp", "symbol", "timeframe", "open", "high", "low", "close", "spread", "tick_volume"]].copy()
    out["close_time"] = out["timestamp"] + pd.Timedelta(minutes=timeframe_minutes)

    o, h, l, c = out["open"], out["high"], out["low"], out["close"]
    specs: list[FeatureSpec] = []

    def add(
        name: str,
        series,
        category: str,
        formula: str,
        lookback: int,
        deps: list[str],
        source: str = SOURCE_OHLC,
        null_policy: str = NULL_NAN_UNTIL_WARMUP,
        minimum_history: int | None = None,
    ):
        out[name] = series
        specs.append(FeatureSpec(
            name, category, formula, lookback, timeframe, "float64", version, deps,
            source=source,
            minimum_history=lookback if minimum_history is None else int(minimum_history),
            availability_time=AVAILABILITY_CLOSE_TIME,
            null_policy=null_policy,
            allowed_for_training=True,
            allowed_for_live=True,
            lookahead_safe=True,
            live_parity=True,
        ))

    bi = features_cfg.get("baseline_indicators", {})
    pac = features_cfg.get("price_action", {})
    opt = features_cfg.get("optional_structure", {})

    # Internal ATR (needed by several dependents regardless of whether the
    # 'atr' feature itself is exposed) computed with the baseline atr config.
    atr_period = bi.get("atr", {}).get("period", 14)
    internal_atr = ind.atr(h, l, c, atr_period, method="wilder")

    if bi.get("rsi", {}).get("enabled"):
        p = bi["rsi"]["period"]
        add("rsi", ind.rsi(c, p), "baseline_indicator", f"Wilder RSI({p})", p, ["close"])

    if bi.get("atr", {}).get("enabled"):
        add("atr", internal_atr, "baseline_indicator", f"Wilder ATR({atr_period})", atr_period, ["high", "low", "close"])

    if bi.get("macd", {}).get("enabled"):
        m = bi["macd"]
        macd_line, signal_line, hist = ind.macd(c, m["fast"], m["slow"], m["signal"])
        add("macd_line", macd_line, "baseline_indicator", f"EMA({m['fast']})-EMA({m['slow']})", m["slow"], ["close"])
        add("macd_signal", signal_line, "baseline_indicator", f"EMA({m['signal']}) of macd_line", m["signal"], ["macd_line"])
        add("macd_histogram", hist, "baseline_indicator", "macd_line - macd_signal", m["slow"], ["macd_line", "macd_signal"])

    if bi.get("trend_strength", {}).get("enabled"):
        t = bi["trend_strength"]
        add("trend_strength", ind.trend_strength(c, t["ema_fast"], t["ema_slow"]),
            "baseline_indicator", f"(EMA{t['ema_fast']}-EMA{t['ema_slow']})/close", t["ema_slow"], ["close"])

    if bi.get("trend_score", {}).get("enabled"):
        lb = bi["trend_score"]["lookback"]
        add("trend_score", ind.trend_score(c, lb), "baseline_indicator",
            f"normalized linreg slope of close over {lb} bars", lb, ["close"])

    if bi.get("momentum_score", {}).get("enabled"):
        lb = bi["momentum_score"]["lookback"]
        add("momentum_score", ind.momentum_score(c, lb), "baseline_indicator", f"close.pct_change({lb})", lb, ["close"])

    if bi.get("volatility_score", {}).get("enabled"):
        v = bi["volatility_score"]
        vol_atr = internal_atr if v["atr_period"] == atr_period else ind.atr(h, l, c, v["atr_period"], "wilder")
        add("volatility_score", ind.volatility_score(vol_atr, v["lookback"]), "baseline_indicator",
            f"ATR({v['atr_period']}) / rolling_mean(ATR,{v['lookback']})", v["lookback"], ["high", "low", "close"])

    if bi.get("market_regime", {}).get("enabled"):
        r = bi["market_regime"]
        adx_r, _, _ = ind.directional_movement(h, l, c, r["adx_period"])
        add("market_regime", ind.market_regime(adx_r, r["adx_trend_threshold"]), "baseline_indicator",
            f"1 if ADX({r['adx_period']}) >= {r['adx_trend_threshold']} else 0", r["adx_period"], ["high", "low", "close"])

    if bi.get("session", {}).get("enabled"):
        ts_session = out["timestamp"].dt.tz_convert(session_timezone)
        session_df = ind.session_indicator(ts_session, bi["session"]["sessions"])
        for col in session_df.columns:
            add(col, session_df[col], "baseline_indicator", f"1 if timestamp (session tz) in {col} window", 0, ["timestamp"],
                source=SOURCE_TIMESTAMP, null_policy=NULL_DEFINED_FROM_FIRST_BAR)

    if bi.get("direction", {}).get("enabled"):
        p = bi["direction"]["ema_period"]
        add("direction", ind.direction_indicator(c, p), "baseline_indicator", f"sign(close - EMA({p}))", p, ["close"])

    if pac.get("returns", {}).get("enabled"):
        periods = pac["returns"]["periods"]
        r_df = pa.returns(c, periods)
        for col in r_df.columns:
            p = int(col.split("_")[1])
            add(col, r_df[col], "price_action", f"close.pct_change({p})", p, ["close"])

    if pac.get("normalized_returns", {}).get("enabled"):
        p = pac["normalized_returns"]["atr_period"]
        nr_atr = internal_atr if p == atr_period else ind.atr(h, l, c, p, "wilder")
        add("normalized_returns", pa.normalized_returns(c, nr_atr), "price_action",
            f"close.diff() / ATR({p})", p, ["close", "high", "low"])

    if pac.get("range", {}).get("enabled"):
        add("range", pa.candle_range(h, l), "price_action", "high - low", 0, ["high", "low"])

    if pac.get("body_size", {}).get("enabled"):
        add("body_size", pa.body_size(o, c), "price_action", "abs(close - open)", 0, ["open", "close"])

    if pac.get("upper_wick", {}).get("enabled"):
        add("upper_wick", pa.upper_wick(o, h, c), "price_action", "high - max(open, close)", 0, ["open", "high", "close"])

    if pac.get("lower_wick", {}).get("enabled"):
        add("lower_wick", pa.lower_wick(o, l, c), "price_action", "min(open, close) - low", 0, ["open", "low", "close"])

    if pac.get("distance_from_recent_high", {}).get("enabled"):
        lb = pac["distance_from_recent_high"]["lookback"]
        add("distance_from_recent_high", pa.distance_from_recent_high(c, h, lb), "price_action",
            f"(close - rolling_max(high,{lb})) / close", lb, ["close", "high"])

    if pac.get("distance_from_recent_low", {}).get("enabled"):
        lb = pac["distance_from_recent_low"]["lookback"]
        add("distance_from_recent_low", pa.distance_from_recent_low(c, l, lb), "price_action",
            f"(close - rolling_min(low,{lb})) / close", lb, ["close", "low"])

    if pac.get("rolling_volatility", {}).get("enabled"):
        lb = pac["rolling_volatility"]["lookback"]
        add("rolling_volatility", pa.rolling_volatility(c, lb), "price_action",
            f"rolling_std(close.pct_change(), {lb})", lb, ["close"])

    if opt.get("ema", {}).get("enabled"):
        for p in opt["ema"]["periods"]:
            add(f"ema_{p}", ind.ema(c, p), "structure", f"EMA({p})", p, ["close"])

    if opt.get("adx", {}).get("enabled"):
        p = opt["adx"]["period"]
        adx_s, plus_di, minus_di = ind.directional_movement(h, l, c, p)
        add("adx", adx_s, "structure", f"ADX({p})", p, ["high", "low", "close"])
        add("plus_di", plus_di, "structure", f"+DI({p})", p, ["high", "low", "close"])
        add("minus_di", minus_di, "structure", f"-DI({p})", p, ["high", "low", "close"])

    if opt.get("support_resistance", {}).get("enabled"):
        lb = opt["support_resistance"]["lookback"]
        sr = struct_.support_resistance(h, l, lb)
        add("resistance_level", sr["resistance_level"], "structure", f"rolling_max(high,{lb})", lb, ["high"])
        add("support_level", sr["support_level"], "structure", f"rolling_min(low,{lb})", lb, ["low"])

    if opt.get("fractals", {}).get("enabled"):
        w = opt["fractals"]["window"]
        lag = opt["fractals"]["confirmation_lag"]
        fr = struct_.fractal_signals(h, l, w, lag)
        add("fractal_high", fr["fractal_high"], "structure",
            f"Williams fractal high(window={w}), shifted +{w + lag} bars for confirmation", w + lag, ["high"])
        add("fractal_low", fr["fractal_low"], "structure",
            f"Williams fractal low(window={w}), shifted +{w + lag} bars for confirmation", w + lag, ["low"])

    if opt.get("swing_points", {}).get("enabled"):
        lb = opt["swing_points"]["lookback"]
        sw = struct_.swing_points(h, l, lb)
        add("swing_high_level", sw["swing_high_level"], "structure",
            f"last confirmed fractal-high level (lookback~{lb})", lb, ["high"])
        add("swing_low_level", sw["swing_low_level"], "structure",
            f"last confirmed fractal-low level (lookback~{lb})", lb, ["low"])

    # ----------------------------------------------------------------------
    # Extended features (feature_schema 1.1.0).
    #
    # Deliberately appended AFTER every block above rather than interleaved
    # into the thematically-matching one: feature ORDER is the dataset's
    # column order, so inserting `bb_upper` next to `atr` would shift the
    # index of every later feature and silently invalidate any model already
    # trained against the 1.0.0 ordering. Appending keeps every pre-existing
    # feature at exactly the position it had before.
    # ----------------------------------------------------------------------
    ext = features_cfg.get("extended_indicators", {})
    ext_pa = features_cfg.get("extended_price_action", {})

    if ext.get("bollinger_bands", {}).get("enabled"):
        b = ext["bollinger_bands"]
        p, nstd = b["period"], b["std"]
        bb = ind.bollinger_bands(c, p, nstd)
        add("bb_upper", bb["bb_upper"], "extended_indicator",
            f"SMA({p}) + {nstd}*rolling_std({p}, ddof=0)", p, ["close"])
        add("bb_lower", bb["bb_lower"], "extended_indicator",
            f"SMA({p}) - {nstd}*rolling_std({p}, ddof=0)", p, ["close"])
        add("bb_width", bb["bb_width"], "extended_indicator",
            "(bb_upper - bb_lower) / SMA(period)", p, ["close"])
        add("bb_percent_b", bb["bb_percent_b"], "extended_indicator",
            "(close - bb_lower) / (bb_upper - bb_lower)", p, ["close"])

    if ext.get("stochastic", {}).get("enabled"):
        st = ext["stochastic"]
        kp, dp, sm = st["k_period"], st["d_period"], st["smooth"]
        stoch = ind.stochastic(h, l, c, kp, dp, sm)
        add("stochastic_k", stoch["stochastic_k"], "extended_indicator",
            f"SMA(100*(close-LL({kp}))/(HH({kp})-LL({kp})), {sm})", kp + sm, ["high", "low", "close"])
        add("stochastic_d", stoch["stochastic_d"], "extended_indicator",
            f"SMA(stochastic_k, {dp})", kp + sm + dp, ["high", "low", "close"])

    if ext.get("cci", {}).get("enabled"):
        p = ext["cci"]["period"]
        add(f"cci_{p}", ind.cci(h, l, c, p), "extended_indicator",
            f"(TP - SMA(TP,{p})) / (0.015 * MAD(TP,{p})), TP=(h+l+c)/3", p, ["high", "low", "close"])

    # ATR-normalised candle geometry. The raw price-unit versions above are
    # kept untouched; these make the same quantities comparable across
    # instruments and volatility regimes.
    if ext_pa.get("atr_normalized_price_action", {}).get("enabled"):
        p = ext_pa["atr_normalized_price_action"].get("atr_period", atr_period)
        na = internal_atr if p == atr_period else ind.atr(h, l, c, p, "wilder")
        add("range_atr", pa.normalize_by_atr(pa.candle_range(h, l), na),
            "extended_price_action", f"(high - low) / ATR({p})", p, ["high", "low"])
        add("body_atr", pa.normalize_by_atr(pa.body_size(o, c), na),
            "extended_price_action", f"abs(close - open) / ATR({p})", p, ["open", "close"])
        add("upper_wick_atr", pa.normalize_by_atr(pa.upper_wick(o, h, c), na),
            "extended_price_action", f"(high - max(open,close)) / ATR({p})", p, ["open", "high", "close"])
        add("lower_wick_atr", pa.normalize_by_atr(pa.lower_wick(o, l, c), na),
            "extended_price_action", f"(min(open,close) - low) / ATR({p})", p, ["open", "low", "close"])

    # Distance to the EMAs the project already computes, in ATR units so the
    # value does not scale with the instrument's price level.
    if ext_pa.get("ema_distance", {}).get("enabled"):
        ed = ext_pa["ema_distance"]
        fast_p, slow_p = ed["ema_fast"], ed["ema_slow"]
        p = ed.get("atr_period", atr_period)
        na = internal_atr if p == atr_period else ind.atr(h, l, c, p, "wilder")
        ema_fast = ind.ema(c, fast_p)
        ema_slow = ind.ema(c, slow_p)
        add(f"distance_close_ema{fast_p}_atr", pa.normalize_by_atr(c - ema_fast, na),
            "extended_price_action", f"(close - EMA({fast_p})) / ATR({p})", max(fast_p, p), ["close"])
        add(f"distance_close_ema{slow_p}_atr", pa.normalize_by_atr(c - ema_slow, na),
            "extended_price_action", f"(close - EMA({slow_p})) / ATR({p})", max(slow_p, p), ["close"])
        add(f"ema{fast_p}_ema{slow_p}_distance_atr", pa.normalize_by_atr(ema_fast - ema_slow, na),
            "extended_price_action", f"(EMA({fast_p}) - EMA({slow_p})) / ATR({p})", max(slow_p, p), ["close"])

    # Where price sits inside the CURRENT day's range so far. Running
    # cummax/cummin within the day -- never the day's final high/low.
    if ext_pa.get("daily_structure", {}).get("enabled"):
        p = ext_pa["daily_structure"].get("atr_period", atr_period)
        na = internal_atr if p == atr_period else ind.atr(h, l, c, p, "wilder")
        ts_session = out["timestamp"].dt.tz_convert(session_timezone)
        day = pa.daily_structure(ts_session, h, l, c, na)
        add("distance_from_day_high", day["distance_from_day_high"], "extended_price_action",
            "(close - running day high) / close, day in session_timezone", 0, ["high", "close", "timestamp"],
            null_policy=NULL_DEFINED_FROM_FIRST_BAR)
        add("distance_from_day_low", day["distance_from_day_low"], "extended_price_action",
            "(close - running day low) / close, day in session_timezone", 0, ["low", "close", "timestamp"],
            null_policy=NULL_DEFINED_FROM_FIRST_BAR)
        add("day_range", day["day_range"], "extended_price_action",
            "running day high - running day low", 0, ["high", "low", "timestamp"],
            null_policy=NULL_DEFINED_FROM_FIRST_BAR)
        add("day_range_atr", day["day_range_atr"], "extended_price_action",
            f"day_range / ATR({p})", p, ["high", "low", "timestamp"])
        add("position_in_day_range", day["position_in_day_range"], "extended_price_action",
            "(close - running day low) / day_range, 0.5 when day_range == 0", 0,
            ["high", "low", "close", "timestamp"],
            null_policy=NULL_DEFINED_FROM_FIRST_BAR)

    # Momentum acceleration + returns in ATR units. Reuses the same periods
    # the existing `returns` block is configured with.
    if ext_pa.get("momentum_acceleration", {}).get("enabled"):
        periods = pac.get("returns", {}).get("periods", [1, 3, 5])
        p = ext_pa["momentum_acceleration"].get("atr_period", atr_period)
        na = internal_atr if p == atr_period else ind.atr(h, l, c, p, "wilder")
        r_df = pa.returns(c, periods)
        for fast, slow in zip(periods, periods[1:]):
            add(f"return_acceleration_{fast}_{slow}",
                pa.return_acceleration(r_df[f"return_{fast}"], r_df[f"return_{slow}"]),
                "extended_price_action", f"return_{fast} - return_{slow}", slow, ["close"])
        for n in periods:
            add(f"return_{n}_atr", pa.return_in_atr(c, n, na), "extended_price_action",
                f"close.diff({n}) / ATR({p})", max(n, p), ["close", "high", "low"])

    # ----------------------------------------------------------------------
    # Phase 3 additions (feature_schema 1.2.0).
    #
    # Like the 1.1.0 block above: deliberately APPENDED after every existing
    # block -- feature ORDER is dataset column order, so appending keeps every
    # pre-existing feature at exactly its historical position and a model
    # trained against 1.0.0/1.1.0 still maps onto the same leading columns.
    #
    # `real_volume` is intentionally NOT offered as a feature anywhere in this
    # section: it is identically zero across every dataset in the repository
    # (Phase 3 audit measured it directly), so it carries no information.
    # ----------------------------------------------------------------------
    p3 = features_cfg.get("phase3_additions", {})

    if p3.get("sma", {}).get("enabled"):
        s = p3["sma"]
        sp = int(s["period"])
        sma_series = ind.sma(c, sp)
        add(f"sma_{sp}", sma_series, "baseline_indicator", f"SMA({sp}) of close", sp, ["close"])
        add(f"close_to_sma{sp}_atr", pa.normalize_by_atr(c - sma_series, internal_atr),
            "extended_price_action", f"(close - SMA({sp})) / ATR({atr_period})", sp, ["close"])

    if p3.get("log_returns", {}).get("enabled"):
        for n in p3["log_returns"].get("periods", [1]):
            n = int(n)
            add(f"log_return_{n}", ind.log_return(c, n), "price_action",
                f"ln(close / close.shift({n}))", n, ["close"])

    if p3.get("atr_pct", {}).get("enabled"):
        add("atr_pct", pa.atr_percent(internal_atr, c), "baseline_indicator",
            f"ATR({atr_period}) / close (price-level-normalized volatility)", atr_period,
            ["high", "low", "close"])

    if p3.get("session_clock", {}).get("enabled"):
        sc = p3["session_clock"]
        ts_session_p3 = out["timestamp"].dt.tz_convert(session_timezone)
        symbols_present = out["symbol"].dropna().unique()
        symbol_name = str(symbols_present[0]) if len(symbols_present) else ""
        open_map = sc.get("trading_day_open_utc", {})
        day_open = open_map.get(symbol_name, sc.get("default_trading_day_open_utc", "00:00"))
        add("hour_of_day", ind.hour_of_day(ts_session_p3), "session_feature",
            f"hour-of-day in session timezone ({session_timezone})", 0, ["timestamp"],
            source=SOURCE_TIMESTAMP, null_policy=NULL_DEFINED_FROM_FIRST_BAR)
        add("day_of_week", ind.day_of_week(ts_session_p3), "session_feature",
            f"day-of-week (Mon=0..Sun=6) in {session_timezone}", 0, ["timestamp"],
            source=SOURCE_TIMESTAMP, null_policy=NULL_DEFINED_FROM_FIRST_BAR)
        add("minute_of_day", ind.minute_of_day(ts_session_p3), "session_feature",
            f"minutes since local midnight in {session_timezone}", 0, ["timestamp"],
            source=SOURCE_TIMESTAMP, null_policy=NULL_DEFINED_FROM_FIRST_BAR)
        add("minutes_from_session_open", ind.minutes_from_session_open(ts_session_p3, day_open),
            "session_feature",
            f"minutes since this symbol's trading-day open ({symbol_name} opens {day_open} UTC "
            f"-- dominant empirical value from Phase 2; fixed-UTC approximation across DST)",
            0, ["timestamp"],
            source=SOURCE_TIMESTAMP, null_policy=NULL_DEFINED_FROM_FIRST_BAR)

    if p3.get("spread", {}).get("enabled") and "spread" in out.columns:
        sp_cfg = p3["spread"]
        spread_series = out["spread"].astype(float)
        add("spread_points", spread_series, "market_data",
            "broker spread of this bar, in points", 0, ["spread"],
            source=SOURCE_SPREAD, null_policy=NULL_DEFINED_FROM_FIRST_BAR)
        rel_lb = int(sp_cfg.get("relative_lookback", 200))
        # zero_median_policy: see src/features/price_action.spread_relative.
        # The pre-repair "STRICT" behaviour turned a legitimately zero trailing
        # median into NaN, which the dataset builder then treated as a reason to
        # delete the whole bar.
        zm_policy = sp_cfg.get("zero_median_policy", "UNIT_WHEN_ALSO_ZERO")
        add("spread_relative", pa.spread_relative(spread_series, rel_lb, zm_policy),
            "market_data",
            f"spread / rolling_median(spread, {rel_lb}) (dimensionless; "
            f"zero_median_policy={zm_policy})", rel_lb, ["spread"],
            source=SOURCE_SPREAD)
        mean_lb = int(sp_cfg.get("mean_lookback", 50))
        add("spread_ma", pa.spread_mean(spread_series, mean_lb), "market_data",
            f"rolling_mean(spread, {mean_lb}), points", mean_lb, ["spread"],
            source=SOURCE_SPREAD)

    return out, specs
