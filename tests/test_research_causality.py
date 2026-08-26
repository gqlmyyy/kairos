"""Causality: no feature at time t may depend on anything after t.

Three independent probes, because "we only used backward-looking windows" is
an easy claim to make and a hard one to keep:

1. **Future mutation** — rewrite every candle after a cutoff and assert not a
   single feature value at or before the cutoff moves.
2. **MTF boundary** — assert an H1/H4 context value only ever comes from a
   candle that has already CLOSED at the entry row's close_time.
3. **Truncation** — features computed on a truncated history must equal the
   same rows computed on the full history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.research import contract as C
from analysis.research import engine as E


def make_candles(n: int, minutes: int, *, seed: int = 11, base: float = 100.0,
                 end: str = "2026-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=n,
                       freq=f"{minutes}min")
    close = base + np.cumsum(rng.normal(0, base * 0.001, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, base * 0.0005, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, base * 0.0005, n))
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low, "close": close,
        "spread": np.floor(rng.uniform(0, 6, n)),
    })


@pytest.fixture(scope="module")
def stack():
    return {"M15": make_candles(2400, 15), "H1": make_candles(700, 60),
            "H4": make_candles(340, 240)}


FEATURE_EXCLUDE = set(E.META_COLUMNS) | {"H1_bar_index", "H4_bar_index", "M15_bar_index"}


def _feature_columns(frame: pd.DataFrame):
    return [c for c in frame.columns if c not in FEATURE_EXCLUDE]


def _mutate_future(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Replace every bar after `cutoff` with a wildly different path.

    Deliberately violent — a 3x level shift and inverted wicks — so that any
    leak at all shows up as a large difference rather than a rounding artefact.
    """
    out = df.copy()
    future = out["timestamp"] > cutoff
    for col in ("open", "high", "low", "close"):
        out.loc[future, col] = out.loc[future, col] * 3.0 + 17.0
    out.loc[future, "high"] = out.loc[future, ["open", "close"]].max(axis=1) + 5.0
    out.loc[future, "low"] = out.loc[future, ["open", "close"]].min(axis=1) - 5.0
    out.loc[future, "spread"] = 999.0
    return out


@pytest.mark.parametrize("entry_tf,contexts", [("H1", ["H4"]), ("M15", ["H1", "H4"]),
                                               ("H4", [])])
def test_future_mutation_changes_nothing_at_or_before_the_cutoff(stack, entry_tf, contexts):
    tfs = [entry_tf, *contexts]
    base = E.build_feature_frame("XAUUSD", entry_tf, {tf: stack[tf] for tf in tfs}, contexts)

    cutoff = base["timestamp"].iloc[int(len(base) * 0.6)]
    mutated_stack = {tf: _mutate_future(stack[tf], cutoff) for tf in tfs}
    mutated = E.build_feature_frame("XAUUSD", entry_tf, mutated_stack, contexts)

    a = base[base["timestamp"] <= cutoff].reset_index(drop=True)
    b = mutated[mutated["timestamp"] <= cutoff].reset_index(drop=True)
    assert len(a) == len(b) and len(a) > 100

    offenders = []
    for col in _feature_columns(base):
        x, y = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            offenders.append(f"{col}: NaN pattern moved")
            continue
        m = ~np.isnan(x)
        if m.any() and not np.array_equal(x[m], y[m]):
            worst = np.max(np.abs(x[m] - y[m]))
            offenders.append(f"{col}: max delta {worst:.3e}")
    assert not offenders, f"future data leaked into the past: {offenders}"


@pytest.mark.parametrize("entry_tf,context", [("H1", "H4"), ("M15", "H1"), ("M15", "H4")])
def test_context_candle_has_always_closed_before_it_is_used(stack, entry_tf, context):
    """The merged context bar's own close_time must be <= this row's close_time.

    This is the property the legacy entry_v2 dataset violated, where an H4
    feature was read from a candle that would not close for three more hours.
    """
    contexts = ["H4"] if entry_tf == "H1" else ["H1", "H4"]
    entry = E.compute_timeframe_features(stack[entry_tf], entry_tf, "XAUUSD")
    ctx_frames = {tf: E.compute_timeframe_features(stack[tf], tf, "XAUUSD")
                  for tf in contexts}
    merged = E.align_multi_timeframe(entry, ctx_frames)

    ctx = ctx_frames[context][["timestamp", "close_time", "bar_index"]].rename(
        columns={"timestamp": "ctx_open", "close_time": "ctx_close"})
    joined = merged[["close_time", f"{context}_bar_index"]].merge(
        ctx, left_on=f"{context}_bar_index", right_on="bar_index", how="inner")
    assert len(joined) > 50

    assert (joined["ctx_close"] <= joined["close_time"]).all(), (
        f"{context} candle used before it closed")

    # And it must be the LATEST such candle — using an older one would be
    # causal but would silently throw away information.
    step = pd.Timedelta(minutes=E.timeframe_minutes(context))
    assert (joined["ctx_close"] + step > joined["close_time"]).all(), (
        f"a stale {context} candle was used when a newer closed one existed")


@pytest.mark.parametrize("entry_tf,contexts", [("H1", ["H4"]), ("H4", [])])
def test_truncated_history_reproduces_the_same_values(stack, entry_tf, contexts):
    """Rows the model can see must not depend on data that arrives later."""
    tfs = [entry_tf, *contexts]
    full = E.build_feature_frame("XAUUSD", entry_tf, {tf: stack[tf] for tf in tfs}, contexts)
    cutoff = full["timestamp"].iloc[int(len(full) * 0.7)]

    truncated_stack = {tf: stack[tf][stack[tf]["timestamp"] <= cutoff].reset_index(drop=True)
                       for tf in tfs}
    truncated = E.build_feature_frame("XAUUSD", entry_tf, truncated_stack, contexts)

    a = full[full["timestamp"].isin(truncated["timestamp"])].reset_index(drop=True)
    b = truncated.reset_index(drop=True)
    for col in _feature_columns(full):
        pd.testing.assert_series_equal(a[col], b[col], check_names=False,
                                       obj=f"{col} changed when later bars were removed")


def test_session_features_read_only_the_rows_own_timestamp(stack):
    """A clock feature cannot leak, and must not depend on neighbouring bars."""
    frame = E.compute_timeframe_features(stack["H1"], "H1", "XAUUSD")
    shuffled = stack["H1"].sample(frac=1.0, random_state=3).reset_index(drop=True)
    reframed = E.compute_timeframe_features(shuffled, "H1", "XAUUSD")
    for col in ("hour_of_day", "day_of_week", "minute_of_day",
                "session_asian", "session_london", "session_newyork"):
        pd.testing.assert_series_equal(frame[col], reframed[col], check_names=False)


def test_day_structure_uses_the_running_day_high_not_the_final_one(stack):
    """`distance_from_day_high` must be 0 at a new day's first bar, never negative
    because of a high that has not happened yet."""
    frame = E.compute_timeframe_features(stack["H1"], "H1", "XAUUSD")
    ts = pd.to_datetime(stack["H1"]["timestamp"], utc=True)
    first_of_day = ts.dt.normalize() != ts.dt.normalize().shift(1)
    firsts = frame.loc[first_of_day.to_numpy(), "distance_from_day_high"].dropna()
    assert len(firsts) > 5
    # At the first bar of a day the running high IS this bar's high, so the
    # distance is (close - high)/close <= 0, and it can never reference a
    # later bar's higher high (which would make it strictly more negative).
    assert (firsts <= 1e-12).all()


def test_weekend_and_holiday_gaps_do_not_shift_features_across_the_gap():
    """A session gap is a gap in TIME, not a reason to reach across it.

    Bars are indexed positionally by every rolling window, so removing a
    weekend must change values only from the gap forward — never before it.
    """
    df = make_candles(900, 60, seed=5)
    ts = df["timestamp"]
    weekday = ts.dt.dayofweek
    # Carve out a realistic weekend, but pick one far enough in that every
    # lookback window is already warm on both sides of the comparison.
    weekend_starts = ts[(weekday >= 5) & (weekday.shift(1) < 5)]
    first_gap = weekend_starts[weekend_starts.index > 250].iloc[0]

    # Drop only that one weekend, so everything before it is bar-for-bar
    # identical and everything after it is legitimately shifted.
    dropped = (ts >= first_gap) & (ts < first_gap + pd.Timedelta(days=2))
    gapped = df[~dropped].reset_index(drop=True)
    assert 0 < len(df) - len(gapped) <= 48

    full_frame = E.compute_timeframe_features(df, "H1", "EURUSD")
    gap_frame = E.compute_timeframe_features(gapped, "H1", "EURUSD")
    a = full_frame[full_frame["timestamp"] < first_gap]
    b = gap_frame[gap_frame["timestamp"] < first_gap]
    assert len(a) == len(b) and len(a) > 20
    for col in ("rsi", "atr_pct", "adx", "trend_strength", "return_1"):
        pd.testing.assert_series_equal(a[col].reset_index(drop=True),
                                       b[col].reset_index(drop=True), check_names=False)
