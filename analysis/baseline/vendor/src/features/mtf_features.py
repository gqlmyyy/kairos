"""Explicit cross-timeframe agreement features (feature_schema 1.1.0).

These are the only features in the project that cannot be computed inside
`compute_timeframe_features`, because each one combines the entry timeframe
with its already-aligned context timeframes. They are therefore computed
**after** `align_multi_timeframe`, reading only columns that merge produced.

Causality is inherited, not re-derived: every context column here arrived
through `merge_asof(..., direction="backward")` on `close_time`, so an H1
value is only present once that H1 candle has closed, and likewise H4. This
module never touches raw OHLC, never shifts anything forward, and never
looks at a row other than its own. The existing MTF architecture is used
exactly as-is -- nothing about the merge changed.

Trend state reuses the project's EXISTING trend definition rather than
inventing a competing one: `direction` = `sign(close - EMA(period))` from
`src/features/indicators.py: direction_indicator`, already computed per
timeframe. `H1_trend_state` is therefore the H1 candle's own `direction`,
carried across by the aligner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_schema import (
    SOURCE_CROSS_TIMEFRAME,
    NULL_NAN_UNTIL_WARMUP,
    FeatureSpec,
)

# The per-timeframe feature whose value defines "trend state". Changing this
# would change the meaning of every feature below, so it is named once here.
TREND_SOURCE_FEATURE = "direction"


def _trend_column_for(timeframe: str, entry_timeframe: str) -> str:
    """Where this timeframe's trend value lives in the merged frame: the
    entry timeframe keeps its bare name, context timeframes were prefixed by
    the aligner."""
    return TREND_SOURCE_FEATURE if timeframe == entry_timeframe else f"{timeframe}_{TREND_SOURCE_FEATURE}"


def alignment_feature_names(entry_timeframe: str, context_timeframes: list[str]) -> list[str]:
    """The names this module will add, in the order it adds them. Exposed so
    tests and docs can assert the schema without building a dataset."""
    tfs = [entry_timeframe] + list(context_timeframes)
    names = [f"{tf}_trend_state" for tf in tfs]
    names.append("trend_alignment_score")
    for a, b in zip(tfs, tfs[1:]):
        names.append(f"{a}_{b}_trend_agreement")
    if len(tfs) >= 2:
        names.append("_".join(tfs) + "_full_alignment")
    return names


def add_alignment_features(
    merged: pd.DataFrame,
    merged_specs: list[FeatureSpec],
    entry_timeframe: str,
    context_timeframes: list[str],
    version: str = "1.1.0",
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Append cross-timeframe agreement features to an aligned frame.

    Returns (frame, specs) with the new features appended at the END of the
    spec list, so every pre-existing feature keeps its column position.

    A missing trend column (a context timeframe that produced no `direction`
    feature, e.g. because it is disabled in config) is not an error: the
    corresponding features are simply not added, and that is reported by the
    returned spec list rather than by silently emitting zeros.
    """
    tfs = [entry_timeframe] + list(context_timeframes)
    columns = {tf: _trend_column_for(tf, entry_timeframe) for tf in tfs}
    available = [tf for tf in tfs if columns[tf] in merged.columns]
    if len(available) < 2:
        # Nothing meaningful to compare across timeframes.
        return merged, merged_specs

    specs = list(merged_specs)
    # Collected here and concatenated once at the end: inserting ~7 columns
    # one at a time into a wide merged frame fragments it badly.
    new_columns: dict[str, pd.Series] = {}

    def add(name: str, series, formula: str, lookback: int, deps: list[str], timeframe: str):
        if name in merged.columns or name in new_columns:
            raise ValueError(
                f"MTF alignment feature '{name}' collides with an existing column. "
                f"Rename the feature rather than overwriting aligned data."
            )
        new_columns[name] = series
        specs.append(FeatureSpec(
            name=name, category="mtf_alignment", formula=formula, lookback=lookback,
            timeframe=timeframe, dtype="float64", version=version, dependencies=deps,
            source=SOURCE_CROSS_TIMEFRAME, minimum_history=max(lookback, 0),
            availability_time="close_time", null_policy=NULL_NAN_UNTIL_WARMUP,
            allowed_for_training=True, allowed_for_live=True,
            lookahead_safe=True, live_parity=True,
        ))

    # 1. Per-timeframe trend state, restated under an explicit name so the
    #    model (and a human reading SHAP output) sees "H4 trend" rather than
    #    having to know that `H4_direction` means that.
    trend = {}
    for tf in available:
        col = columns[tf]
        series = merged[col].astype(float)
        trend[tf] = series
        add(f"{tf}_trend_state", series,
            f"{tf} candle's own `{TREND_SOURCE_FEATURE}` = sign(close - EMA(n)); "
            f"context timeframes arrive via merge_asof on close_time (last CLOSED candle only)",
            0, [col], tf)

    # 2. Net agreement across all available timeframes, in [-1, +1]:
    #    +1 = every timeframe bullish, -1 = every timeframe bearish,
    #    0 = fully split. Mean rather than sum so the range does not depend
    #    on how many context timeframes are configured.
    stacked = pd.concat([trend[tf] for tf in available], axis=1)
    add("trend_alignment_score", stacked.mean(axis=1),
        "mean of every available <tf>_trend_state, in [-1, +1]",
        0, [f"{tf}_trend_state" for tf in available], entry_timeframe)

    # 3. Pairwise agreement between ADJACENT timeframes (entry vs next
    #    coarser, and so on). 1.0 only when both are non-zero AND same sign;
    #    a flat (0) timeframe is not "agreement".
    for a, b in zip(available, available[1:]):
        agree = ((np.sign(trend[a]) == np.sign(trend[b])) & (trend[a] != 0)).astype(float)
        agree = agree.where(trend[a].notna() & trend[b].notna())
        add(f"{a}_{b}_trend_agreement", agree,
            f"1.0 if {a}_trend_state and {b}_trend_state share a non-zero sign, else 0.0",
            0, [f"{a}_trend_state", f"{b}_trend_state"], entry_timeframe)

    # 4. Full stack alignment: every timeframe pointing the same non-zero way.
    if len(available) >= 2:
        first = trend[available[0]]
        full = (first != 0)
        notna = first.notna()
        for tf in available[1:]:
            full &= (np.sign(trend[tf]) == np.sign(first))
            notna &= trend[tf].notna()
        add("_".join(available) + "_full_alignment", full.astype(float).where(notna),
            "1.0 only when every available <tf>_trend_state shares one non-zero sign",
            0, [f"{tf}_trend_state" for tf in available], entry_timeframe)

    out = pd.concat([merged, pd.DataFrame(new_columns, index=merged.index)], axis=1)
    return out, specs
