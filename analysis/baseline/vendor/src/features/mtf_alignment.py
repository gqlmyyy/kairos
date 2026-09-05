"""Multi-Timeframe Alignment Engine (Section 14). H4=trend, H1=decision/setup,
M15=entry timing, but never using an unclosed higher-timeframe candle.

Every row of the entry timeframe carries `close_time` = the instant its own
candle becomes known/actionable. For each context timeframe we as-of merge on
`close_time`, direction="backward", allow_exact_matches=True: this picks the
context candle whose OWN close_time is <= the entry row's close_time — i.e.
strictly the last fully CLOSED higher-timeframe candle, never one still open.
"""
from __future__ import annotations

import pandas as pd

from src.features.feature_schema import FeatureSpec


def align_multi_timeframe(
    entry_df: pd.DataFrame,
    entry_specs: list[FeatureSpec],
    context_frames: dict[str, tuple[pd.DataFrame, list[FeatureSpec]]],
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    merged = entry_df.sort_values("close_time").reset_index(drop=True)
    merged_specs: list[FeatureSpec] = list(entry_specs)

    for tf, (ctx_df, ctx_specs) in context_frames.items():
        ctx_sorted = ctx_df.sort_values("close_time").reset_index(drop=True)
        rename_map = {s.name: f"{tf}_{s.name}" for s in ctx_specs}
        cols = ["close_time"] + [s.name for s in ctx_specs]
        ctx_small = ctx_sorted[cols].rename(columns=rename_map)

        merged = pd.merge_asof(
            merged, ctx_small,
            left_on="close_time", right_on="close_time",
            direction="backward", allow_exact_matches=True,
        )
        # renamed() carries EVERY FeatureSpec contract field (source,
        # minimum_history, null_policy, lookahead_safe, ...) rather than
        # rebuilding the spec field-by-field, so prefixed features keep
        # their full contract intact. The timeframe itself is re-pointed to
        # the context timeframe whose candles actually produce the value.
        merged_specs.extend(s.renamed(f"{tf}_{s.name}") for s in ctx_specs)
    return merged, merged_specs
