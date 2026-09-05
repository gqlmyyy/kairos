"""Live feature pipeline (Phase 3, Section 10 -- Training/Live Parity).

There is exactly ONE feature calculator in this project:
`compute_timeframe_features` + `align_multi_timeframe` +
`add_alignment_features`. The historical dataset builder uses it over the
full validated store; this wrapper uses THE SAME functions over a bounded
history buffer to produce the feature vector for the most recently CLOSED
entry candle at inference time. No second formula set exists anywhere --
parity is structural, not aspirational:

    historical_feature(t) == live_feature(t)

whenever both see the same input frames and configuration, which
tests/test_phase3_live_parity.py proves numerically.

The caller supplies canonical-schema frames per timeframe (entry + every
configured context timeframe). Slicing rule for callers: keep candles whose
timestamp <= the entry candle's own timestamp; anything still open is
harmless anyway because the aligner only ever accepts candles already
closed at the entry instant (merge_asof backward on close_time).
"""
from __future__ import annotations

import pandas as pd

from src.config.loader import Config
from src.features.engine import compute_timeframe_features
from src.features.feature_schema import FeatureSpec, feature_columns
from src.features.mtf_alignment import align_multi_timeframe
from src.features.mtf_features import add_alignment_features


class LiveFeaturePipeline:
    def __init__(self, cfg: Config, symbol: str, entry_timeframe: str):
        self.cfg = cfg
        self.symbol = symbol
        self.entry_timeframe = entry_timeframe
        self.context_timeframes = cfg.context_timeframes_for(entry_timeframe)

    def compute(self, frames: dict[str, pd.DataFrame]) -> tuple[pd.Series, list[FeatureSpec]]:
        """Compute the feature vector of the LAST CLOSED entry candle.

        `frames` maps timeframe -> canonical dataframe (timestamp tz-aware
        UTC, sorted). Missing context frames raise: silently degrading the
        feature vector would break parity with training."""
        missing = [tf for tf in [self.entry_timeframe] + self.context_timeframes if tf not in frames]
        if missing:
            raise KeyError(f"LiveFeaturePipeline requires frames for: {missing}")

        entry_minutes = self.cfg.timeframe_cfg(self.entry_timeframe)["minutes"]
        entry_feat, entry_specs = compute_timeframe_features(
            frames[self.entry_timeframe], self.entry_timeframe, entry_minutes,
            self.cfg.features, self.cfg.session_timezone(),
        )

        context_frames = {}
        for tf in self.context_timeframes:
            minutes = self.cfg.timeframe_cfg(tf)["minutes"]
            feat, specs = compute_timeframe_features(
                frames[tf], tf, minutes, self.cfg.features, self.cfg.session_timezone(),
            )
            context_frames[tf] = (feat, specs)

        merged, specs = align_multi_timeframe(entry_feat, entry_specs, context_frames)
        if self.cfg.features.get("mtf_alignment_features", {}).get("enabled"):
            merged, specs = add_alignment_features(
                merged, specs, self.entry_timeframe, self.context_timeframes,
                version=self.cfg.features.get("feature_schema_version", "1.0.0"),
            )

        cols = feature_columns(specs)
        last_row = merged.iloc[-1]
        return last_row[cols], specs

    @staticmethod
    def slice_history(
        frames: dict[str, pd.DataFrame], asof_entry_timestamp: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        """Cut every frame to candles that had STARTED at or before the entry
        candle's open timestamp. Candles still open at the entry instant may
        survive this cut, but can never influence the result: the shared
        aligner only accepts candles CLOSED at the entry close_time."""
        out = {}
        for tf, df in frames.items():
            mask = pd.to_datetime(df["timestamp"]) <= pd.Timestamp(asof_entry_timestamp)
            out[tf] = df.loc[mask].reset_index(drop=True)
        return out
