"""Feature schema (Section 13): the authoritative, deterministic record of
every feature's name/category/formula/lookback/timeframe/dtype/version/deps.
Column order in every dataset always follows this schema's order.

Phase 3 (feature-engineering freeze) extends FeatureSpec with the full
Feature Contract fields. Every field added in v1.2.0 carries a default so a
legacy 8-field schema JSON (experiments from 1.0.0/1.1.0) still loads via
``FeatureSpec(**payload)`` unchanged; freshly generated schemas always emit
the complete contract."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path


# Contract vocabulary (frozen values -- manifests validate against these).
SOURCE_OHLC = "ohlc"                    # computed from the candle's own OHLC(V)
SOURCE_TIMESTAMP = "timestamp"          # pure function of the row timestamp
SOURCE_SPREAD = "spread"                # broker spread column
SOURCE_CROSS_TIMEFRAME = "cross_timeframe"  # needs another timeframe's candles
SOURCE_META = "meta"                    # descriptive sample metadata (entry side)
NULL_NAN_UNTIL_WARMUP = "nan_until_warmup"      # NaN until minimum_history rows exist
NULL_DEFINED_FROM_FIRST_BAR = "defined_from_first_bar"  # computable at row 0
AVAILABILITY_CLOSE_TIME = "close_time"  # known once the own candle closes


@dataclass
class FeatureSpec:
    name: str
    category: str
    formula: str
    lookback: int
    timeframe: str
    dtype: str
    version: str
    dependencies: list[str] = field(default_factory=list)
    # ---------------- Phase 3 contract fields (defaults = legacy-compatible) --
    source: str = SOURCE_OHLC
    minimum_history: int | None = None       # None -> falls back to `lookback`
    availability_time: str = AVAILABILITY_CLOSE_TIME
    null_policy: str = NULL_NAN_UNTIL_WARMUP
    allowed_for_training: bool = True
    allowed_for_live: bool = True
    lookahead_safe: bool = True              # asserted by the Future Mutation Test
    live_parity: bool = True                 # computed by the single shared calculator

    @property
    def effective_minimum_history(self) -> int:
        """Bars of own-timeframe history required before this feature holds a
        defined value. Falls back to ``lookback`` for legacy schemas."""
        if self.minimum_history is not None:
            return int(self.minimum_history)
        return int(self.lookback)

    def renamed(self, new_name: str) -> "FeatureSpec":
        """A copy of this spec under another name (MTF prefixing), carrying
        every contract field -- never rebuilt field-by-field by callers."""
        return replace(self, name=new_name)

    def as_dict(self) -> dict:
        return self.__dict__


def write_feature_schema(specs: list[FeatureSpec], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_count": len(specs),
        "feature_order": [s.name for s in specs],
        "features": [s.as_dict() for s in specs],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def feature_columns(specs: list[FeatureSpec]) -> list[str]:
    return [s.name for s in specs]

