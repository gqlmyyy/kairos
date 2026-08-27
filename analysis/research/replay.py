"""Offline historical replay: stored candles -> features -> model -> p_win.

Runs entirely on Linux from files on disk. No MT5, no broker session, no
account, no order. The replay is how a research model is verified before
anyone considers it for anything else, and it is the only execution path this
integration provides.

    stored OHLC candles
        -> canonical feature frame (analysis.research.engine)
        -> the model's own feature list, in order
        -> availability classification
        -> p_win, or a status explaining why not

Determinism is a property, not a hope: the same candles, the same model and
the same range produce identical vectors and identical probabilities on every
run. ``tests/test_research_determinism.py`` asserts it byte-for-byte.

Warm-up is respected rather than filled in. A row whose lookback windows are
not yet complete produces MISSING features and therefore no prediction — it
does not produce a prediction computed from a substituted value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from analysis.research import candles as cd
from analysis.research import engine as E
from analysis.research import inference as inf
from analysis.research.model_loader import LoadedModel, ModelNotCompatible, load_model

#: Both sides of every bar are scored. `target=1` means something different
#: for each side, so a bar has two probabilities, not one.
DIRECTIONS = ("long", "short")


@dataclass(frozen=True)
class ReplayResult:
    """Everything one replay produced, including what it refused to produce."""

    model_id: str
    symbol: str
    timeframe: str
    predictions: pd.DataFrame
    feature_frame: pd.DataFrame
    unavailable_columns: tuple
    unavailable_features: tuple
    rows_scored: int
    rows_refused: int
    status_counts: Dict[str, int]

    @property
    def served_any(self) -> bool:
        return self.rows_scored > 0

    def summary(self) -> str:
        head = (f"{self.model_id}: {self.rows_scored} scored, "
                f"{self.rows_refused} refused, statuses={self.status_counts}")
        if self.unavailable_columns:
            head += (f"\n  source is missing columns {list(self.unavailable_columns)}, "
                     f"which makes {len(self.unavailable_features)} contract features "
                     f"UNAVAILABLE: {list(self.unavailable_features)[:6]}")
        return head


def replay(
    symbol: str,
    timeframe: str,
    source: cd.CandleSource,
    *,
    registry_path=None,
    version: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: Optional[int] = None,
    tail: Optional[int] = None,
    directions: Sequence[str] = DIRECTIONS,
    model: Optional[LoadedModel] = None,
) -> ReplayResult:
    """Replay one model over a stored candle history.

    ``start``/``end`` bound the SCORED rows, never the candles loaded: history
    before ``start`` is still needed to warm the lookback windows, and
    trimming it would change every rolling value.

    ``version`` names the research generation when more than one is registered
    for this symbol/timeframe; the registry refuses to choose on its own.

    ``limit`` scores the FIRST n rows of the range, ``tail`` the LAST n. On a
    short history the first rows are still inside their lookback windows and
    correctly produce no prediction, so ``tail`` is usually what a caller
    wants when spot-checking a model.
    """
    if limit is not None and tail is not None:
        raise ValueError("pass limit or tail, not both")
    if model is None:
        kwargs = {} if registry_path is None else {"registry_path": registry_path}
        if version is not None:
            kwargs["version"] = version
        model = load_model(symbol, timeframe, **kwargs)

    stack_tfs = [model.timeframe, *model.card.context_timeframes]
    stack = cd.load_stack(source, symbol, stack_tfs)

    unavailable_columns = tuple(
        c for c in source.unavailable_columns()
        if c in model.contract.required_columns()
    )
    unavailable_features = tuple(
        s.name for s in model.contract.specs
        if any(col in set(unavailable_columns) for col in s.requires)
    )

    frame = E.build_feature_frame(
        symbol, model.timeframe, stack, list(model.card.context_timeframes),
        has_spread=source.provides_spread,
    )

    scored = frame
    if start is not None:
        scored = scored[scored["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        scored = scored[scored["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    if limit is not None:
        scored = scored.head(int(limit))
    if tail is not None:
        scored = scored.tail(int(tail))

    records: List[dict] = []
    status_counts: Dict[str, int] = {}
    n_ok = 0
    for row in scored.to_dict("records"):
        ts = row["timestamp"]
        for direction in directions:
            # warm-up is derived per feature from the row's own bar counters
            pred = inf.predict_row(
                model, row, entry_direction=direction,
                unavailable_columns=unavailable_columns, timestamp=ts,
            )
            status_counts[pred.status] = status_counts.get(pred.status, 0) + 1
            if pred.available:
                n_ok += 1
            records.append({
                "timestamp": ts, "direction": direction, "status": pred.status,
                "p_win": pred.p_win, "raw_probability": pred.raw_probability,
                "reason": pred.reason,
            })

    predictions = pd.DataFrame.from_records(
        records, columns=["timestamp", "direction", "status", "p_win",
                          "raw_probability", "reason"])
    return ReplayResult(
        model_id=model.card.model_id, symbol=symbol, timeframe=model.timeframe,
        predictions=predictions, feature_frame=scored,
        unavailable_columns=unavailable_columns,
        unavailable_features=unavailable_features,
        rows_scored=n_ok, rows_refused=len(records) - n_ok,
        status_counts=status_counts,
    )
