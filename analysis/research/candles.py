"""Candle sources for offline research inference.

No MT5, no broker, no account. A source is a directory of stored OHLC files
and a declaration of which canonical columns those files actually carry. The
declaration is the point: the research contract needs ``spread``, KAIROS's
stored historical snapshot does not have it, and the honest response is to
say so rather than to invent a column.

Supported layouts
-----------------
``KairosHistoricalSource``
    ``data/historical/<SYMBOL>_<TF>.json`` — the format
    ``scripts/fetch_training_candles.py`` writes: a JSON array of
    ``{"t", "open", "high", "low", "close", "volume"}``. ``t`` is a Unix epoch
    in seconds. **No spread column**, so ``provides_spread`` is False.

``CsvCandleSource``
    ``<dir>/<SYMBOL>_<TF>.csv`` with named columns. Used by the replay CLI and
    by the golden fixtures, which DO carry spread.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


class CandleSourceError(Exception):
    """The source cannot supply what was asked of it."""


@dataclass(frozen=True)
class CandleSource:
    """Base: a place to read candles from, plus what it can actually provide."""

    root: Path
    provides_spread: bool

    def path_for(self, symbol: str, timeframe: str) -> Path:  # pragma: no cover
        raise NotImplementedError

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    def available(self, symbol: str, timeframe: str) -> bool:
        return self.path_for(symbol, timeframe).exists()

    def unavailable_columns(self) -> List[str]:
        """Canonical candle columns this source cannot supply."""
        return [] if self.provides_spread else ["spread"]


def _finalize(df: pd.DataFrame, symbol: str, timeframe: str,
              provides_spread: bool) -> pd.DataFrame:
    """Normalise a loaded frame: UTC timestamps, sorted, deduplicated, typed."""
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise CandleSourceError(f"{symbol}/{timeframe}: candles are missing {missing}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype(float)
    if provides_spread:
        if "spread" not in out.columns:
            raise CandleSourceError(
                f"{symbol}/{timeframe}: source declares spread but the file has no "
                f"spread column")
        out["spread"] = out["spread"].astype(float)
    else:
        # Not filled with a placeholder: the column stays absent so that
        # everything downstream reports UNAVAILABLE rather than reading a zero.
        out = out.drop(columns=[c for c in ("spread",) if c in out.columns])

    out = out.sort_values("timestamp").reset_index(drop=True)
    if out["timestamp"].duplicated().any():
        n = int(out["timestamp"].duplicated().sum())
        raise CandleSourceError(
            f"{symbol}/{timeframe}: {n} duplicate timestamps. Deduplicating silently "
            f"would change every rolling window, so this is refused rather than repaired.")

    bad = out[(out["high"] < out["low"])
              | (out["high"] < out[["open", "close"]].max(axis=1))
              | (out["low"] > out[["open", "close"]].min(axis=1))]
    if len(bad):
        raise CandleSourceError(
            f"{symbol}/{timeframe}: {len(bad)} bars violate OHLC ordering "
            f"(first at {bad['timestamp'].iloc[0]})")

    keep = ["timestamp", "open", "high", "low", "close"] + (["spread"] if provides_spread else [])
    return out[keep]


#: How many bytes of a candle file are read to decide whether it carries a
#: spread column. The writer emits one flat record per bar, so the first record
#: is well inside this and the whole file never has to be parsed.
_PROBE_BYTES = 4096


def _file_has_spread(path: Path) -> bool:
    """Does this candle file carry a per-bar spread field?

    Probed from the file rather than assumed, because ``fetch_training_candles``
    DOES capture MT5's per-bar spread — a snapshot without it is simply older
    than that fix. Hard-coding "KAIROS has no spread" would mean a refreshed
    snapshot silently kept being refused.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            head = fh.read(_PROBE_BYTES)
    except OSError:
        return False
    return '"spread"' in head


@dataclass(frozen=True)
class KairosHistoricalSource(CandleSource):
    """``data/historical/<SYMBOL>_<TF>.json`` as written by the KAIROS fetcher.

    Whether this source can supply ``spread`` is DETECTED from the files, not
    assumed. ``scripts/fetch_training_candles.py`` records MT5's per-bar spread
    alongside OHLC, so a refreshed snapshot satisfies the research contract and
    is served; an older snapshot that predates that change does not, and every
    spread-derived feature is reported UNAVAILABLE rather than filled in.

    The detection is conservative and whole-source: if ANY readable candle file
    lacks the column, the source declares it absent. A stack where H1 has spread
    and H4 does not would produce an entry vector whose context features came
    from a different contract than its own, which is worse than refusing.
    """

    def __init__(self, root="data/historical"):
        object.__setattr__(self, "root", Path(root))
        files = sorted(Path(root).glob("*_*.json")) if Path(root).is_dir() else []
        candles = [f for f in files if f.name != "manifest.json"]
        has_spread = bool(candles) and all(_file_has_spread(f) for f in candles)
        object.__setattr__(self, "provides_spread", has_spread)

    def path_for(self, symbol: str, timeframe: str) -> Path:
        return self.root / f"{symbol}_{timeframe}.json"

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            raise CandleSourceError(f"no candles at {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, list) or not raw:
            raise CandleSourceError(f"{path}: expected a non-empty JSON array")
        df = pd.DataFrame(raw)
        if "t" not in df.columns:
            raise CandleSourceError(f"{path}: expected a 't' epoch column")
        # Seconds vs milliseconds is auto-detected rather than assumed: a
        # wrong unit would silently move every bar by decades.
        unit = "ms" if float(df["t"].iloc[0]) > 1e11 else "s"
        df["timestamp"] = pd.to_datetime(df["t"], unit=unit, utc=True)
        return _finalize(df, symbol, timeframe, provides_spread=self.provides_spread)


@dataclass(frozen=True)
class CsvCandleSource(CandleSource):
    """``<root>/<SYMBOL>_<TF>.csv`` with a header row."""

    def __init__(self, root, provides_spread: bool = True):
        object.__setattr__(self, "root", Path(root))
        object.__setattr__(self, "provides_spread", bool(provides_spread))

    def path_for(self, symbol: str, timeframe: str) -> Path:
        return self.root / f"{symbol}_{timeframe}.csv"

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            raise CandleSourceError(f"no candles at {path}")
        return _finalize(pd.read_csv(path), symbol, timeframe, self.provides_spread)


@dataclass(frozen=True)
class JsonCandleSource(CandleSource):
    """``<root>/<SYMBOL>_<TF>.json`` holding records with named columns.

    Used by the golden fixtures, which carry an explicit ``spread`` column and
    an ISO-8601 ``timestamp`` (rather than the KAIROS fetcher's epoch ``t``).
    """

    def __init__(self, root, provides_spread: bool = True):
        object.__setattr__(self, "root", Path(root))
        object.__setattr__(self, "provides_spread", bool(provides_spread))

    def path_for(self, symbol: str, timeframe: str) -> Path:
        return self.root / f"{symbol}_{timeframe}.json"

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            raise CandleSourceError(f"no candles at {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return _finalize(pd.DataFrame(raw), symbol, timeframe, self.provides_spread)


def load_stack(source: CandleSource, symbol: str, timeframes: Sequence[str]
               ) -> Dict[str, pd.DataFrame]:
    """Load every timeframe of one symbol's stack, or fail naming what is absent."""
    absent = [tf for tf in timeframes if not source.available(symbol, tf)]
    if absent:
        raise CandleSourceError(
            f"{symbol}: no candles for {absent} under {source.root}")
    return {tf: source.load(symbol, tf) for tf in timeframes}
