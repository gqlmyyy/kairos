"""Deterministic M30 derivation from M15 (Phase 3, Section 4).

`config/timeframes.yaml` declares M30 as a valid ENTRY timeframe, but no M30
candles exist in this repository's raw store (data/raw/mt5 carries only
M15/H1/H4) and adding an independent M30 acquisition stream was explicitly
ruled unnecessary when M15 exists. The source of M30 is therefore FIXED to:

    M30 := deterministic OHLCV aggregation of validated M15

Rules (all enforced here and in tests):
  * bins are epoch-aligned half-hour windows [t, t+30m); the emitted
    timestamp is the bin START, close_time = start + 30m;
  * open=first, high=max, low=min, close=last over the bin's M15 bars,
    tick_volume/real_volume summed, spread averaged (mean of points);
  * a bin is emitted ONLY if BOTH constituent M15 slots are present. A bin
    missing either slot is DROPPED -- aggregation never synthesizes a candle
    from partial data (Section 6: no fabrication);
  * output schema/units/timezone are identical to the canonical layer, so
    every downstream consumer (validator semantics, feature engine, aligner)
    treats derived M30 exactly like any stored timeframe;
  * pure function: same input frame -> byte-identical output frame.
"""
from __future__ import annotations

import pandas as pd

DERIVED_TIMEFRAME_MINUTES = {"M30": (30, "M15")}


def derive_timeframe(m15: pd.DataFrame, target_tf: str = "M30") -> pd.DataFrame:
    """Aggregate canonical-schema M15 candles into `target_tf` candles."""
    if target_tf not in DERIVED_TIMEFRAME_MINUTES:
        raise ValueError(
            f"No deterministic derivation defined for '{target_tf}'. "
            f"Defined: {sorted(DERIVED_TIMEFRAME_MINUTES)}"
        )
    minutes, required_source = DERIVED_TIMEFRAME_MINUTES[target_tf]
    if minutes % 15 != 0 or minutes // 15 < 2:
        raise ValueError(f"Unsupported derivation window: {minutes}m")

    df = m15.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)

    slot = ((ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds() // (15 * 60)).astype("int64")
    bin_start_ns = (slot // (minutes // 15)) * (minutes // 15)
    bin_key = pd.to_datetime(bin_start_ns * 15 * 60, unit="s", utc=True)
    slot_in_bin = (slot % (minutes // 15)).astype("int64")

    work = pd.DataFrame({
        "bin": bin_key,
        "slot": slot_in_bin,
        "open": df["open"].to_numpy(),
        "high": df["high"].to_numpy(),
        "low": df["low"].to_numpy(),
        "close": df["close"].to_numpy(),
        "spread": df["spread"].astype(float).to_numpy() if "spread" in df else 0.0,
        "tick_volume": df["tick_volume"].to_numpy() if "tick_volume" in df else 0,
        "real_volume": df["real_volume"].to_numpy() if "real_volume" in df else 0,
    })

    slots_needed = set(range(minutes // 15))
    complete = work.groupby("bin")["slot"].agg(lambda s: set(s.unique()) == slots_needed)
    dup = work.duplicated(subset=["bin", "slot"]).any()
    if dup:
        raise ValueError("Duplicate M15 timestamps inside one derivation bin -- refusing to guess.")

    full_bins = complete[complete].index
    agg = work[work["bin"].isin(full_bins)].groupby("bin", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        spread=("spread", "mean"),
        tick_volume=("tick_volume", "sum"),
        real_volume=("real_volume", "sum"),
        n_bars=("close", "size"),
    )

    out = pd.DataFrame({
        "timestamp": agg.index,
        "symbol": df["symbol"].iloc[0] if "symbol" in df and len(df) else "",
        "timeframe": target_tf,
        "open": agg["open"].to_numpy(),
        "high": agg["high"].to_numpy(),
        "low": agg["low"].to_numpy(),
        "close": agg["close"].to_numpy(),
        "spread": agg["spread"].to_numpy(),
        "tick_volume": agg["tick_volume"].to_numpy(),
        "real_volume": agg["real_volume"].to_numpy(),
        f"derived_from_{required_source}_bars": agg["n_bars"].to_numpy(),
    })
    return out.reset_index(drop=True)


def derivation_provenance(target_tf: str = "M30") -> dict:
    """Machine-readable provenance block for manifests/reports."""
    minutes, source = DERIVED_TIMEFRAME_MINUTES[target_tf]
    return {
        "timeframe": target_tf,
        "method": "deterministic_ohlcv_aggregation",
        "source_timeframe": source,
        "bin_minutes": minutes,
        "bin_alignment": "epoch_aligned_utc",
        "incomplete_bin_policy": "drop",
        "fabrication": "none",
    }
