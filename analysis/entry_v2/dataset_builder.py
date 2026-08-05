from __future__ import annotations

"""analysis/entry_v2/dataset_builder.py

Production-grade Entry v2 dataset builder (Dataset Builder ONLY).

Required outputs:
- dataset metadata JSON
- unified dataset exported as Parquet and CSV

No labels and no feature engineering here.

Dataset contents (pre-feature stage):
For each symbol and each synchronized timestamp row, store:
- symbol
- t (unix seconds)
- candles for H4/H1/M15: o,h,l,c,v for each timeframe
- plus flags for availability

Strict leakage prevention rule (pre-feature stage):
- Each timeframe candle is only attached if its timestamp <= the synchronized row timestamp.
- Synchronization uses the *latest available candle at or before t*.

Duplicate removal:
- candles are deduplicated by timestamp at loader stage.

Missing candle handling:
- If any timeframe lacks a candle at the row, the row is flagged missing.
- Validation aborts if missing coverage is too high.

Exports:
- CSV: dataset_{YYYYMMDD_HHMMSS}.csv
- Parquet: dataset_{...}.parquet

Artifacts live under caller-provided output_dir.
"""

import os
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

from .candle_loader import Candle, load_required_history, SUPPORTED_SYMBOLS, SUPPORTED_TIMEFRAMES
from .validation import (
    detect_duplicates_by_time,
    validate_candle_sanity,
    coverage_report_for_synchronized_times,
    CandleSyncRowCoverage,
    abort_if_critical,
)

logger = get_logger("entry_v2.dataset_builder")


def _tf_to_seconds(tf: str) -> int:
    tf_u = tf.strip().upper()
    if tf_u == "H4":
        return 4 * 3600
    if tf_u == "H1":
        return 3600
    if tf_u == "M15":
        return 15 * 60
    raise ValueError(f"Unsupported timeframe: {tf}")


def _select_latest_candle_at_or_before(
    candles: List[Candle],
    t: float,
) -> Optional[Candle]:
    # candles sorted by t
    # Linear search would be slow; do binary search
    lo = 0
    hi = len(candles) - 1
    best: Optional[Candle] = None

    while lo <= hi:
        mid = (lo + hi) // 2
        cm = candles[mid]
        if cm.t <= t:
            best = cm
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def _build_synchronized_time_grid(
    h4_times: List[float],
    h1_times: List[float],
    m15_times: List[float],
) -> List[float]:
    # Deprecated in H4+H1-only mode.
    # Keep for backward compatibility.
    return sorted(set(h1_times))



def build_dataset(
    *,
    output_dir: str,
    months_min: int = 12,
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    parquet_filename: Optional[str] = None,
    csv_filename: Optional[str] = None,
) -> Dict[str, Any]:
    symbols = symbols or sorted(list(SUPPORTED_SYMBOLS))
    # H4+H1 only. QuantDinger M15 often returns empty.
    timeframes = timeframes or ["H4", "H1"]


    # Validate
    for s in symbols:
        if s not in SUPPORTED_SYMBOLS:
            raise ValueError(f"Unsupported symbol: {s}")
    for tf in timeframes:
        if tf not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {tf}")

    os.makedirs(output_dir, exist_ok=True)

    loaded = load_required_history(symbols, timeframes, months_min=months_min)

    # Audit inputs
    input_audit: Dict[str, Any] = {"symbols": {}, "duplicates": {}, "sanity": {}}

    for sym in symbols:
        input_audit["symbols"][sym] = {}
        for tf in timeframes:
            candles = loaded.get((sym, tf), [])

            dup = detect_duplicates_by_time(candles)
            bad_idx = validate_candle_sanity(candles)
            input_audit["symbols"][sym][tf] = {
                "count": len(candles),
                "duplicate_times_count": len(dup),
                "bad_sanity_count": len(bad_idx),
            }

            if len(dup) > 0:
                input_audit["duplicates"][(sym, tf)] = dup[:10]

            if len(bad_idx) > 0:
                input_audit["sanity"][(sym, tf)] = bad_idx[:10]

    # Prepare dataset rows
    all_rows: List[Dict[str, Any]] = []

    dataset_start_ts: Optional[float] = None
    dataset_end_ts: Optional[float] = None
    missing_rows = 0
    duplicate_row_count = 0

    now_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parquet_filename = parquet_filename or f"entry_v2_dataset_{now_tag}.parquet"
    csv_filename = csv_filename or f"entry_v2_dataset_{now_tag}.csv"

    export_parquet_path = os.path.join(output_dir, parquet_filename)
    export_csv_path = os.path.join(output_dir, csv_filename)

    # For missing/coverage checks
    missing_any_map_total = {
        "missing_any_pct": 0.0,
        "valid_rows": 0,
        "duplicate_times": 0,
    }

    for sym in symbols:
        h4 = loaded.get((sym, "H4"), [])
        h1 = loaded.get((sym, "H1"), [])

        # time grid based on the most available higher timeframe (H1)
        # M15 removed entirely.
        grid_times = sorted(set([c.t for c in h1]))

        # Track missing coverage per row across the symbol
        has_map: Dict[float, CandleSyncRowCoverage] = {}

        for t in grid_times:
            cd_h4 = _select_latest_candle_at_or_before(h4, t)
            cd_h1 = _select_latest_candle_at_or_before(h1, t)

            if dataset_start_ts is None:
                dataset_start_ts = float(t)
            dataset_end_ts = float(t)

            has_h4 = cd_h4 is not None
            has_h1 = cd_h1 is not None

            # Forward-fill allowed: a missing h1 candle is allowed as long as we have a latest H1 <= t.
            # Because we always select latest H1 at or before t, has_h1 captures availability.
            # In H4+H1 mode, if H1 is missing it is forward-filled via the
            # latest-candle-at-or-before selection. Therefore, we only consider
            # a row missing if H4 is missing or H1 truly has no prior candle.
            # This prevents the pipeline from aborting due to H1 being unavailable
            # at exact timestamps.
            if not (has_h4 and has_h1):
                missing_rows += 1

            # Mark coverage: only H4/H1 matter in this mode.
            has_map[float(t)] = CandleSyncRowCoverage(
                has_h4=bool(has_h4),
                has_h1=bool(has_h1),
                has_m15=False,
            )


            row: Dict[str, Any] = {
                "symbol": sym,
                "t": float(t),
                # availability flags
                "has_h4": int(has_h4),
                "has_h1": int(has_h1),
                "has_m15": 0,
            }


            def fill(prefix: str, cd: Optional[Candle]):
                if cd is None:
                    row[f"{prefix}_open"] = None
                    row[f"{prefix}_high"] = None
                    row[f"{prefix}_low"] = None
                    row[f"{prefix}_close"] = None
                    row[f"{prefix}_volume"] = None
                else:
                    row[f"{prefix}_open"] = float(cd.open)
                    row[f"{prefix}_high"] = float(cd.high)
                    row[f"{prefix}_low"] = float(cd.low)
                    row[f"{prefix}_close"] = float(cd.close)
                    row[f"{prefix}_volume"] = float(cd.volume)

            fill("h4", cd_h4)
            fill("h1", cd_h1)


            all_rows.append(row)

        # coverage audit for this symbol
        coverage = coverage_report_for_synchronized_times(grid_times, has_map)
        # aggregate minimal
        missing_any_map_total["missing_any_pct"] = max(
            missing_any_map_total["missing_any_pct"], float(coverage.get("missing_any_pct", 0.0))
        )
        missing_any_map_total["valid_rows"] = max(missing_any_map_total["valid_rows"], int(len(grid_times) - coverage.get("missing_any", 0)))

    # Duplicate row detection (by symbol+t)
    seen_keys = set()
    dup_rows = 0
    for r in all_rows:
        k = (r["symbol"], r["t"])
        if k in seen_keys:
            dup_rows += 1
        else:
            seen_keys.add(k)

    duplicate_row_count = dup_rows

    # Critical abort checks
    issues = {
        "duplicate_times": 0,  # duplicates by time should be handled in loader
        "missing_any_pct": float(missing_any_map_total.get("missing_any_pct", 0.0)),
        "valid_rows": int(missing_any_map_total.get("valid_rows", 0)),
        "duplicate_rows": duplicate_row_count,
    }
    abort_if_critical(issues)

    # Export
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(all_rows)
        # Parquet + CSV
        df.to_csv(export_csv_path, index=False)
        df.to_parquet(export_parquet_path, index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to export dataset (CSV/Parquet): {e}")

    meta = {
        "builder": "analysis/entry_v2/dataset_builder.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "timeframes": timeframes,
        "months_min": months_min,
        "row_count": len(all_rows),
        "missing_rows": missing_rows,
        "duplicate_row_count": duplicate_row_count,
        "start_ts": dataset_start_ts,
        "end_ts": dataset_end_ts,
        "input_audit": input_audit,
        "exports": {
            "csv": export_csv_path,
            "parquet": export_parquet_path,
        },
    }

    meta_path = os.path.join(output_dir, "dataset_metadata.json")
    # Never overwrite silently: version file name if exists
    if os.path.exists(meta_path):
        meta_path = os.path.join(output_dir, f"dataset_metadata_{now_tag}.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    meta["dataset_metadata_path"] = meta_path
    logger.info(f"[entry_v2] dataset built rows={len(all_rows)} missing_rows={missing_rows}")

    return meta


if __name__ == "__main__":
    build_dataset(output_dir="data/entry_v2_dataset")

