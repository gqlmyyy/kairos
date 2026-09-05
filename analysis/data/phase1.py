"""PHASE 1 — DATA FOUNDATION.

Integrity validation and the Phase 1 gap taxonomy for the nine historical
(symbol, timeframe) datasets under ``data/historical/``. Data only: nothing
here trains, labels, selects a model, or touches live trading.

This module WRAPS the project's existing, tested data engine
(``analysis/features/timeframe_alignment.py``) rather than reimplementing it:
``validate_series`` owns structural OHLC checks and ``classify_gaps`` owns
calendar-aware gap classification. What Phase 1 adds on top is the reporting
contract the phase requires (expected/actual/missing per dataset) and the
phase's own gap taxonomy, mapped from the engine's categories:

=================================  ==================  =====================
engine category (reason)           Phase 1 category    meaning
=================================  ==================  =====================
EXPECTED_MARKET_GAP (weekly_close) WEEKEND             ordinary Fri->Sun close
EXPECTED_MARKET_GAP (known_holiday,
                    broker_maint.) MARKET_CLOSED       holiday / daily pause
EXPECTED_MARKET_GAP (any other)    EXPECTED            expected, unmapped reason
SUSPICIOUS_GAP                     UNKNOWN             small, unexplained,
                                                       honest ignorance
DATA_ERROR                         DATA_GAP            large, unexplained --
                                                       blocks PASS
=================================  ==================  =====================

Storage contract (kept as-is, see scripts/fetch_training_candles.py):
``data/historical/<SYMBOL>_<TF>.json`` — a JSON array of rows::

    {"t", "open", "high", "low", "close", "volume", "spread", "real_volume"}

``t`` is a Unix epoch in seconds (UTC, the bar's OPEN time). ``volume`` is
MT5's tick_volume. ``spread`` is the broker's per-bar spread in points. The
(symbol, timeframe) identity lives in the filename and the manifest entry —
``CANONICAL_FIELD_MAP`` documents the mapping to the phase's canonical field
names without duplicating a per-row symbol column into half a million rows.

No forward-fill, no interpolation, no synthetic candles: a gap stays a gap and
is reported, never repaired.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from analysis.features import timeframe_alignment as ta

#: The phase's fixed scope. Nothing else is fetched or validated here.
PHASE1_SYMBOLS = ("EURUSD", "GBPUSD", "XAUUSD")
PHASE1_TIMEFRAMES = ("M15", "H1", "H4")

#: Version of the stored candle-row schema this phase validates: the eight
#: canonical fields documented in CANONICAL_FIELD_MAP. Bump when the row
#: schema itself changes, never when only more rows are added.
DATA_SCHEMA_VERSION = "kairos-candles-v2"

#: Full coverage target. A dataset covering less is INCOMPLETE — reported as
#: such, never padded to look complete (Phase 1 spec, section 15).
TARGET_YEARS = 5.0

#: The row schema ``fetch_training_candles.py`` writes. A row missing any of
#: the required fields fails validation rather than being silently tolerated.
REQUIRED_FIELDS = ("t", "open", "high", "low", "close", "volume", "spread")
OPTIONAL_FIELDS = ("real_volume",)

#: Mapping from the stored field names to the phase's canonical names. This is
#: documentation, not a transformation: the stored schema is kept unchanged so
#: every existing loader (KairosHistoricalSource, train_entry_model,
#: validate_m30_candles) keeps working.
CANONICAL_FIELD_MAP = {
    "t": "timestamp",              # Unix epoch seconds, UTC, bar OPEN time
    "volume": "tick_volume",       # MT5 tick count per bar
    "spread": "spread",            # broker points, per closed bar
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "real_volume": "real_volume",  # often 0 on FX/CFD — reported, not assumed
    # symbol / timeframe: identity lives in the filename + manifest entry,
    # not repeated per row.
}

#: Gap categories this phase reports. The engine's categories are mapped onto
#: these — see the module docstring for the mapping table.
CATEGORY_WEEKEND = "WEEKEND"
CATEGORY_MARKET_CLOSED = "MARKET_CLOSED"
CATEGORY_EXPECTED = "EXPECTED"
CATEGORY_DATA_GAP = "DATA_GAP"
CATEGORY_UNKNOWN = "UNKNOWN"
PHASE1_CATEGORIES = (CATEGORY_WEEKEND, CATEGORY_MARKET_CLOSED,
                     CATEGORY_EXPECTED, CATEGORY_DATA_GAP, CATEGORY_UNKNOWN)

_REASON_TO_PHASE1 = {
    "weekly_close": CATEGORY_WEEKEND,
    "known_holiday": CATEGORY_MARKET_CLOSED,
    "broker_maintenance": CATEGORY_MARKET_CLOSED,
}

# engine categories with no reason-specific mapping:
_ENGINE_CATEGORY_TO_PHASE1 = {
    "SUSPICIOUS_GAP": CATEGORY_UNKNOWN,
    "DATA_ERROR": CATEGORY_DATA_GAP,
    "EXPECTED_MARKET_GAP": CATEGORY_EXPECTED,  # fallback for an unmapped reason
}


def map_engine_gap(gap: Dict[str, Any]) -> str:
    """Engine gap record -> Phase 1 category (the docstring's mapping table)."""
    reason = gap.get("reason", "")
    if gap.get("category") == "EXPECTED_MARKET_GAP" and reason in _REASON_TO_PHASE1:
        return _REASON_TO_PHASE1[reason]
    return _ENGINE_CATEGORY_TO_PHASE1.get(gap.get("category", ""),
                                          CATEGORY_UNKNOWN)


def utc_iso(timestamp: float) -> str:
    """Unix epoch seconds -> ISO-8601 UTC string (metadata/report format)."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()


def check_fields(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Which required fields does each row carry, and how many rows are short?

    A missing field is a schema violation, not a zero: the report names the
    field and the count instead of inventing a value.
    """
    per_field = {field: 0 for field in REQUIRED_FIELDS}
    unknown: set = set()
    for row in rows:
        for field in REQUIRED_FIELDS:
            if field not in row or row[field] is None:
                per_field[field] += 1
        unknown.update(k for k in row if k not in CANONICAL_FIELD_MAP)
    return {
        "missing_field_counts": {k: v for k, v in per_field.items() if v},
        "rows_missing_required_fields": sum(
            1 for row in rows
            if any(row.get(f) is None for f in REQUIRED_FIELDS)),
        "unknown_fields": sorted(unknown),
    }


def validate_ohlc_rows(rows: Sequence[Dict[str, Any]],
                       timeframe: str) -> Dict[str, Any]:
    """Structural OHLC checks for one series, on top of ``ta.validate_series``.

    Adds the one check the engine does not make — prices must be positive —
    and keeps every engine count under its own name so nothing is hidden.
    """
    structural = ta.validate_series(rows, timeframe)

    non_positive = 0
    for row in rows:
        try:
            values = [float(row[f]) for f in ("open", "high", "low", "close")]
        except (KeyError, TypeError, ValueError):
            continue  # already counted by the engine as bad_ohlc
        if any(v <= 0.0 for v in values):
            non_positive += 1

    return {
        "row_count": structural["count"],
        "unsorted_count": structural["unsorted"],
        "duplicate_count": structural["duplicate_timestamps"],
        "misaligned_to_grid_count": structural["misaligned_to_grid"],
        "invalid_ohlc_count": structural["bad_ohlc"],
        "non_finite_count": structural["non_finite"],
        "non_positive_price_count": non_positive,
        "raw_gap_count": structural["gaps"],
        "largest_raw_gap_bars": structural["largest_gap_bars"],
    }


def grid_slots(first_t: float, last_t: float, timeframe: str) -> int:
    """Naive expected slot count: every timeframe step from the first bar's
    open to the last bar's open, inclusive.

    Deliberately naive — weekends and closures land in this count and are then
    separated by gap classification. It is the honest upper bound a reader
    needs to see alongside ``actual_count``; the classified gap report is what
    explains the difference. Not a per-row loop: O(1).
    """
    span = ta.duration(timeframe)
    if last_t < first_t:
        return 0
    return int((float(last_t) - float(first_t)) // span) + 1


def classify_gaps_phase1(rows: Sequence[Dict[str, Any]],
                         timeframe: str) -> Dict[str, Any]:
    """Run the existing gap engine, then map every gap onto the Phase 1
    taxonomy. Never repairs anything: the raw OHLC keeps its gaps.

    ``suspicious_max_missing_bars`` stays at the engine's default (2): gaps
    that small are UNKNOWN here, which is the phase's honest-ignorance bucket,
    and everything larger that no rule explains is DATA_GAP and blocks PASS.
    """
    engine = ta.classify_gaps(rows, timeframe)
    gaps: List[Dict[str, Any]] = []
    for g in engine["gaps"]:
        gaps.append({
            "category": map_engine_gap(g),
            "engine_category": g.get("category"),
            "reason": g.get("reason"),
            "missing_bars": g["missing_bars"],
            "duration_hours": g["duration_hours"],
            "gap_start_utc": g["gap_start_utc"],
            "gap_end_utc": g["gap_end_utc"],
            "start_weekday": g["start_weekday"],
            "stop_weekday": g["stop_weekday"],
            "start_hour_utc": g["start_hour_utc"],
            "stop_hour_utc": g["stop_hour_utc"],
            "rule_checks": g.get("rule_checks", {}),
        })

    counts = {category: 0 for category in PHASE1_CATEGORIES}
    for g in gaps:
        counts[g["category"]] += 1

    largest: Optional[Dict[str, Any]] = None
    if gaps:
        largest = max(gaps, key=lambda g: g["missing_bars"])

    return {
        "gaps": gaps,
        "gap_count": len(gaps),
        "counts": counts,
        "largest_gap": largest,
        "daily_close_hours_utc": engine.get("daily_close_hours_utc", []),
    }


def check_isolation(symbol: str, timeframe: str,
                    manifest_entry: Optional[Dict[str, Any]]) -> List[str]:
    """Dataset isolation: the manifest must agree with the file's identity.

    Files are one symbol, one timeframe by construction (``<SYMBOL>_<TF>.json``);
    the failure mode to catch is a manifest entry that declares something else
    — the mixing the phase forbids would otherwise be invisible.
    """
    problems: List[str] = []
    if manifest_entry is None:
        problems.append(
            f"{symbol}_{timeframe}: no manifest entry — the file's provenance "
            f"is unrecorded (re-run scripts/fetch_training_candles.py)")
        return problems
    if str(manifest_entry.get("symbol", symbol)).upper() != symbol.upper():
        problems.append(
            f"{symbol}_{timeframe}: manifest declares symbol "
            f"{manifest_entry.get('symbol')!r} — datasets would be mixed")
    if str(manifest_entry.get("timeframe", timeframe)).upper() != timeframe.upper():
        problems.append(
            f"{symbol}_{timeframe}: manifest declares timeframe "
            f"{manifest_entry.get('timeframe')!r} — timeframes would be mixed")
    return problems


def merge_manifest(existing: Optional[Dict[str, Any]],
                   updated_files: Dict[str, Dict[str, Any]],
                   fetched_at: str) -> Dict[str, Any]:
    """Merge a fetch run's file entries into the manifest, deterministically.

    Re-running a fetch must not erase entries for files it did not fetch this
    run (a partial run — one symbol failing, a KeyboardInterrupt — would
    otherwise drop provenance for data that is still perfectly good on disk).
    Entries present in both are replaced wholesale by the new run's entry;
    entries only in the existing manifest are kept untouched. Same inputs in,
    same manifest out — no clock-dependent ordering inside the structure.
    """
    manifest: Dict[str, Any] = dict(existing or {})
    files = dict(manifest.get("files", {}))
    files.update(updated_files)
    manifest["files"] = files
    manifest["fetched_at"] = fetched_at
    return manifest


def coverage_years(first_t: float, last_t: float) -> float:
    """Span of a dataset in years (365.25-day years), for the coverage report."""
    return (float(last_t) - float(first_t)) / (365.25 * 86400.0)


def _pass_reasons(structure: Dict[str, Any], fields: Dict[str, Any],
                  gaps: Dict[str, Any], isolation: List[str],
                  timeframe: str) -> List[str]:
    """Every concrete reason the dataset is not PASS, or an empty list."""
    reasons: List[str] = []
    if structure["duplicate_count"]:
        reasons.append(f"{structure['duplicate_count']} duplicate timestamp(s)")
    if structure["unsorted_count"]:
        reasons.append(f"{structure['unsorted_count']} out-of-order timestamp(s)")
    if structure["misaligned_to_grid_count"]:
        reasons.append(
            f"{structure['misaligned_to_grid_count']} candle(s) off the "
            f"{timeframe} grid")
    if structure["invalid_ohlc_count"]:
        reasons.append(f"{structure['invalid_ohlc_count']} invalid OHLC row(s)")
    if structure["non_finite_count"]:
        reasons.append(f"{structure['non_finite_count']} non-finite value row(s)")
    if structure["non_positive_price_count"]:
        reasons.append(
            f"{structure['non_positive_price_count']} row(s) with a price <= 0")
    if fields["rows_missing_required_fields"]:
        reasons.append(
            f"{fields['rows_missing_required_fields']} row(s) missing required "
            f"fields {fields['missing_field_counts']}")
    if gaps["counts"][CATEGORY_DATA_GAP]:
        reasons.append(
            f"{gaps['counts'][CATEGORY_DATA_GAP]} unexplained DATA_GAP window(s)")
    reasons.extend(isolation)
    return reasons


def dataset_integrity_report(rows: Sequence[Dict[str, Any]], symbol: str,
                             timeframe: str,
                             manifest_entry: Optional[Dict[str, Any]] = None,
                             ) -> Dict[str, Any]:
    """The Phase 1 integrity block for one dataset (spec section 8), plus the
    phase's PASS/FAIL verdict and the reasons behind it.

    PASS requires: no duplicates, sorted, on-grid, clean OHLC (finite,
    positive, high/low consistent), no missing required fields, and zero
    DATA_GAP gaps. UNKNOWN gaps do not block — they are reported, which is
    what "classified" means here — but they stay visible in the counts.
    Coverage (COMPLETE vs INCOMPLETE) is deliberately NOT part of the verdict:
    a short dataset with clean data PASSES integrity and is still honestly
    reported INCOMPLETE by the caller.
    """
    first_t = float(rows[0]["t"]) if rows else None
    last_t = float(rows[-1]["t"]) if rows else None

    fields = check_fields(rows)
    structure = validate_ohlc_rows(rows, timeframe)
    if rows:
        gaps = classify_gaps_phase1(rows, timeframe)
    else:
        gaps = {"gaps": [], "gap_count": 0,
                "counts": {c: 0 for c in PHASE1_CATEGORIES},
                "largest_gap": None, "daily_close_hours_utc": []}
    isolation = check_isolation(symbol, timeframe, manifest_entry)

    expected = grid_slots(first_t, last_t, timeframe) if rows else 0
    actual = structure["row_count"]

    reasons = _pass_reasons(structure, fields, gaps, isolation, timeframe)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start_timestamp": utc_iso(first_t) if first_t is not None else None,
        "end_timestamp": utc_iso(last_t) if last_t is not None else None,
        "row_count": actual,
        "duplicate_count": structure["duplicate_count"],
        "invalid_ohlc_count": structure["invalid_ohlc_count"],
        "missing_count": expected - actual,
        "expected_count": expected,
        "actual_count": actual,
        "gap_count": gaps["gap_count"],
        "largest_gap": gaps["largest_gap"],
        "gap_counts_by_category": gaps["counts"],
        "gaps": gaps["gaps"],
        "unsorted_count": structure["unsorted_count"],
        "misaligned_to_grid_count": structure["misaligned_to_grid_count"],
        "non_finite_count": structure["non_finite_count"],
        "non_positive_price_count": structure["non_positive_price_count"],
        "rows_missing_required_fields": fields["rows_missing_required_fields"],
        "missing_field_counts": fields["missing_field_counts"],
        "unknown_fields": fields["unknown_fields"],
        "isolation_problems": isolation,
        "validation_status": "PASS" if not reasons else "FAIL",
        "failure_reasons": reasons,
    }
