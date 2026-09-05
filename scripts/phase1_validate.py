"""PHASE 1 gate: validate the nine historical datasets and write the report.

Scope, deliberately narrow: read-only over the candle files, integrity
validation, gap classification, and the metadata/report the phase requires.
Nothing here builds a dataset, labels a trade, trains anything, or touches
the live path.

Usage (Windows machine, where data/historical lives)::

    python scripts/phase1_validate.py

Exit code 0 = every dataset present and PASS (coverage INCOMPLETE is allowed
and must be documented, never hidden). Exit code 1 = something must be fixed
or understood first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.data import phase1  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("phase1_validate")


def load_manifest(candles_dir: str) -> dict:
    path = os.path.join(candles_dir, "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_rows(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array of candle rows")
    return raw


def coverage_block(report: dict, entry: dict) -> dict:
    """COMPLETE vs INCOMPLETE, with the shortfall stated in plain numbers.

    Coverage is measured from the data itself (first/last bar), not from what
    was requested. A dataset covering the full ~5-year target is COMPLETE;
    anything shorter is INCOMPLETE and the block says how much is there, what
    window is missing, and why (terminal history depth is the only cause this
    pipeline can produce; anything else would come from the fetch manifest).
    """
    first = report.get("start_timestamp")
    last = report.get("end_timestamp")
    if not first or not last:
        return {"coverage_status": "MISSING", "years_covered": 0.0}
    t0 = datetime.fromisoformat(first).timestamp()
    t1 = datetime.fromisoformat(last).timestamp()
    years = phase1.coverage_years(t0, t1)
    # 5 years minus a small tolerance for the window edges and market
    # closures at the boundary weekends.
    complete = years >= phase1.TARGET_YEARS - 0.1
    block = {
        "coverage_status": "COMPLETE" if complete else "INCOMPLETE",
        "years_covered": round(years, 2),
        "target_years": phase1.TARGET_YEARS,
    }
    if not complete:
        if entry.get("depth_limited"):
            block["reason"] = (
                "terminal history depth for this timeframe ends at "
                f"{str(entry.get('actual_start', '?'))[:10]} -- the shortfall "
                f"is real and unfilled (no synthetic candles, no forward-fill)")
        else:
            block["reason"] = "dataset does not span the full target window"
        if entry.get("requested_start"):
            block["requested_start"] = entry["requested_start"]
    return block


def metadata_block(report: dict, entry: dict, coverage: dict) -> dict:
    """The per-dataset metadata the phase requires (spec section 11).

    Every field is read from the manifest/report, never from hidden constants
    buried in code: source, timezone and schema_version travel with the data.
    """
    return {
        "symbol": report["symbol"],
        "timeframe": report["timeframe"],
        "source": entry.get("source", "unknown"),
        "downloaded_at": entry.get("fetched_at"),
        "data_start": report["start_timestamp"],
        "data_end": report["end_timestamp"],
        "row_count": report["row_count"],
        "schema_version": entry.get("schema_version", phase1.DATA_SCHEMA_VERSION),
        "timezone": entry.get("timezone", "UTC"),
        "gap_summary": {
            "gap_count": report["gap_count"],
            "by_category": report["gap_counts_by_category"],
            "largest_gap": report["largest_gap"],
        },
        "coverage": coverage,
        "validation_status": report["validation_status"],
        "path": entry.get("path"),
    }


def _failed_dataset(symbol: str, timeframe: str, reasons: list,
                    coverage_status: str = "MISSING") -> dict:
    return {
        "symbol": symbol, "timeframe": timeframe,
        "validation_status": "FAIL",
        "failure_reasons": reasons,
        "coverage": {"coverage_status": coverage_status},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", default=os.path.join("data", "historical"))
    parser.add_argument("--reports", default=os.path.join("reports", "phase1"))
    parser.add_argument("--symbols", nargs="*", default=list(phase1.PHASE1_SYMBOLS))
    parser.add_argument("--timeframes", nargs="*",
                        default=list(phase1.PHASE1_TIMEFRAMES))
    args = parser.parse_args()

    os.makedirs(os.path.join(args.reports, "metadata"), exist_ok=True)
    manifest = load_manifest(args.candles)
    checked_at = datetime.now(timezone.utc).isoformat()

    datasets: dict = {}
    updated_entries: dict = {}
    for symbol in args.symbols:
        for timeframe in args.timeframes:
            key = f"{symbol}_{timeframe}"
            path = os.path.join(args.candles, f"{key}.json")
            entry = manifest.get("files", {}).get(key)

            if not os.path.exists(path):
                datasets[key] = _failed_dataset(symbol, timeframe, [
                    f"{path} does not exist -- run "
                    f"scripts/fetch_training_candles.py --start ..."])
                continue
            try:
                rows = load_rows(path)
            except (OSError, ValueError) as exc:
                datasets[key] = _failed_dataset(symbol, timeframe, [str(exc)])
                continue

            # Manifest/file agreement is part of integrity: a file whose bar
            # count no longer matches its provenance has changed after the
            # fact and its history cannot be trusted.
            extra_reasons = []
            if entry is None:
                extra_reasons.append("no manifest entry -- provenance unrecorded")
            elif entry.get("bars") not in (None, len(rows)):
                extra_reasons.append(
                    f"manifest says {entry.get('bars')} bars, file holds "
                    f"{len(rows)} -- the file changed after it was fetched")

            report = phase1.dataset_integrity_report(rows, symbol, timeframe, entry)
            report["failure_reasons"] = extra_reasons + report["failure_reasons"]
            if extra_reasons:
                report["validation_status"] = "FAIL"
            # The full gap list STAYS in the JSON report: a DATA_GAP verdict
            # must be auditable row by row (the printed summary is the short
            # form; this file is the evidence).

            coverage = coverage_block(report, entry or {})
            report["coverage"] = coverage
            datasets[key] = report

            meta = metadata_block(report, entry or {}, coverage)
            with open(os.path.join(args.reports, "metadata", f"{key}.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)

            if entry is not None:
                enriched = dict(entry)
                enriched["validation_status"] = report["validation_status"]
                enriched["coverage_status"] = coverage["coverage_status"]
                enriched["years_covered"] = coverage.get("years_covered")
                enriched["validation_checked_at"] = checked_at
                updated_entries[key] = enriched

    # Manifest entries gain their validation verdict (metadata only — the
    # candle files themselves are never written by this script).
    if updated_entries:
        merged = phase1.merge_manifest(manifest, updated_entries, checked_at)
        with open(os.path.join(args.candles, "manifest.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2)

    consolidated = {
        "phase": "PHASE 1 — DATA FOUNDATION",
        "generated_at": checked_at,
        "source": "mt5 (local MetaTrader 5 terminal via data.market.mt5_session)",
        "schema_version": phase1.DATA_SCHEMA_VERSION,
        "timezone": "UTC",
        "candles_dir": args.candles,
        "datasets": datasets,
    }
    report_path = os.path.join(args.reports, "data_integrity_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(consolidated, fh, indent=2)

    # ------------------------- the human report -------------------------
    print("=" * 70)
    print("PHASE 1 — DATA FOUNDATION")
    print("=" * 70)
    passed = failed = 0
    for symbol in args.symbols:
        for timeframe in args.timeframes:
            key = f"{symbol}_{timeframe}"
            block = datasets.get(key)
            if block is None:
                continue
            status = block["validation_status"]
            cov = block.get("coverage", {})
            counts = block.get("gap_counts_by_category", {})
            gap_line = "n/a"
            if counts:
                parts = [f"{c}={n}" for c, n in counts.items() if n]
                gap_line = f"{block.get('gap_count', 0)} ({', '.join(parts) or 'none'})"
            largest = block.get("largest_gap")
            if largest:
                max_gap = (f"{largest['missing_bars']} bars "
                           f"({largest['gap_start_utc'][:16]} -> {largest['gap_end_utc'][:16]}, "
                           f"{largest['category']})")
            else:
                max_gap = "none"
            print(f"\n{symbol} {timeframe}:")
            print(f"  Rows:       {block.get('row_count', 0)}")
            print(f"  Start:      {block.get('start_timestamp')}")
            print(f"  End:        {block.get('end_timestamp')}")
            print(f"  Coverage:   {cov.get('years_covered', 0)} years "
                  f"({cov.get('coverage_status', 'MISSING')})")
            print(f"  Gaps:       {gap_line}")
            print(f"  Max Gap:    {max_gap}")
            print(f"  Validation: {status}")
            if status == "PASS":
                passed += 1
            else:
                failed += 1
                for reason in block.get("failure_reasons", []):
                    print(f"    FAIL: {reason}")

    print("\n" + "-" * 70)
    print(f"Datasets: {passed} PASS, {failed} FAIL "
          f"(report: {report_path})")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

