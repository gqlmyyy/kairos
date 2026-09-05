"""Phase 1 data-foundation tests: integrity validation, gap taxonomy, schema,
isolation, and manifest reproducibility — all synthetic, no MT5, no network.

The gap-classification expectations here mirror the documented mapping in
``analysis/data/phase1.py``: the existing engine
(``analysis/features/timeframe_alignment.py``) classifies, Phase 1 renames.
Where a real broker series would decide a gap by its own recurring structure
(daily-close hours, holiday evidence), these fixtures construct the simplest
series that still exercises the same rule.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.data import phase1  # noqa: E402
from analysis.features import timeframe_alignment as ta  # noqa: E402

_FETCH_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "scripts", "fetch_training_candles.py")
_spec = importlib.util.spec_from_file_location("fetch_training_candles", _FETCH_PATH)
fetch_training_candles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_training_candles)

M15 = 15 * 60
H1 = 60 * 60


def row(t: float, o: float = 1.1000, h: float = 1.1010, l: float = 1.0990,
        c: float = 1.1005, spread: float = 1.0, volume: float = 100.0) -> dict:
    return {"t": float(t), "open": o, "high": h, "low": l, "close": c,
            "volume": volume, "spread": spread, "real_volume": 0.0}


def grid(start: datetime, count: int, step: int) -> list:
    """`count` clean bars on the `step` grid from `start` (UTC epoch opens)."""
    t0 = start.timestamp()
    return [row(t0 + i * step, o=1.1 + 0.0001 * i, h=1.1005 + 0.0001 * i,
                l=1.0995 + 0.0001 * i, c=1.1002 + 0.0001 * i)
            for i in range(count)]


def drop_bars(rows: list, start: datetime, n: int, step: int) -> list:
    """Remove `n` consecutive grid slots starting at `start` — a clean gap."""
    t_start = start.timestamp()
    return [r for r in rows
            if not (t_start <= r["t"] < t_start + n * step)]


def series_with_gap(start: datetime, before: int, gap_at: datetime,
                    gap_bars: int, after: int, step: int) -> list:
    rows = grid(start, before + after, step)
    return drop_bars(rows, gap_at, gap_bars, step)


def entry_for(rows: list, symbol: str, timeframe: str) -> dict:
    """A manifest entry that matches the fixture.

    Provenance is required for a PASS verdict — a file without it fails
    isolation by design — so tests that expect PASS carry a matching entry.
    """
    return {"path": f"data/historical/{symbol}_{timeframe}.json",
            "symbol": symbol, "timeframe": timeframe, "bars": len(rows)}


class TestSchema:
    def test_every_required_field_present_in_clean_rows(self):
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 50, M15)
        fields = phase1.check_fields(rows)
        assert fields["rows_missing_required_fields"] == 0
        assert fields["missing_field_counts"] == {}
        assert fields["unknown_fields"] == []

    def test_a_row_missing_spread_is_a_schema_violation_not_a_zero(self):
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 50, M15)
        del rows[7]["spread"]
        report = phase1.dataset_integrity_report(rows, "EURUSD", "M15")
        assert report["rows_missing_required_fields"] == 1
        assert report["missing_field_counts"] == {"spread": 1}
        assert report["validation_status"] == "FAIL"

    def test_unknown_fields_are_reported_not_silently_accepted(self):
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 10, M15)
        rows[0]["junk_column"] = 1.0
        assert phase1.check_fields(rows)["unknown_fields"] == ["junk_column"]

    def test_schema_version_declares_the_stored_row_shape(self):
        assert phase1.DATA_SCHEMA_VERSION == "kairos-candles-v2"
        assert set(phase1.REQUIRED_FIELDS) == {
            "t", "open", "high", "low", "close", "volume", "spread"}


class TestOhlcValidity:
    def test_high_below_max_open_close_is_invalid(self):
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 30, M15)
        rows[5]["high"] = min(rows[5]["open"], rows[5]["close"]) - 0.001
        report = phase1.dataset_integrity_report(rows, "EURUSD", "M15")
        assert report["invalid_ohlc_count"] == 1
        assert report["validation_status"] == "FAIL"

    def test_low_above_min_open_close_is_invalid(self):
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 30, M15)
        rows[3]["low"] = max(rows[3]["open"], rows[3]["close"]) + 0.001
        assert phase1.dataset_integrity_report(
            rows, "EURUSD", "M15")["invalid_ohlc_count"] == 1

    def test_non_finite_prices_are_counted(self):
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 30, M15)
        rows[4]["close"] = float("nan")
        report = phase1.dataset_integrity_report(rows, "EURUSD", "M15")
        assert report["non_finite_count"] == 1
        assert report["validation_status"] == "FAIL"

    def test_non_positive_prices_are_counted(self):
        """The check the engine does not make; Phase 1 adds it explicitly."""
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 30, M15)
        rows[2]["open"] = 0.0
        report = phase1.dataset_integrity_report(rows, "EURUSD", "M15")
        assert report["non_positive_price_count"] == 1
        assert report["validation_status"] == "FAIL"

    def test_a_clean_series_passes_every_ohlc_check(self):
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 200, M15)
        report = phase1.dataset_integrity_report(rows, "EURUSD", "M15")
        assert report["invalid_ohlc_count"] == 0
        assert report["non_finite_count"] == 0
        assert report["non_positive_price_count"] == 0


class TestOrderingAndDuplicates:
    def test_duplicate_timestamp_is_detected_and_fails(self):
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 50, M15)
        rows.insert(10, dict(rows[9]))
        report = phase1.dataset_integrity_report(rows, "EURUSD", "M15")
        assert report["duplicate_count"] == 1
        assert report["validation_status"] == "FAIL"

    def test_out_of_order_rows_are_detected_and_fail(self):
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 50, M15)
        rows[10], rows[11] = rows[11], rows[10]
        report = phase1.dataset_integrity_report(rows, "EURUSD", "M15")
        assert report["unsorted_count"] >= 1
        assert report["validation_status"] == "FAIL"

    def test_off_grid_timestamps_are_detected(self):
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 50, M15)
        rows[6]["t"] += 60  # one minute off the M15 grid
        report = phase1.dataset_integrity_report(rows, "EURUSD", "M15")
        assert report["misaligned_to_grid_count"] == 1
        assert report["validation_status"] == "FAIL"


class TestGapTaxonomy:
    """Every fixture drops whole grid slots — no synthetic candles are ever
    created to fill a gap, mirroring what the fetch pipeline must not do."""

    MON = datetime(2025, 12, 1, tzinfo=timezone.utc)      # a Monday
    FRI_1300 = datetime(2025, 12, 5, 13, 0, tzinfo=timezone.utc)
    NEXT_MON = datetime(2025, 12, 8, tzinfo=timezone.utc)

    def test_m15_is_a_known_timeframe(self):
        assert ta.duration("M15") == M15
        assert "M15" in phase1.PHASE1_TIMEFRAMES

    def test_the_weekend_gap_is_WEEKEND(self):
        # Bars Mon Dec 1 .. Mon Dec 8, with [Fri 13:15, Mon 00:00) removed —
        # the ordinary FX weekend. The last present bar before the gap is
        # Friday afternoon, so the weekly-close rule fires.
        rows = grid(self.MON, 8 * 96, M15)                    # .. next Mon 23:45
        rows = drop_bars(rows, self.FRI_1300 + timedelta(minutes=15), 235, M15)
        report = phase1.dataset_integrity_report(
            rows, "EURUSD", "M15", entry_for(rows, "EURUSD", "M15"))
        weekend = [g for g in report["gaps"]
                   if g["category"] == phase1.CATEGORY_WEEKEND]
        assert weekend, f"expected a WEEKEND gap, got {report['gap_counts_by_category']}"
        assert report["validation_status"] == "PASS"  # weekends never block

    def test_a_holiday_gap_is_MARKET_CLOSED(self):
        # Christmas 2025 fell on a Thursday: bars stop Wednesday 23:00 and
        # resume Friday 00:00. The gap overlaps the evidenced holiday.
        wed_end = datetime(2025, 12, 24, 23, 0, tzinfo=timezone.utc)
        fri_start = datetime(2025, 12, 26, tzinfo=timezone.utc)
        n_before = int((wed_end.timestamp() - self.MON.timestamp()) // H1) + 1
        rows = grid(self.MON, n_before, H1)
        rows += grid(fri_start, 48, H1)
        report = phase1.dataset_integrity_report(
            rows, "EURUSD", "H1", entry_for(rows, "EURUSD", "H1"))
        closed = [g for g in report["gaps"]
                  if g["category"] == phase1.CATEGORY_MARKET_CLOSED]
        assert closed, f"expected MARKET_CLOSED, got {report['gap_counts_by_category']}"
        assert report["validation_status"] == "PASS"

    def test_a_small_unexplained_gap_is_UNKNOWN_and_does_not_block(self):
        # 2 missing M15 bars mid-week: under the engine's suspicious threshold.
        tue = datetime(2025, 12, 2, 14, 0, tzinfo=timezone.utc)
        rows = grid(self.MON, 3 * 96, M15)
        rows = drop_bars(rows, tue, 2, M15)
        report = phase1.dataset_integrity_report(
            rows, "EURUSD", "M15", entry_for(rows, "EURUSD", "M15"))
        assert report["gap_counts_by_category"][phase1.CATEGORY_UNKNOWN] == 1
        assert report["validation_status"] == "PASS"  # reported, not hidden

    def test_a_large_unexplained_gap_is_DATA_GAP_and_blocks(self):
        # 10 missing M15 bars mid-week morning: no rule explains it.
        tue = datetime(2025, 12, 2, 9, 0, tzinfo=timezone.utc)
        rows = grid(self.MON, 3 * 96, M15)
        rows = drop_bars(rows, tue, 10, M15)
        report = phase1.dataset_integrity_report(
            rows, "EURUSD", "M15", entry_for(rows, "EURUSD", "M15"))
        assert report["gap_counts_by_category"][phase1.CATEGORY_DATA_GAP] == 1
        assert report["validation_status"] == "FAIL"
        assert any("DATA_GAP" in r for r in report["failure_reasons"])

    def test_largest_gap_is_reported_with_its_window(self):
        tue = datetime(2025, 12, 2, 9, 0, tzinfo=timezone.utc)
        rows = grid(self.MON, 3 * 96, M15)
        rows = drop_bars(rows, tue, 10, M15)
        largest = phase1.dataset_integrity_report(
            rows, "EURUSD", "M15", entry_for(rows, "EURUSD", "M15")
        )["largest_gap"]
        assert largest["missing_bars"] == 10
        assert largest["category"] == phase1.CATEGORY_DATA_GAP
        assert largest["gap_start_utc"] and largest["gap_end_utc"]

    def test_the_engine_mapping_table_is_total(self):
        # Every category/reason pair the engine emits must map somewhere in
        # the Phase 1 taxonomy — no gap may fall through unmapped.
        for engine_cat, reason, expected in [
                ("EXPECTED_MARKET_GAP", "weekly_close", "WEEKEND"),
                ("EXPECTED_MARKET_GAP", "known_holiday", "MARKET_CLOSED"),
                ("EXPECTED_MARKET_GAP", "broker_maintenance", "MARKET_CLOSED"),
                ("EXPECTED_MARKET_GAP", "something_new", "EXPECTED"),
                ("SUSPICIOUS_GAP", "thin_liquidity_or_irregular_session", "UNKNOWN"),
                ("DATA_ERROR", "unexplained_missing_candles", "DATA_GAP")]:
            assert phase1.map_engine_gap(
                {"category": engine_cat, "reason": reason}) == expected
            assert expected in phase1.PHASE1_CATEGORIES


class TestDatasetIsolation:
    ENTRY = {"path": "data/historical/EURUSD_M15.json",
             "symbol": "EURUSD", "timeframe": "M15", "bars": 50}

    def test_matching_manifest_entry_is_isolated(self):
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 50, M15)
        report = phase1.dataset_integrity_report(
            rows, "EURUSD", "M15", self.ENTRY)
        assert report["isolation_problems"] == []
        assert report["validation_status"] == "PASS"

    def test_a_manifest_entry_declaring_another_symbol_fails(self):
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 50, M15)
        entry = dict(self.ENTRY, symbol="GBPUSD")
        report = phase1.dataset_integrity_report(rows, "EURUSD", "M15", entry)
        assert any("symbol" in p for p in report["isolation_problems"])
        assert report["validation_status"] == "FAIL"

    def test_a_manifest_entry_declaring_another_timeframe_fails(self):
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 50, M15)
        entry = dict(self.ENTRY, timeframe="H1")
        report = phase1.dataset_integrity_report(rows, "EURUSD", "M15", entry)
        assert any("timeframe" in p for p in report["isolation_problems"])
        assert report["validation_status"] == "FAIL"

    def test_a_file_without_provenance_fails(self):
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 50, M15)
        report = phase1.dataset_integrity_report(rows, "EURUSD", "M15", None)
        assert report["isolation_problems"]
        assert report["validation_status"] == "FAIL"


class TestManifestReproducibility:
    """fetch -> validate -> report must be re-runnable without corrupting or
    losing anything: the manifest merge is the contract under test."""

    A = {"symbol": "EURUSD", "timeframe": "H1", "bars": 100}
    B = {"symbol": "GBPUSD", "timeframe": "M15", "bars": 200}

    def test_merge_into_empty_manifest(self):
        merged = phase1.merge_manifest(None, {"EURUSD_H1": self.A}, "t0")
        assert merged["files"] == {"EURUSD_H1": self.A}
        assert merged["fetched_at"] == "t0"

    def test_a_partial_run_keeps_untouched_entries(self):
        first = phase1.merge_manifest(None, {"EURUSD_H1": self.A}, "t0")
        second = phase1.merge_manifest(first, {"GBPUSD_M15": self.B}, "t1")
        assert second["files"]["EURUSD_H1"] == self.A      # kept, not erased
        assert second["files"]["GBPUSD_M15"] == self.B     # added
        assert second["fetched_at"] == "t1"

    def test_refetched_entries_are_replaced_wholesale(self):
        first = phase1.merge_manifest(None, {"EURUSD_H1": self.A}, "t0")
        newer = dict(self.A, bars=150, source="mt5")
        second = phase1.merge_manifest(first, {"EURUSD_H1": newer}, "t1")
        assert second["files"]["EURUSD_H1"] == newer

    def test_merge_is_deterministic(self):
        one = phase1.merge_manifest(None, {"EURUSD_H1": self.A}, "t0")
        two = phase1.merge_manifest(None, {"EURUSD_H1": self.A}, "t0")
        assert one == two


class TestLoaderCompatibility:
    """The Phase 1 files must keep loading through the existing research
    source — that is what 'extend, don't fork' means in practice."""

    def test_phase1_rows_load_through_kairos_historical_source(self, tmp_path):
        from analysis.research.candles import KairosHistoricalSource
        rows = grid(datetime(2025, 12, 1, tzinfo=timezone.utc), 64, M15)
        path = tmp_path / "EURUSD_M15.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        source = KairosHistoricalSource(tmp_path)
        assert source.available("EURUSD", "M15")
        assert source.provides_spread, "Phase 1 rows carry spread"
        df = source.load("EURUSD", "M15")
        assert len(df) == 64
        assert list(df["timestamp"])[-1] > list(df["timestamp"])[0]
        assert df["spread"].notna().all()


class TestFetchScriptExtension:
    """The fetch script gains deep-fetch capability without changing its
    existing contract (the existing test file pins TIMEFRAMES)."""

    def test_existing_timeframes_tuple_is_untouched(self):
        assert fetch_training_candles.TIMEFRAMES == ("H4", "H1", "M30")

    def test_m15_is_fetchable_and_countable(self):
        class _FakeMT5:  # attribute mirror: the dict lookup reads all four
            TIMEFRAME_M15 = 15
            TIMEFRAME_M30 = 30
            TIMEFRAME_H1 = 16385
            TIMEFRAME_H4 = 16388

        assert fetch_training_candles._mt5_timeframe(_FakeMT5(), "M15") == 15
        assert fetch_training_candles.count_for("M15", 5) == 5 * 25600

    def test_fetch_range_exists_for_deep_history(self):
        assert callable(fetch_training_candles.fetch_range)

    def test_rows_from_rates_builds_the_phase1_schema(self):
        now = 1_000_000.0 + 4 * M15 + 1  # all four bars closed a second ago

        class _Rate(dict):
            pass  # _closed_by/row build only need mapping access

        rates = [_Rate(time=1_000_000.0 + i * M15, open=1.1, high=1.2,
                       low=1.0, close=1.15, tick_volume=7, spread=2,
                       real_volume=0) for i in range(4)]
        rows, dropped = fetch_training_candles._rows_from_rates(
            rates, "M15", now)
        assert dropped == 0
        assert [r["t"] for r in rows] == sorted(r["t"] for r in rows)
        for r in rows:
            assert set(phase1.REQUIRED_FIELDS) <= set(r)
            assert r["volume"] == 7 and r["spread"] == 2
