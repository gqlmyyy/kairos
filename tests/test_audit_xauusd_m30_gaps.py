"""Tests for scripts/audit_xauusd_m30_gaps.py — all mocked, no live MT5.

The tool itself needs a live MT5 terminal, which this sandbox does not
have, so every test here exercises its logic against a FakeMT5 stand-in.
No test asserts what any REAL gap's root cause is — only a live run can
answer that. These tests prove two things: the tool's classification logic
is sound, and it degrades honestly (UNKNOWN, QUERY_RANGE_VIOLATION,
M1_HISTORY_UNAVAILABLE) whenever evidence is thin or unreliable, rather
than manufacturing a confident-sounding answer.

This file replaces an earlier version after a real run against MT5
5.0.5735 found copy_rates_range() returning an M1 bar timestamped months
after the requested historical window (2026-05-01 for a 2024-01-15
request) — proof that a returned "0 bars" cannot be trusted without first
checking the response actually stayed inside the requested range.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "audit_xauusd_m30_gaps", os.path.join(_ROOT, "scripts", "audit_xauusd_m30_gaps.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class _TerminalInfo:
    def __init__(self, connected=True, company="MetaQuotes Ltd.",
                 path=r"C:\Program Files\MetaTrader 5", maxbars=100000):
        self.connected = connected
        self.company = company
        self.path = path
        self.maxbars = maxbars


class _SymbolInfo:
    def __init__(self, visible=True, description="Gold vs US Dollar"):
        self.visible = visible
        self.description = description


class FakeMT5:
    """Stands in for the MetaTrader5 module. `m30_bars_in_gap`/
    `m1_bars_in_gap`/`ticks_in_gap` place well-behaved (in-range) evidence
    relative to the ACTUAL gap (derived from audit.MARGIN, not the window
    midpoint, so placement is exact and test-predictable).
    `m30_out_of_range`/`m1_out_of_range`/`ticks_out_of_range`, when set to
    an epoch, additionally inject ONE bar/tick at that exact out-of-window
    timestamp — reproducing the real bug this rewrite fixes."""

    TIMEFRAME_M30 = "M30"
    TIMEFRAME_M1 = "M1"
    COPY_TICKS_ALL = -1
    __version__ = "5.0.5735"

    def __init__(self, *, initialize_ok=True, symbol_known=True, symbol_selectable=True,
                 m30_bars_in_gap=0, m1_bars_in_gap=0, tick_status="OK",
                 ticks_before=3, ticks_after=3, ticks_in_gap=0,
                 mt5_extra_in_gap=0, m30_out_of_range=None, m1_out_of_range=None,
                 ticks_out_of_range=None, m1_totally_empty=False):
        self._initialize_ok = initialize_ok
        self._symbol_known = symbol_known
        self._symbol_selectable = symbol_selectable
        self.m30_bars_in_gap = m30_bars_in_gap
        self.m1_bars_in_gap = m1_bars_in_gap
        self.tick_status = tick_status
        self.ticks_before = ticks_before
        self.ticks_after = ticks_after
        self.ticks_in_gap = ticks_in_gap
        self.mt5_extra_in_gap = mt5_extra_in_gap
        self.m30_out_of_range = m30_out_of_range
        self.m1_out_of_range = m1_out_of_range
        self.ticks_out_of_range = ticks_out_of_range
        self.m1_totally_empty = m1_totally_empty
        self.shutdown_called = False

    def initialize(self):
        return self._initialize_ok

    def terminal_info(self):
        return _TerminalInfo() if self._initialize_ok else None

    def symbol_info(self, symbol):
        return _SymbolInfo() if self._symbol_known else None

    def symbol_select(self, symbol, enable):
        if self._symbol_selectable:
            self._symbol_known = True
        return self._symbol_selectable

    def version(self):
        return (500, 6116, "14 Aug 2026")

    def last_error(self):
        return (1, "no error")

    def copy_rates_range(self, symbol, timeframe, start, end):
        if timeframe == self.TIMEFRAME_M1 and self.m1_totally_empty:
            return []
        span = timedelta(minutes=30) if timeframe == self.TIMEFRAME_M30 else timedelta(minutes=1)
        gap_start = start + audit.MARGIN
        gap_end = end - audit.MARGIN
        n_in_gap = self.m30_bars_in_gap if timeframe == self.TIMEFRAME_M30 else self.m1_bars_in_gap
        bars = [{"time": (gap_start + i * span).replace(tzinfo=timezone.utc).timestamp(),
                 "open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.5}
                for i in range(n_in_gap)]
        if timeframe == self.TIMEFRAME_M30:
            for i in range(self.mt5_extra_in_gap):
                bars.append({"time": (gap_start + (n_in_gap + i) * span).replace(tzinfo=timezone.utc).timestamp(),
                             "open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.5})
        bars.append({"time": (gap_start - span).replace(tzinfo=timezone.utc).timestamp(),
                     "open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.5})
        bars.append({"time": gap_end.replace(tzinfo=timezone.utc).timestamp(),
                     "open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.5})
        out_of_range = self.m30_out_of_range if timeframe == self.TIMEFRAME_M30 else self.m1_out_of_range
        if out_of_range is not None:
            bars.append({"time": out_of_range, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0})
        return bars

    def copy_ticks_range(self, symbol, start, end, flags):
        if self.tick_status == "ERROR":
            return None
        if self.tick_status == "RAISE":
            raise RuntimeError("simulated tick-query crash")
        gap_start = start + audit.MARGIN
        gap_end = end - audit.MARGIN
        ticks = []
        for i in range(self.ticks_before):
            ticks.append({"time": (start + timedelta(minutes=i)).replace(tzinfo=timezone.utc).timestamp()})
        for i in range(self.ticks_in_gap):
            # sub-second spacing so even a large tick count (tens of
            # thousands, matching the real report) fits inside the gap
            # window instead of spilling past its end
            ticks.append({"time": (gap_start + timedelta(milliseconds=100 * i)).replace(tzinfo=timezone.utc).timestamp(),
                           "bid": 2000.0 + i * 0.1})
        for i in range(self.ticks_after):
            ticks.append({"time": (gap_end + timedelta(minutes=i)).replace(tzinfo=timezone.utc).timestamp()})
        if self.ticks_out_of_range is not None:
            ticks.append({"time": self.ticks_out_of_range})
        return ticks

    def shutdown(self):
        self.shutdown_called = True


class FakeMT5WithSession(FakeMT5):
    def symbol_info_session_quote(self, symbol, day_of_week, index):
        return (timedelta(hours=1), timedelta(hours=23))


def gap():
    return (datetime(2025, 7, 3, 3, 30, tzinfo=timezone.utc),
            datetime(2025, 7, 3, 12, 0, tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# Connection handling
# ---------------------------------------------------------------------------

class TestConnection:
    def test_initialization_failure_raises_with_a_specific_reason(self):
        mt5 = FakeMT5(initialize_ok=False)
        with pytest.raises(RuntimeError, match="initialize"):
            audit.connect_mt5(mt5, "XAUUSD")

    def test_symbol_missing_and_unselectable_raises(self):
        mt5 = FakeMT5(symbol_known=False, symbol_selectable=False)
        with pytest.raises(RuntimeError, match="symbol_select"):
            audit.connect_mt5(mt5, "XAUUSD")

    def test_symbol_missing_but_selectable_recovers(self):
        mt5 = FakeMT5(symbol_known=False, symbol_selectable=True)
        connection = audit.connect_mt5(mt5, "XAUUSD")
        assert connection["initialized"] is True
        assert connection["symbol"] == "XAUUSD"

    def test_a_healthy_connection_reports_the_expected_fields(self):
        mt5 = FakeMT5()
        connection = audit.connect_mt5(mt5, "XAUUSD")
        assert connection["symbol_visible"] is True
        assert connection["terminal_connected"] is True
        assert connection["package_version"] == "5.0.5735"
        assert connection["terminal_maxbars"] == 100000


# ---------------------------------------------------------------------------
# 1/2/3 — range validation: the actual bug this rewrite fixes
# ---------------------------------------------------------------------------

class TestRangeValidation:
    def test_an_m30_timestamp_outside_the_requested_range_is_flagged_and_excluded(self):
        start, end = gap()
        far_future = datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp()
        mt5 = FakeMT5(m30_out_of_range=far_future)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["m30_query"]["violation"] is True
        # the injected far-future bar must not appear in any in-gap/before/after count
        assert result["m30_bars_in_gap"] == 0

    def test_an_m1_timestamp_outside_the_requested_range_is_flagged_as_m1_history_unavailable(self):
        """The literal real bug: copy_rates_range returned an M1 bar dated
        2026-05-01 for a 2024/2025-era historical request."""
        start, end = gap()
        far_future = datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp()
        mt5 = FakeMT5(m1_out_of_range=far_future)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["m1_query"]["violation"] is True
        assert result["m1_status"] == "M1_HISTORY_UNAVAILABLE"
        assert result["m1_bars_in_gap"] == 0

    def test_a_tick_timestamp_outside_the_requested_range_is_flagged_and_excluded(self):
        start, end = gap()
        far_future = datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp()
        mt5 = FakeMT5(ticks_out_of_range=far_future)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["tick_query"]["violation"] is True

    def test_a_well_behaved_query_reports_no_violation(self):
        start, end = gap()
        mt5 = FakeMT5()
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["m30_query"]["violation"] is False
        assert result["m1_query"]["violation"] is False
        assert result["tick_query"]["violation"] is False

    def test_validate_range_excludes_out_of_range_rows_from_valid(self):
        window_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        window_end = datetime(2025, 1, 2, tzinfo=timezone.utc)
        rows = [{"time": window_start.timestamp() + 3600},
                {"time": window_end.timestamp() + 999999}]  # way outside
        result = audit._validate_range(rows, window_start, window_end, lambda r: r["time"])
        assert result["violation"] is True
        assert len(result["valid"]) == 1
        assert result["raw_count"] == 2


# ---------------------------------------------------------------------------
# 4 — MT5 has no M1 history at all for the window
# ---------------------------------------------------------------------------

class TestM1HistoryUnavailable:
    def test_a_completely_empty_m1_response_is_unavailable_not_zero(self):
        start, end = gap()
        mt5 = FakeMT5(m1_totally_empty=True)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["m1_status"] == "M1_HISTORY_UNAVAILABLE"
        assert any("M1_HISTORY_UNAVAILABLE" in line for line in result["evidence"])

    def test_m1_unavailable_blocks_the_calendar_mismatch_path(self):
        """Even with ticks corroborating before/after and zero in the gap,
        unreliable M1 coverage must prevent CALENDAR_RULE_MISMATCH — that
        conclusion requires validated M1 coverage elsewhere in the window."""
        start, end = gap()
        mt5 = FakeMT5(m1_totally_empty=True, ticks_before=5, ticks_after=5, ticks_in_gap=0)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["conclusion"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# 5 / 7 — ticks exist, M30 does not
# ---------------------------------------------------------------------------

class TestTickActivityWithoutM30:
    def test_ticks_in_gap_with_no_m30_sets_the_evidence_flag(self):
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0, ticks_in_gap=10)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["tick_activity_without_m30"] is True

    def test_ticks_without_m30_never_auto_resolves_to_broker_session_gap(self):
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0, ticks_in_gap=1073)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["conclusion"] != "BROKER_SESSION_GAP"
        assert result["conclusion"] != "FETCH_PIPELINE_FAILURE"
        assert result["conclusion"] == "UNKNOWN"

    def test_matches_the_real_reported_scale_of_activity(self):
        """Regression tied to the actual report: 1073 / 6565 / 86205 ticks
        with zero M30 must all land as UNKNOWN, none silently resolved."""
        start, end = gap()
        for n in (1073, 6565, 86205):
            mt5 = FakeMT5(m30_bars_in_gap=0, ticks_in_gap=n)
            result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
            assert result["conclusion"] == "UNKNOWN"
            assert result["ticks_in_gap"] == n

    def test_no_ticks_and_no_m30_does_not_set_the_flag(self):
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0, ticks_in_gap=0)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["tick_activity_without_m30"] is False


# ---------------------------------------------------------------------------
# 6 — exact timestamp-set comparison
# ---------------------------------------------------------------------------

class TestTimestampSetComparison:
    def test_matching_and_only_sets_are_computed_correctly(self):
        start, end = gap()
        mt5 = FakeMT5(mt5_extra_in_gap=2)
        # matches the "last bar before the gap" boundary bar FakeMT5 always emits
        historical = [{"t": start.timestamp() - 1800.0, "open": 1, "high": 1, "low": 1, "close": 1}]
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=historical)
        assert len(result["mt5_only_in_gap"]) == 2
        assert result["matching_timestamps"] >= 1

    def test_mt5_only_and_historical_only_report_actual_timestamps_not_only_counts(self):
        start, end = gap()
        mt5 = FakeMT5(mt5_extra_in_gap=1)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert isinstance(result["mt5_only_in_gap"], list)
        assert len(result["mt5_only_in_gap"]) == 1
        assert isinstance(result["mt5_only_in_gap"][0], (int, float))


# ---------------------------------------------------------------------------
# 8 — session metadata unavailable
# ---------------------------------------------------------------------------

class TestSessionDetection:
    def test_a_package_with_no_session_attribute_is_reported_unavailable(self):
        mt5 = FakeMT5()
        assert audit.detect_session_api(mt5) == []
        start, end = gap()
        result = audit.query_session_evidence(mt5, "XAUUSD", start, end)
        assert result["status"] == "SESSION_METADATA_UNAVAILABLE"
        assert result["confirms_closed"] is None

    def test_the_old_hardcoded_call_is_gone(self):
        import inspect
        source = inspect.getsource(audit)
        assert "mt5.symbol_info_session_trade" not in source

    def test_a_session_attribute_is_detected_but_not_auto_interpreted(self):
        mt5 = FakeMT5WithSession()
        found = audit.detect_session_api(mt5)
        assert "symbol_info_session_quote" in found
        start, end = gap()
        result = audit.query_session_evidence(mt5, "XAUUSD", start, end)
        assert result["status"] == "SESSION_METADATA_QUERIED"
        assert result["confirms_closed"] is None

    def test_session_unavailable_never_alone_produces_calendar_rule_mismatch(self):
        start, end = gap()
        mt5 = FakeMT5(ticks_before=0, ticks_after=0, ticks_in_gap=0)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["session"]["status"] == "SESSION_METADATA_UNAVAILABLE"
        assert result["conclusion"] != "CALENDAR_RULE_MISMATCH"


# ---------------------------------------------------------------------------
# Broader classification behavior
# ---------------------------------------------------------------------------

class TestClassification:
    def test_mt5_has_extra_bars_the_file_lacks_is_a_pipeline_failure_high_confidence(self):
        start, end = gap()
        mt5 = FakeMT5(mt5_extra_in_gap=3)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["conclusion"] == "FETCH_PIPELINE_FAILURE"
        assert result["confidence"] == "HIGH"

    def test_everything_empty_and_unreliable_is_unknown(self):
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0, m1_bars_in_gap=0, ticks_before=0, ticks_after=0, ticks_in_gap=0)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["conclusion"] == "UNKNOWN"
        assert result["confidence"] == "LOW"

    def test_historical_only_bars_are_flagged_as_corruption_medium_confidence(self):
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0)
        fake_historical = [{"t": (start + timedelta(hours=1)).timestamp(),
                             "open": 1, "high": 1, "low": 1, "close": 1}]
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=fake_historical)
        assert result["conclusion"] == "HISTORICAL_FILE_CORRUPTION"
        assert result["confidence"] == "MEDIUM"

    def test_tick_query_failure_is_never_read_as_market_closed(self):
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0, m1_bars_in_gap=0, tick_status="ERROR")
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["conclusion"] == "UNKNOWN"
        assert result["tick_status"] == "ERROR"

    def test_tick_query_crash_is_caught_and_reported_not_raised(self):
        start, end = gap()
        mt5 = FakeMT5(tick_status="RAISE")
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["tick_status"] == "ERROR"
        assert result["tick_error"] is not None

    def test_ticks_around_the_gap_but_none_inside_supports_calendar_mismatch_not_broker_gap(self):
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0, m1_bars_in_gap=0,
                       ticks_before=5, ticks_after=5, ticks_in_gap=0)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["conclusion"] == "CALENDAR_RULE_MISMATCH"
        assert result["confidence"] == "MEDIUM"

    def test_broker_session_gap_is_unreachable_without_an_interpreted_confirmation(self):
        start, end = gap()
        mt5 = FakeMT5WithSession(m30_bars_in_gap=0, m1_bars_in_gap=0, ticks_before=0, ticks_after=0)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["conclusion"] != "BROKER_SESSION_GAP"


class TestSyntheticFromTicks:
    def test_synthetic_aggregation_is_labelled_and_in_memory_only(self):
        ticks = [{"time": 1_700_000_000.0 + i * 5, "bid": 2000.0 + i * 0.01} for i in range(20)]
        bars = audit.synthetic_ohlc_from_ticks(ticks, minute_span=1)
        assert bars
        assert all(b["label"] == "SYNTHETIC_FROM_TICKS" for b in bars)

    def test_no_ticks_yields_no_synthetic_bars(self):
        assert audit.synthetic_ohlc_from_ticks([]) == []

    def test_synthetic_aggregation_is_never_written_to_disk(self):
        import inspect
        source = inspect.getsource(audit.synthetic_ohlc_from_ticks)
        assert "open(" not in source and "json.dump" not in source

    def test_the_tool_never_writes_a_file_anywhere_in_its_core_logic(self):
        import inspect
        source = (inspect.getsource(audit.audit_one_gap) + inspect.getsource(audit.connect_mt5)
                   + inspect.getsource(audit.synthetic_ohlc_from_ticks))
        assert "open(" not in source and "json.dump" not in source


class TestReadOnly:
    def test_shutdown_is_reachable_even_after_a_connection_error(self):
        import inspect
        source = inspect.getsource(audit.main)
        assert "finally" in source
        assert "shutdown" in source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_candles_in_window_is_half_open(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        candles = [{"t": start.timestamp()}, {"t": (start + timedelta(minutes=30)).timestamp()},
                   {"t": (start + timedelta(hours=1)).timestamp()}]
        result = audit.candles_in_window(candles, start, start + timedelta(minutes=30))
        assert result == [candles[0]]

    def test_naive_utc_strips_tzinfo_after_converting(self):
        aware = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        naive = audit._naive_utc(aware)
        assert naive.tzinfo is None
        assert naive == datetime(2025, 1, 1, 12, 0)

    def test_grid_point_count_is_diagnostic_arithmetic_only(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert audit.number_of_grid_points(start, start + timedelta(hours=1), 30) == 2
        assert audit.number_of_grid_points(start, start + timedelta(hours=1), 1) == 60
        assert audit.number_of_grid_points(start, start, 30) == 0

    def test_boundary_bar_picks_the_nearest_one(self):
        bars = [{"time": 100.0}, {"time": 200.0}, {"time": 300.0}]
        assert audit._boundary_bar(bars, closest_to="start")["time"] == 300.0
        assert audit._boundary_bar(bars, closest_to="end")["time"] == 100.0
        assert audit._boundary_bar([], closest_to="start") is None

    def test_tick_epoch_prefers_time_msc(self):
        assert audit.tick_epoch({"time": 100, "time_msc": 100500}) == pytest.approx(100.5)
        assert audit.tick_epoch({"time": 100}) == 100.0
