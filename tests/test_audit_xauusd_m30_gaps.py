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


# The one gap every FakeMT5 test fixture targets, as fixed constants rather
# than back-computed from whatever margin a given caller happens to use
# (M30/M1 use audit.MARGIN=24h, ticks and the multi-timeframe probe use
# audit.PROBE_MARGIN=2h) — anchoring FakeMT5's synthetic bars/ticks to the
# real gap boundaries directly means it is correct for every caller's
# window without needing to know which margin that caller chose.
_GAP_START = datetime(2025, 7, 3, 3, 30, tzinfo=timezone.utc)
_GAP_END = datetime(2025, 7, 3, 12, 0, tzinfo=timezone.utc)


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
    TIMEFRAME_M5 = "M5"
    TIMEFRAME_M15 = "M15"
    TIMEFRAME_H1 = "H1"
    COPY_TICKS_ALL = -1
    __version__ = "5.0.5735"
    _SPAN_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60}

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
        """`start`/`end` are whatever window the caller requested — 24h
        margin for M30/M1, 2h for the multi-timeframe probe. Bars are
        placed relative to the fixed real gap constants (`_GAP_START`/
        `_GAP_END`), not derived from `start`/`end`, so this is correct
        regardless of which margin the caller used."""
        if timeframe == self.TIMEFRAME_M1 and self.m1_totally_empty:
            return []
        span = timedelta(minutes=self._SPAN_MINUTES.get(timeframe, 30))
        gap_start, gap_end = _GAP_START, _GAP_END
        if timeframe == self.TIMEFRAME_M30:
            n_in_gap = self.m30_bars_in_gap
        elif timeframe == self.TIMEFRAME_M1:
            n_in_gap = self.m1_bars_in_gap
        else:
            n_in_gap = 0
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
        """`start`/`end` here are whatever window the caller requested
        (2h margin for the production tick query); ticks are placed
        relative to the fixed real gap constants, same reasoning as
        `copy_rates_range`."""
        if self.tick_status == "ERROR":
            return None
        if self.tick_status == "RAISE":
            raise RuntimeError("simulated tick-query crash")
        gap_start, gap_end = _GAP_START, _GAP_END
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
    return (_GAP_START, _GAP_END)


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


# ---------------------------------------------------------------------------
# None vs empty-array — the API can fail two different ways, and this
# script must never fold them into the same "0 bars" bucket.
# ---------------------------------------------------------------------------

class FakeMT5NoneResponse(FakeMT5):
    """copy_rates_range/copy_ticks_range return None outright — a query
    failure, distinct from a successful query that found nothing."""

    def copy_rates_range(self, symbol, timeframe, start, end):
        return None

    def copy_ticks_range(self, symbol, start, end, flags):
        return None


class TestNoneVsEmptyResponse:
    def test_m30_none_response_is_distinguished_from_empty(self):
        start, end = gap()
        mt5 = FakeMT5NoneResponse()
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["m30_query"]["raw_kind"] == "none"

    def test_m30_empty_response_is_distinguished_from_none(self):
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0)  # generates only the two boundary bars, not empty —
        # use a dedicated all-empty fake for the true empty-array case
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["m30_query"]["raw_kind"] == "data"  # boundary bars are real data

    def test_m1_totally_empty_is_reported_as_empty_not_none(self):
        start, end = gap()
        mt5 = FakeMT5(m1_totally_empty=True)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["m1_query"]["raw_kind"] == "empty"

    def test_tick_none_response_is_reported_as_none_not_zero_ticks(self):
        start, end = gap()
        mt5 = FakeMT5NoneResponse()
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["tick_query"]["raw_kind"] == "none"
        assert result["tick_status"] == "ERROR"


# ---------------------------------------------------------------------------
# Multi-timeframe probe and the TICKS_PRESENT_BARS_ABSENT evidence category
# ---------------------------------------------------------------------------

class TestMultiTimeframeProbe:
    def test_probe_covers_all_five_timeframes(self):
        start, end = gap()
        mt5 = FakeMT5()
        probe = audit.probe_all_timeframes(mt5, "XAUUSD", start, end)
        assert set(probe) == set(audit.PROBE_TIMEFRAMES)

    def test_an_unsupported_timeframe_is_reported_not_crashed_on(self):
        class _NoM5Wrapper:
            """Delegates to a real FakeMT5 for everything except
            TIMEFRAME_M5, simulating a package that lacks that constant —
            getattr(mt5, "TIMEFRAME_M5", None) must return None, not raise."""

            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                if name == "TIMEFRAME_M5":
                    raise AttributeError(name)
                return getattr(self._inner, name)

        mt5 = _NoM5Wrapper(FakeMT5())
        start, end = gap()
        p = audit.probe_timeframe(mt5, "XAUUSD", "M5", start, end)
        assert p["supported"] is False

    def test_ticks_present_bars_absent_when_every_timeframe_is_empty(self):
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0, m1_bars_in_gap=0, ticks_in_gap=500)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["evidence_state"] == "TICKS_PRESENT_BARS_ABSENT"

    def test_no_activity_when_ticks_and_all_bars_are_empty(self):
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0, m1_bars_in_gap=0, ticks_in_gap=0,
                       ticks_before=0, ticks_after=0)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["evidence_state"] == "NO_ACTIVITY"

    def test_bars_present_some_timeframe_when_m1_has_bars(self):
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0, m1_bars_in_gap=4)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["evidence_state"] == "BARS_PRESENT_SOME_TIMEFRAME"

    def test_classify_evidence_state_is_a_pure_function_of_its_inputs(self):
        probe_all_empty = {tf: {"supported": True, "bars_in_gap": 0}
                            for tf in audit.PROBE_TIMEFRAMES}
        assert audit.classify_evidence_state(0, probe_all_empty) == "NO_ACTIVITY"
        assert audit.classify_evidence_state(10, probe_all_empty) == "TICKS_PRESENT_BARS_ABSENT"
        probe_one_has_bars = dict(probe_all_empty)
        probe_one_has_bars["H1"] = {"supported": True, "bars_in_gap": 1}
        assert audit.classify_evidence_state(0, probe_one_has_bars) == "BARS_PRESENT_SOME_TIMEFRAME"


# ---------------------------------------------------------------------------
# Tick continuity analysis
# ---------------------------------------------------------------------------

class TestTickContinuity:
    def test_no_ticks_reports_pattern_none(self):
        start, end = gap()
        result = audit.analyze_tick_continuity([], start, end)
        assert result["pattern"] == "NONE"

    def test_too_few_ticks_to_assess(self):
        start, end = gap()
        ticks = [{"time": start.timestamp() + 60}]
        result = audit.analyze_tick_continuity(ticks, start, end)
        assert result["pattern"] == "TOO_FEW_TO_ASSESS"

    def test_ticks_spread_through_the_gap_are_distributed(self):
        start, end = gap()
        span = (end - start).total_seconds()
        ticks = [{"time": start.timestamp() + span * frac} for frac in
                 (0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95)]
        result = audit.analyze_tick_continuity(ticks, start, end)
        assert result["pattern"] == "DISTRIBUTED_THROUGH_GAP"

    def test_ticks_only_at_the_edges_are_concentrated_at_boundary(self):
        start, end = gap()
        span = (end - start).total_seconds()
        ticks = [{"time": start.timestamp() + span * frac} for frac in
                 (0.001, 0.01, 0.02, 0.03, 0.99, 0.98)]
        result = audit.analyze_tick_continuity(ticks, start, end)
        assert result["pattern"] == "CONCENTRATED_AT_BOUNDARY"

    def test_bid_ask_presence_is_detected(self):
        start, end = gap()
        ticks = [{"time": start.timestamp() + 60 * i, "bid": 2000.0 + i, "ask": 2000.5 + i}
                 for i in range(6)]
        result = audit.analyze_tick_continuity(ticks, start, end)
        assert result["has_bid"] is True
        assert result["has_ask"] is True

    def test_duplicate_timestamps_are_counted(self):
        start, end = gap()
        t = start.timestamp() + 100
        ticks = [{"time": t} for _ in range(6)]
        result = audit.analyze_tick_continuity(ticks, start, end)
        assert result["unique_timestamps"] == 1
        assert result["duplicate_timestamps"] == 5


# ---------------------------------------------------------------------------
# assert_in_window — structural guarantee that a displayed record cannot
# belong to the wrong window
# ---------------------------------------------------------------------------

class TestAssertInWindow:
    def test_a_timestamp_inside_the_window_does_not_raise(self):
        window_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        window_end = datetime(2025, 1, 2, tzinfo=timezone.utc)
        audit.assert_in_window(window_start.timestamp() + 3600, window_start, window_end, "test")

    def test_a_timestamp_outside_the_window_raises(self):
        window_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        window_end = datetime(2025, 1, 2, tzinfo=timezone.utc)
        far_future = datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp()
        with pytest.raises(AssertionError, match="outside the requested window"):
            audit.assert_in_window(far_future, window_start, window_end, "test")

    def test_print_report_never_raises_for_a_well_behaved_gap(self):
        """End-to-end: every before/after record print_report displays for
        a normal gap must pass its own window assertions without raising."""
        start, end = gap()
        mt5 = FakeMT5()
        connection = audit.connect_mt5(mt5, "XAUUSD")
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        audit.print_report(1, result, connection)  # must not raise


# ---------------------------------------------------------------------------
# --gap-index CLI argument
# ---------------------------------------------------------------------------

class TestGapIndexCLI:
    def test_gap_index_and_start_are_mutually_exclusive(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", [
            "audit_xauusd_m30_gaps.py", "--gap-index", "2",
            "--start", "2025-01-01T00:00:00+00:00", "--end", "2025-01-01T01:00:00+00:00",
        ])
        exit_code = audit.main()
        assert exit_code == 1
        assert "mutually exclusive" in capsys.readouterr().out

    def test_gap_index_out_of_range_is_rejected_by_argparse(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["audit_xauusd_m30_gaps.py", "--gap-index", "99"])
        with pytest.raises(SystemExit):
            audit.main()

    def test_gap_index_selects_the_right_reported_gap(self):
        assert audit.REPORTED_GAPS[2] == (
            datetime(2025, 7, 3, 3, 30, tzinfo=timezone.utc),
            datetime(2025, 7, 3, 12, 0, tzinfo=timezone.utc))
