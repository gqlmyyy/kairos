"""Tests for scripts/audit_xauusd_m30_gaps.py — all mocked, no live MT5.

The tool itself needs a live MT5 terminal, which this sandbox does not
have, so every test here exercises its logic (connection handling, gap
evidence collection, and the conservative 5-way classification) against a
FakeMT5 stand-in. No test asserts what any REAL gap's root cause is — that
can only come from an actual run against the live terminal; these tests
only prove the tool's own reasoning is sound and, critically, that it
degrades honestly (UNKNOWN, not a forced guess) whenever evidence is thin.
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
    def __init__(self, connected=True, company="MetaQuotes Ltd.", path=r"C:\Program Files\MetaTrader 5"):
        self.connected = connected
        self.company = company
        self.path = path


class _SymbolInfo:
    def __init__(self, visible=True, description="Gold vs US Dollar"):
        self.visible = visible
        self.description = description


class FakeMT5:
    """Stands in for the MetaTrader5 module, matching the surface this
    script actually calls: initialize, terminal_info, symbol_info,
    symbol_select, copy_rates_range, copy_ticks_range, version,
    last_error, shutdown. `session_attrs` controls whether any
    session-named attribute exists on this fake module at all — by default
    none does, matching the real installed package (5.0.5735)."""

    TIMEFRAME_M30 = "M30"
    TIMEFRAME_M1 = "M1"
    COPY_TICKS_ALL = -1
    __version__ = "5.0.5735"

    def __init__(self, *, initialize_ok=True, symbol_known=True, symbol_selectable=True,
                 m30_bars_in_gap=0, m1_bars_in_gap=0, tick_status="OK",
                 ticks_before=3, ticks_after=3, ticks_in_gap=0,
                 historical_only_bars=0, mt5_extra_in_gap=0):
        self._initialize_ok = initialize_ok
        self._symbol_known = symbol_known
        self._symbol_selectable = symbol_selectable
        self.m30_bars_in_gap = m30_bars_in_gap
        self.m1_bars_in_gap = m1_bars_in_gap
        self.tick_status = tick_status
        self.ticks_before = ticks_before
        self.ticks_after = ticks_after
        self.ticks_in_gap = ticks_in_gap
        self.mt5_extra_in_gap = mt5_extra_in_gap  # extra M30 bars only MT5 has
        self.shutdown_called = False

    def initialize(self):
        return self._initialize_ok

    def terminal_info(self):
        return _TerminalInfo() if self._initialize_ok else None

    def symbol_info(self, symbol):
        if not self._symbol_known:
            return None
        return _SymbolInfo()

    def symbol_select(self, symbol, enable):
        if self._symbol_selectable:
            self._symbol_known = True
        return self._symbol_selectable

    def version(self):
        return (500, 6116, "14 Aug 2026")

    def last_error(self):
        return (1, "no error")

    def copy_rates_range(self, symbol, timeframe, start, end):
        """`start`/`end` are the WINDOW bounds (gap +/- audit.MARGIN, by
        construction of the caller) — bars are placed relative to the
        actual gap, derived from that margin, not the window midpoint, so
        "in gap" vs "boundary" placement is exact and test-predictable."""
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
        # the last bar before the gap, and the first bar after it — both
        # inside the window, both real "boundary" candidates for a
        # matching-timestamp comparison against the historical file
        bars.append({"time": (gap_start - span).replace(tzinfo=timezone.utc).timestamp(),
                     "open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.5})
        bars.append({"time": gap_end.replace(tzinfo=timezone.utc).timestamp(),
                     "open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.5})
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
            ticks.append({"time": (gap_start + timedelta(minutes=i)).replace(tzinfo=timezone.utc).timestamp()})
        for i in range(self.ticks_after):
            ticks.append({"time": (gap_end + timedelta(minutes=i)).replace(tzinfo=timezone.utc).timestamp()})
        return ticks

    def shutdown(self):
        self.shutdown_called = True


class FakeMT5WithSession(FakeMT5):
    """A hypothetical future package version exposing SOME session-named
    attribute — used only to prove detection works generically, not to
    assert any particular interpretation of its output."""

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


# ---------------------------------------------------------------------------
# Session API detection — the actual bug this rewrite fixes
# ---------------------------------------------------------------------------

class TestSessionDetection:
    def test_a_package_with_no_session_attribute_is_reported_unavailable(self):
        """Matches the real installed package: symbol_info_session_trade
        does not exist, and nothing here assumes it does."""
        mt5 = FakeMT5()
        assert audit.detect_session_api(mt5) == []
        start, end = gap()
        result = audit.query_session_evidence(mt5, "XAUUSD", start, end)
        assert result["status"] == "SESSION_METADATA_UNAVAILABLE"
        assert result["confirms_closed"] is None

    def test_the_old_hardcoded_call_is_gone(self):
        """Regression guard: the previous version called
        mt5.symbol_info_session_trade directly and only survived because it
        happened to catch the AttributeError. Nothing in this module may
        reference that name at all."""
        import inspect
        source = inspect.getsource(audit)
        assert "mt5.symbol_info_session_trade" not in source
        assert "getattr(mt5, \"symbol_info_session_trade\"" not in source

    def test_a_session_attribute_is_detected_when_present_but_not_auto_interpreted(self):
        mt5 = FakeMT5WithSession()
        found = audit.detect_session_api(mt5)
        assert "symbol_info_session_quote" in found
        start, end = gap()
        result = audit.query_session_evidence(mt5, "XAUUSD", start, end)
        assert result["status"] == "SESSION_METADATA_QUERIED"
        # Detected and called, but never auto-promoted to "closed" — no
        # named interpreter exists for an API this script has never seen
        # documented behavior for.
        assert result["confirms_closed"] is None


# ---------------------------------------------------------------------------
# Classification — conservative, evidence-based
# ---------------------------------------------------------------------------

class TestClassification:
    def test_mt5_has_extra_bars_the_file_lacks_is_a_pipeline_failure_high_confidence(self):
        start, end = gap()
        mt5 = FakeMT5(mt5_extra_in_gap=3)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["conclusion"] == "FETCH_PIPELINE_FAILURE"
        assert result["confidence"] == "HIGH"

    def test_mt5_and_historical_both_empty_with_no_ticks_and_no_session_is_unknown(self):
        """The real-world case actually observed: M30=0, M1=0 everywhere,
        no session API, AND no tick corroboration either — this must NOT
        be forced into CALENDAR_RULE_MISMATCH just because nothing else
        fits; it must land as UNKNOWN."""
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
        """Ticks before AND after but none inside, with no session data:
        genuine evidenced absence, but NOT a proven session closure — must
        be CALENDAR_RULE_MISMATCH at MEDIUM, never BROKER_SESSION_GAP,
        since nothing here actually confirms why."""
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0, m1_bars_in_gap=0,
                       ticks_before=5, ticks_after=5, ticks_in_gap=0)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["conclusion"] == "CALENDAR_RULE_MISMATCH"
        assert result["confidence"] == "MEDIUM"

    def test_m30_and_m1_empty_but_ticks_present_in_gap_is_not_auto_resolved(self):
        """A real contradiction (tick activity but no aggregated candles)
        must not silently become BROKER_SESSION_GAP or CALENDAR_RULE_MISMATCH."""
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0, m1_bars_in_gap=0, ticks_in_gap=4, ticks_before=2, ticks_after=2)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["conclusion"] == "UNKNOWN"

    def test_broker_session_gap_is_unreachable_without_an_interpreted_confirmation(self):
        """Even with a detected session attribute, confirms_closed stays
        None (see TestSessionDetection), so BROKER_SESSION_GAP can never be
        reached until a named interpreter is deliberately added — proving
        the classifier cannot manufacture this conclusion from an unknown
        API's raw output."""
        start, end = gap()
        mt5 = FakeMT5WithSession(m30_bars_in_gap=0, m1_bars_in_gap=0, ticks_before=0, ticks_after=0)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["conclusion"] != "BROKER_SESSION_GAP"


# ---------------------------------------------------------------------------
# Timestamp-set comparison — exact sets, not just counts
# ---------------------------------------------------------------------------

class TestTimestampSetComparison:
    def test_matching_and_only_sets_are_computed_correctly(self):
        start, end = gap()
        mt5 = FakeMT5(mt5_extra_in_gap=2)
        # matches the "last bar before the gap" boundary bar FakeMT5 always emits
        historical = [{"t": start.timestamp() - 1800.0, "open": 1, "high": 1, "low": 1, "close": 1}]
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=historical)
        assert len(result["mt5_only_in_gap"]) == 2
        assert result["matching_timestamps"] >= 1  # the shared boundary bar


class TestReadOnly:
    def test_the_tool_never_writes_a_file(self):
        import inspect
        source = inspect.getsource(audit.audit_one_gap) + inspect.getsource(audit.connect_mt5)
        assert "open(" not in source and "json.dump" not in source

    def test_shutdown_is_reachable_even_after_a_connection_error(self):
        """main()'s try/finally must call mt5.shutdown() even when
        connect_mt5 raises — verified structurally since main() needs a
        real MT5 import to run end to end."""
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
