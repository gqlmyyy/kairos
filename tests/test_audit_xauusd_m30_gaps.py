"""Decision-tree tests for scripts/audit_xauusd_m30_gaps.py.

The tool itself needs a live MT5 terminal, which this sandbox does not
have, but its decision logic (audit_one_gap) is a pure function of whatever
MT5 and the historical file return — so it is tested here with a fake MT5
standing in, exercising every one of the five possible conclusions without
touching a real broker connection. This is what STEP 7 of the brief asks
for: proof the classification logic itself is sound, independent of
whether a particular real gap turns out to be one category or another.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "audit_xauusd_m30_gaps", os.path.join(_ROOT, "scripts", "audit_xauusd_m30_gaps.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class FakeMT5:
    """Stands in for the MetaTrader5 module. `m30_bars_in_gap` and
    `m1_bars_in_gap` control how many synthetic bars copy_rates_range
    reports as landing inside the gap window; `session_open` controls
    whether symbol_info_session_trade reports a published session."""

    TIMEFRAME_M30 = "M30"
    TIMEFRAME_M1 = "M1"

    def __init__(self, *, m30_bars_in_gap=0, m1_bars_in_gap=0, session_open=False):
        self.m30_bars_in_gap = m30_bars_in_gap
        self.m1_bars_in_gap = m1_bars_in_gap
        self.session_open = session_open

    def copy_rates_range(self, symbol, timeframe, start, end):
        n = self.m30_bars_in_gap if timeframe == self.TIMEFRAME_M30 else self.m1_bars_in_gap
        span = timedelta(minutes=30) if timeframe == self.TIMEFRAME_M30 else timedelta(minutes=1)
        mid = start + (end - start) / 2
        return [{"time": (mid + i * span).replace(tzinfo=timezone.utc).timestamp()}
                for i in range(n)]

    def symbol_info_session_trade(self, symbol, day_of_week, session_index):
        if not self.session_open:
            return None
        return (timedelta(0), timedelta(hours=24)) if session_index == 0 else None


def gap():
    start = datetime(2025, 7, 3, 3, 30, tzinfo=timezone.utc)
    end = datetime(2025, 7, 3, 12, 0, tzinfo=timezone.utc)
    return start, end


class TestConclusions:
    def test_mt5_has_bars_file_is_missing_them_is_a_pipeline_failure(self):
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=17)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["conclusion"] == "FETCH_PIPELINE_FAILURE"

    def test_no_mt5_bars_and_a_closed_session_is_a_broker_gap(self):
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0, m1_bars_in_gap=0, session_open=False)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["conclusion"] == "BROKER_SESSION_GAP"

    def test_no_mt5_bars_and_no_session_evidence_is_a_calendar_mismatch(self):
        """A published session exists (market nominally open) but MT5 still
        has no bars — the market data itself supports absence, but nothing
        confirms a closure, so this is not blithely accepted as expected."""
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0, m1_bars_in_gap=0, session_open=True)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["conclusion"] == "CALENDAR_RULE_MISMATCH"

    def test_mt5_has_no_m30_but_has_m1_is_unknown_not_auto_explained(self):
        """M1 activity during an M30 hole means the market was NOT closed —
        this contradicts a session-based explanation and must not silently
        become BROKER_SESSION_GAP just because M30 itself is empty."""
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0, m1_bars_in_gap=5, session_open=False)
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=[])
        assert result["conclusion"] == "UNKNOWN"

    def test_file_has_bars_mt5_does_not_is_flagged_as_corruption(self):
        start, end = gap()
        mt5 = FakeMT5(m30_bars_in_gap=0, m1_bars_in_gap=0, session_open=False)
        fake_historical = [{"t": (start + timedelta(hours=1)).timestamp(),
                             "open": 1, "high": 1, "low": 1, "close": 1}]
        result = audit.audit_one_gap(mt5, "XAUUSD", start, end, historical=fake_historical)
        assert result["conclusion"] == "HISTORICAL_FILE_CORRUPTION"

    def test_the_tool_never_touches_the_historical_file(self):
        """Read-only by construction: audit_one_gap takes the already-loaded
        candle list and returns a report dict — it has no file-write path."""
        import inspect
        source = inspect.getsource(audit.audit_one_gap)
        assert "open(" not in source and "json.dump" not in source


class TestWindowHelpers:
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
