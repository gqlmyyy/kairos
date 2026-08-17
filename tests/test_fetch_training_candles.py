"""The exporter must never let a still-forming candle into training data.

A real run reported a manifest whose batch-level `fetched_at` was hours
before the file's own newest candle's close — proof that dropping a fixed
array position (`rates[:-1]`) is not a safe way to identify the forming bar.
`_closed_by` replaces that with a direct timestamp comparison against real
UTC "now", which is what these tests hold to account. This module loads
without MetaTrader5 installed (the import lives inside `fetch()`, not at
module scope), so it runs in this sandbox the same as on the Windows target.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

spec = importlib.util.spec_from_file_location(
    "fetch_training_candles",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "fetch_training_candles.py"),
)
fetch_training_candles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_training_candles)


class TestClosedBy:
    def test_a_bar_whose_window_has_fully_elapsed_is_closed(self):
        row = {"time": 1_000_000.0}
        now = 1_000_000.0 + 1800.0 + 1.0  # M30 span plus a second to spare
        assert fetch_training_candles._closed_by(row, "M30", now) is True

    def test_a_bar_whose_window_has_not_elapsed_is_forming(self):
        row = {"time": 1_000_000.0}
        now = 1_000_000.0 + 900.0  # only half the M30 span has passed
        assert fetch_training_candles._closed_by(row, "M30", now) is False

    def test_exactly_at_the_close_boundary_counts_as_closed(self):
        row = {"time": 1_000_000.0}
        now = 1_000_000.0 + 1800.0  # exactly one M30 span later
        assert fetch_training_candles._closed_by(row, "M30", now) is True

    def test_one_second_before_the_close_boundary_is_still_forming(self):
        row = {"time": 1_000_000.0}
        now = 1_000_000.0 + 1800.0 - 1.0
        assert fetch_training_candles._closed_by(row, "M30", now) is False

    @pytest.mark.parametrize("timeframe,span", [("M30", 1800), ("H1", 3600), ("H4", 14400)])
    def test_every_fetched_timeframe_uses_its_own_span(self, timeframe, span):
        row = {"time": 1_000_000.0}
        just_closed = 1_000_000.0 + span
        still_forming = just_closed - 1.0
        assert fetch_training_candles._closed_by(row, timeframe, just_closed) is True
        assert fetch_training_candles._closed_by(row, timeframe, still_forming) is False


class TestModuleLoadsWithoutMT5:
    def test_fetch_and_mt5_timeframe_are_defined(self):
        """Regression guard: fetch() must not import MetaTrader5 at module
        scope, or this file (and any test importing it) breaks on any
        machine without the MT5 terminal installed — including this one."""
        assert hasattr(fetch_training_candles, "fetch")
        assert hasattr(fetch_training_candles, "_mt5_timeframe")
        assert fetch_training_candles.TIMEFRAMES == ("H4", "H1", "M30")
