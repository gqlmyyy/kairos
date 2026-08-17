"""Gate 1's forming-candle handling: exclude, don't fail outright.

The brief is explicit: "If the exporter includes the currently forming
candle, remove/exclude it from validation rather than treating it as
completed" and "the validator should validate the usable completed-candle
view" — the historical file is never touched, only the in-memory view
`check_completed` hands to everything downstream (field usability, gap
classification). These tests exercise that trimming directly, without MT5.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "validate_m30_candles", os.path.join(_ROOT, "scripts", "validate_m30_candles.py"))
vmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vmc)

from analysis.features import timeframe_alignment as ta  # noqa: E402

SPAN = ta.duration("M30")


def series(n, *, ends_at):
    """n candles on a clean M30 grid, newest closing exactly at `ends_at`."""
    start = ends_at - n * SPAN
    return [{"t": start + i * SPAN, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0}
            for i in range(n)]


def reset():
    vmc.failures.clear()
    vmc.warnings.clear()


class TestFormingCandleIsExcludedNotFatal:
    def test_a_fully_closed_final_candle_is_accepted_whole(self):
        reset()
        now = datetime.now(timezone.utc).timestamp()
        # newest candle's close is safely in the past
        candles = series(50, ends_at=now - 3600.0)
        usable = vmc.check_completed(candles, "XAUUSD", {})
        assert usable == candles
        assert vmc.failures == []
        assert vmc.warnings == []

    def test_a_currently_forming_final_candle_is_excluded_not_fatal(self):
        reset()
        now = datetime.now(timezone.utc).timestamp()
        # newest candle opens 10 minutes ago on a 30-minute timeframe: still forming
        candles = series(50, ends_at=now + (SPAN - 600.0))
        usable = vmc.check_completed(candles, "XAUUSD", {})
        assert len(usable) == len(candles) - 1
        assert usable == candles[:-1]
        assert vmc.failures == [], "a forming candle must be excluded, not fail Gate 1 outright"
        assert any("still forming" in w for w in vmc.warnings)

    def test_multiple_trailing_forming_candles_are_all_excluded(self):
        """Defensive: if more than one trailing candle is somehow still
        forming, trim all of them, not just the last."""
        reset()
        now = datetime.now(timezone.utc).timestamp()
        candles = series(50, ends_at=now + 3 * SPAN)
        usable = vmc.check_completed(candles, "XAUUSD", {})
        assert len(usable) == len(candles) - 3
        assert vmc.failures == []

    def test_everything_forming_fails_closed_with_nothing_to_validate(self):
        reset()
        now = datetime.now(timezone.utc).timestamp()
        candles = series(5, ends_at=now + 10 * SPAN)
        usable = vmc.check_completed(candles, "XAUUSD", {})
        assert usable == []
        assert vmc.failures, "an entirely-forming file has nothing usable and must say so"

    def test_a_candle_provisional_at_export_time_is_excluded_even_though_now_closed(self):
        """The file's own manifest says a bar was still forming when MT5
        handed it over — its OHLC may be provisional even though real time
        has since passed its close. Excluded, not trusted, not fatal."""
        reset()
        now = datetime.now(timezone.utc).timestamp()
        candles = series(50, ends_at=now - 3600.0)  # all genuinely closed by now
        fetched_at = datetime.fromtimestamp(
            float(candles[-1]["t"]) + SPAN - 60.0, tz=timezone.utc).isoformat()
        manifest = {"files": {"XAUUSD_M30": {"fetched_at": fetched_at}}}
        usable = vmc.check_completed(candles, "XAUUSD", manifest)
        assert usable == candles[:-1]
        assert vmc.failures == []
        assert any("provisional" in w or "forming when exported" in w for w in vmc.warnings)

    def test_the_historical_file_itself_is_never_touched(self):
        """check_completed must not mutate its input — only the returned
        view is trimmed."""
        reset()
        now = datetime.now(timezone.utc).timestamp()
        candles = series(50, ends_at=now + (SPAN - 600.0))
        original_length = len(candles)
        vmc.check_completed(candles, "XAUUSD", {})
        assert len(candles) == original_length

    def test_no_lookahead_the_verdict_depends_only_on_now_and_the_data(self):
        """Two runs against the same file at the same moment must agree —
        trimming is a pure function of (series, now, manifest), never of
        anything downstream that hasn't been computed yet."""
        reset()
        now = datetime.now(timezone.utc).timestamp()
        candles = series(50, ends_at=now + (SPAN - 600.0))
        first = vmc.check_completed(list(candles), "XAUUSD", {})
        reset()
        second = vmc.check_completed(list(candles), "XAUUSD", {})
        assert first == second
