"""No timeframe may hand a decision a candle that has not closed.

The audit found this in H4, but nothing about the defect was H4-specific: the
same "latest candle at or before t" rule leaks a day from D1 and a week from
W1. So every check here runs across every timeframe the system uses, and the
reproduction of the original bug is included as a test in its own right — a
guard against fixing the symptom while leaving the rule that produced it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from analysis.features import timeframe_alignment as ta

ALL_TIMEFRAMES = ["M15", "M30", "H1", "H4", "D1"]


def ts(y, m, d, h=0, minute=0):
    return datetime(y, m, d, h, minute, tzinfo=timezone.utc).timestamp()


def m30_grid(start, end, *, skip_ranges=()):
    """A clean M30 series from `start` to `end`, dropping any bar whose open
    falls inside a (skip_start, skip_end) range in `skip_ranges` — the way a
    real gap looks: candles simply absent, not marked."""
    span = ta.duration("M30")
    out = []
    t = start
    while t < end:
        if not any(lo <= t < hi for lo, hi in skip_ranges):
            out.append({"t": t, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0})
        t += span
    return out


def series(timeframe, n=50, start=0.0, price=1.10):
    """A clean grid-aligned series; close encodes the index so leaks are visible."""
    span = ta.duration(timeframe)
    return [{"t": float(start + i * span),
             "open": price + i, "high": price + i + 0.5,
             "low": price + i - 0.5, "close": price + i}
            for i in range(n)]


class TestCandleKnowability:
    @pytest.mark.parametrize("timeframe", ALL_TIMEFRAMES)
    def test_a_candle_is_not_knowable_at_its_own_open(self, timeframe):
        """The entry_v2 rule in one assertion.

        The candle opening at t is the one still forming; reading it at t is
        reading its close, which happens a whole bar later.
        """
        candles = series(timeframe)
        opening_now = candles[10]["t"]
        index = ta.last_closed_index(candles, timeframe, opening_now)
        assert index == 9, (
            f"at the open of bar 10 the newest closed bar is 9, got {index}")

    @pytest.mark.parametrize("timeframe", ALL_TIMEFRAMES)
    def test_a_candle_is_knowable_exactly_at_its_close(self, timeframe):
        candles = series(timeframe)
        span = ta.duration(timeframe)
        at = candles[10]["t"] + span
        assert ta.last_closed_index(candles, timeframe, at) == 10

    @pytest.mark.parametrize("timeframe", ALL_TIMEFRAMES)
    def test_one_second_before_its_close_it_is_not(self, timeframe):
        candles = series(timeframe)
        span = ta.duration(timeframe)
        at = candles[10]["t"] + span - 1
        assert ta.last_closed_index(candles, timeframe, at) == 9

    @pytest.mark.parametrize("timeframe", ALL_TIMEFRAMES)
    def test_nothing_is_knowable_before_the_first_close(self, timeframe):
        candles = series(timeframe)
        assert ta.last_closed_index(candles, timeframe, candles[0]["t"]) is None

    def test_an_empty_series_yields_nothing(self):
        assert ta.last_closed_index([], "H4", 1_000_000.0) is None


class TestClosedSlice:
    @pytest.mark.parametrize("timeframe", ALL_TIMEFRAMES)
    def test_the_slice_never_contains_an_unclosed_candle(self, timeframe):
        candles = series(timeframe)
        span = ta.duration(timeframe)
        for i in range(1, 40):
            at = candles[i]["t"] + span
            visible = ta.closed_slice(candles, timeframe, at)
            ta.assert_no_lookahead(visible, timeframe, at)
            assert visible[-1]["t"] == candles[i]["t"]

    @pytest.mark.parametrize("timeframe", ALL_TIMEFRAMES)
    def test_mutating_the_future_cannot_change_the_slice(self, timeframe):
        """The property that matters: features must not move when the future does."""
        candles = series(timeframe)
        span = ta.duration(timeframe)
        at = candles[20]["t"] + span

        before = ta.closed_slice(candles, timeframe, at)
        tampered = [dict(c) for c in candles]
        for j in range(21, len(tampered)):
            for key in ("open", "high", "low", "close"):
                tampered[j][key] *= 1000.0
        after = ta.closed_slice(tampered, timeframe, at)
        assert before == after

    def test_the_check_can_actually_fire(self):
        """Guard against a vacuous suite: a leaked candle must be caught."""
        candles = series("H4")
        at = candles[10]["t"]
        with pytest.raises(ta.AlignmentError, match="not knowable"):
            ta.assert_no_lookahead(candles[:11], "H4", at)


class TestCrossTimeframeAlignment:
    """A decision on a slow timeframe may use everything faster that has closed."""

    def test_an_h4_decision_sees_four_closed_h1_bars(self):
        h4 = series("H4", n=20)
        h1 = series("H1", n=100)
        at = ta.decision_time(h4, 5, "H4")          # close of the 6th H4 bar

        visible = ta.closed_slice(h1, "H1", at)
        ta.assert_no_lookahead(visible, "H1", at)

        # The H4 bar spans [t, t+4h); at its close, the four H1 bars inside it
        # have all closed too. The old bisect-on-open-time rule stopped at the
        # first of them and discarded three hours of legitimate information.
        assert float(visible[-1]["t"]) == h4[5]["t"] + 3 * 3600

    def test_a_d1_bar_is_not_visible_mid_day(self):
        """The defect that would have been next, had D1 been wired in."""
        d1 = series("D1", n=10)
        at = d1[3]["t"] + 6 * 3600          # six hours into the fourth day
        assert ta.last_closed_index(d1, "D1", at) == 2

    @pytest.mark.parametrize("fast,slow", [("M15", "H1"), ("M30", "H4"), ("H1", "H4"),
                                           ("H4", "D1")])
    def test_no_leak_in_any_pairing(self, fast, slow):
        slow_c = series(slow, n=12)
        fast_c = series(fast, n=2000)
        for i in range(1, 10):
            at = ta.decision_time(slow_c, i, slow)
            ta.assert_no_lookahead(ta.closed_slice(fast_c, fast, at), fast, at)
            ta.assert_no_lookahead(ta.closed_slice(slow_c, slow, at), slow, at)


class TestExecutability:
    def test_the_entry_bar_opens_at_or_after_the_decision(self):
        """A decision at a close cannot fill at that close — the price is gone."""
        h4 = series("H4", n=20)
        at = ta.decision_time(h4, 5, "H4")
        index = ta.next_executable_index(h4, "H4", at)
        assert index == 6
        assert float(h4[index]["t"]) >= at

    def test_no_executable_bar_at_the_end_of_history(self):
        h4 = series("H4", n=20)
        at = ta.decision_time(h4, 19, "H4")
        assert ta.next_executable_index(h4, "H4", at) is None

    def test_the_entry_bar_is_never_the_decision_bar(self):
        h4 = series("H4", n=30)
        for i in range(25):
            at = ta.decision_time(h4, i, "H4")
            assert ta.next_executable_index(h4, "H4", at) == i + 1


class TestSeriesValidation:
    def test_a_clean_series_reports_no_problems(self):
        report = ta.validate_series(series("H1", n=100), "H1")
        assert report["unsorted"] == 0
        assert report["duplicate_timestamps"] == 0
        assert report["misaligned_to_grid"] == 0
        assert report["bad_ohlc"] == 0
        assert report["gaps"] == 0

    def test_duplicates_are_counted(self):
        candles = series("H1", n=10)
        candles.insert(5, dict(candles[5]))
        assert ta.validate_series(candles, "H1")["duplicate_timestamps"] == 1

    def test_out_of_order_bars_are_counted(self):
        candles = series("H1", n=10)
        candles[3], candles[6] = candles[6], candles[3]
        assert ta.validate_series(candles, "H1")["unsorted"] > 0

    def test_an_off_grid_series_is_flagged(self):
        """Usually a broker offset, which silently shifts every alignment."""
        candles = [{**c, "t": c["t"] + 137.0} for c in series("H1", n=10)]
        assert ta.validate_series(candles, "H1")["misaligned_to_grid"] == 10

    def test_a_weekend_gap_is_reported_not_condemned(self):
        candles = series("H1", n=20)
        candles = candles[:10] + [{**c, "t": c["t"] + 48 * 3600} for c in candles[10:]]
        report = ta.validate_series(candles, "H1")
        assert report["gaps"] == 1
        # bar 9 sits at +9h, bar 10 moves to +58h: 49 hours apart, 48 missing.
        assert report["largest_gap_bars"] == 48

    def test_impossible_ohlc_is_flagged(self):
        candles = series("H1", n=10)
        candles[4]["high"] = candles[4]["low"] - 1.0
        assert ta.validate_series(candles, "H1")["bad_ohlc"] == 1

    def test_an_unknown_timeframe_is_rejected_loudly(self):
        with pytest.raises(ta.AlignmentError, match="unknown timeframe"):
            ta.duration("H3")


class TestDiagnoseGrid:
    def test_a_utc_aligned_series_is_reported_as_such(self):
        report = ta.diagnose_grid(series("M30", n=200, start=ts(2024, 1, 1)), "M30")
        assert report["on_utc_grid"] is True
        assert report["constant_offset"] is True

    def test_a_constant_broker_offset_is_reported_not_hidden(self):
        """A broker on server time shifts every bar by the same amount — this
        must be visible, not silently treated as 'off grid'. A whole-hour
        shift is invisible to a 30-minute-grid modulo check (an hour is two
        half-hours), so this uses a 15-minute offset, which is not."""
        candles = [{**c, "t": c["t"] + 900} for c in
                   series("M30", n=200, start=ts(2024, 1, 1))]
        report = ta.diagnose_grid(candles, "M30")
        assert report["on_utc_grid"] is False
        assert report["constant_offset"] is True
        assert report["modal_offset_hours"] == 0.25

    def test_mixed_offsets_are_not_masked_as_constant(self):
        candles = series("M30", n=200, start=ts(2024, 1, 1))
        for c in candles[100:]:
            c["t"] += 900  # half a bar, only on the back half
        report = ta.diagnose_grid(candles, "M30")
        assert report["constant_offset"] is False
        assert report["distinct_offsets"] == 2


class TestClassifyGaps:
    """Gate 1 for M30: a gap must be explained by the calendar or by its own
    recurrence, or it fails closed as DATA_ERROR. These fixtures use real
    calendar dates (2024-01-01 is a Monday) because the classifier reasons
    about actual weekdays, not offsets from an arbitrary epoch."""

    def test_the_weekly_close_is_accepted(self):
        """Friday evening to Sunday evening — the ordinary FX/CFD weekend."""
        candles = m30_grid(
            ts(2024, 1, 1), ts(2024, 1, 8),
            skip_ranges=[(ts(2024, 1, 5, 21), ts(2024, 1, 7, 22))],
        )
        result = ta.classify_gaps(candles, "M30")
        assert result["counts"] == {"EXPECTED_MARKET_GAP": 1}
        assert result["gaps"][0]["reason"].startswith("weekly close")

    def test_a_long_weekend_from_a_monday_holiday_is_still_accepted(self):
        """The weekly-close rule keys off the START (Friday), so a gap that
        runs through a Monday holiday must still be accepted without a
        separate holiday-date match for every possible long weekend."""
        candles = m30_grid(
            ts(2024, 1, 1), ts(2024, 1, 15),
            skip_ranges=[(ts(2024, 1, 5, 21), ts(2024, 1, 8, 22))],  # Fri->Mon
        )
        result = ta.classify_gaps(candles, "M30")
        assert result["counts"] == {"EXPECTED_MARKET_GAP": 1}

    def test_a_named_holiday_on_a_weekday_is_accepted(self):
        """Christmas Day 2024 is a Wednesday — not caught by the weekly-close
        rule, so it must be caught by the calendar-holiday rule instead."""
        candles = m30_grid(
            ts(2024, 12, 20), ts(2024, 12, 30),
            skip_ranges=[(ts(2024, 12, 24, 20), ts(2024, 12, 26, 4))],
        )
        result = ta.classify_gaps(candles, "M30")
        assert result["counts"] == {"EXPECTED_MARKET_GAP": 1}
        assert "holiday" in result["gaps"][0]["reason"]

    def test_a_recurring_daily_pause_is_accepted_as_one_explained_pattern(self):
        """A broker that drops one bar at the same UTC hour every weekday is
        exhibiting one explained, recurring event — not ten anomalies."""
        candles = m30_grid(
            ts(2024, 1, 8), ts(2024, 1, 20),  # two full weeks, clear of Jan 1
            skip_ranges=[
                (ts(2024, 1, d, 21), ts(2024, 1, d, 21, 30))
                for d in (8, 9, 10, 11, 12, 15, 16, 17, 18, 19)  # every weekday
            ],
        )
        result = ta.classify_gaps(candles, "M30")
        assert result["counts"].get("DATA_ERROR", 0) == 0
        assert result["counts"].get("SUSPICIOUS_GAP", 0) == 0
        assert result["counts"]["EXPECTED_MARKET_GAP"] >= 10
        assert all("recurring daily pause" in g["reason"] for g in result["gaps"]
                    if g["start_weekday"] < 5)

    def test_a_single_small_weekday_gap_is_suspicious_not_silently_accepted(self):
        """One isolated small gap must not be waved through as 'recurring' —
        recurrence needs more than one data point, or the check is vacuous."""
        candles = m30_grid(
            ts(2024, 1, 2), ts(2024, 1, 3),
            skip_ranges=[(ts(2024, 1, 2, 14), ts(2024, 1, 2, 14, 30))],
        )
        result = ta.classify_gaps(candles, "M30")
        assert result["counts"] == {"SUSPICIOUS_GAP": 1}

    def test_an_unexplained_weekday_gap_fails_closed_as_data_error(self):
        """A large, non-recurring, mid-week gap with no calendar excuse is
        exactly what Gate 1 must block — this is the failure this whole
        module exists to catch, not paper over."""
        candles = m30_grid(
            ts(2024, 1, 2), ts(2024, 1, 3),
            skip_ranges=[(ts(2024, 1, 2, 14), ts(2024, 1, 2, 18))],  # 4h hole
        )
        result = ta.classify_gaps(candles, "M30")
        assert result["counts"] == {"DATA_ERROR": 1}
        assert "not the weekly close" in result["gaps"][0]["reason"]

    def test_a_friday_daytime_outage_is_not_confused_with_the_weekly_close(self):
        """An outage starting Friday morning and lasting into the weekend is
        not the same event as the market closing for the weekend — the
        weekly-close rule requires the gap to start in the afternoon/evening,
        when the market actually closes, not merely to end on a Saturday."""
        candles = m30_grid(
            ts(2024, 1, 1), ts(2024, 1, 8),
            skip_ranges=[(ts(2024, 1, 5, 9), ts(2024, 1, 7, 22))],  # Fri 09:00 start
        )
        result = ta.classify_gaps(candles, "M30")
        assert result["counts"] == {"DATA_ERROR": 1}

    def test_no_gaps_means_no_gaps(self):
        candles = m30_grid(ts(2024, 1, 2), ts(2024, 1, 3))
        result = ta.classify_gaps(candles, "M30")
        assert result["gaps"] == []
        assert result["counts"] == {}

    def test_classification_is_deterministic_and_reads_only_timestamps(self):
        """Same series in, same classification out — no hidden state, no
        dependence on OHLC values (a gap is a timestamp-arithmetic question)."""
        candles = m30_grid(
            ts(2024, 1, 1), ts(2024, 1, 8),
            skip_ranges=[(ts(2024, 1, 5, 21), ts(2024, 1, 7, 22))],
        )
        mutated = [dict(c, close=c["close"] * 99) for c in candles]
        assert ta.classify_gaps(candles, "M30") == ta.classify_gaps(mutated, "M30")
