"""Spread/volume features: available only when the source data actually is,
and never derived from anything past the decision point.

MT5 discarded `spread` and `real_volume` in every export before this session;
these tests pin that the new fields degrade honestly on old-shaped data rather
than silently reporting zero as if it were measured.
"""

from __future__ import annotations

import random

import pytest

from analysis.features import microstructure_features as mf


def candles(n=200, seed=1, spread_base=1.5, with_spread=True, with_real_volume=True,
           constant_real_volume=False):
    rng = random.Random(seed)
    out, price = [], 1.10
    for i in range(n):
        price *= 1 + rng.gauss(0, 0.001)
        row = {"t": float(i * 14400), "open": price, "high": price + 0.001,
               "low": price - 0.001, "close": price, "volume": 100.0 + rng.random() * 50}
        if with_spread:
            row["spread"] = spread_base + rng.random() * 2.0
        if with_real_volume:
            row["real_volume"] = 0.0 if constant_real_volume else rng.random() * 1000
        out.append(row)
    return out


class TestFieldAvailability:
    def test_missing_spread_is_reported_not_available(self):
        report = mf.field_availability(candles(with_spread=False))
        assert report["spread"] == "NOT AVAILABLE"

    def test_present_spread_is_reported_available(self):
        report = mf.field_availability(candles())
        assert report["spread"] == "AVAILABLE"

    def test_short_series_is_insufficient_history(self):
        report = mf.field_availability(candles(n=20))
        assert report["spread"] == "INSUFFICIENT HISTORY"

    def test_constant_real_volume_is_flagged_as_no_information(self):
        """The realistic FX/CFD case: the field exists but every broker tick
        reports 0, which is not the same as the field being absent."""
        report = mf.field_availability(candles(constant_real_volume=True))
        assert "constant" in report["real_volume"]

    def test_an_empty_series_is_not_available(self):
        assert mf.field_availability([])["spread"] == "NOT AVAILABLE"


class TestBuildFeatures:
    def test_none_when_spread_is_absent(self):
        h4 = candles(with_spread=False)
        assert mf.build_microstructure_features(h4, timestamp=h4[-1]["t"], atr=0.001) is None

    def test_none_on_short_history(self):
        h4 = candles(n=50)
        assert mf.build_microstructure_features(h4, timestamp=h4[-1]["t"], atr=0.001) is None

    def test_a_valid_series_returns_all_declared_features(self):
        h4 = candles()
        named = mf.build_microstructure_features(h4, timestamp=h4[-1]["t"], atr=0.001)
        assert named is not None
        assert set(named) == set(mf.FEATURE_NAMES)

    def test_constant_real_volume_yields_the_neutral_marker_not_a_measurement(self):
        h4 = candles(constant_real_volume=True)
        named = mf.build_microstructure_features(h4, timestamp=h4[-1]["t"], atr=0.001)
        assert named["real_volume_zscore"] == 0.0
        assert named["real_volume_percentile"] == 50.0

    def test_missing_real_volume_also_yields_the_neutral_marker(self):
        h4 = candles(with_real_volume=False)
        named = mf.build_microstructure_features(h4, timestamp=h4[-1]["t"], atr=0.001)
        assert named is not None  # spread still present, so the row still builds
        assert named["real_volume_zscore"] == 0.0

    def test_zero_atr_does_not_raise(self):
        h4 = candles()
        named = mf.build_microstructure_features(h4, timestamp=h4[-1]["t"], atr=0.0)
        assert named["spread_atr"] == 0.0

    def test_vector_matches_named_order(self):
        h4 = candles()
        vec = mf.build_vector(h4, timestamp=h4[-1]["t"], atr=0.001)
        named = mf.build_microstructure_features(h4, timestamp=h4[-1]["t"], atr=0.001)
        assert vec == [named[n] for n in mf.FEATURE_NAMES]


class TestNoLookAhead:
    def test_mutating_the_future_does_not_change_the_features(self):
        h4 = candles(n=300)
        cut = 200
        from analysis.features import timeframe_alignment as ta

        at = ta.decision_time(h4, cut - 1, "H4")
        visible = ta.closed_slice(h4, "H4", at)
        before = mf.build_microstructure_features(visible, timestamp=at, atr=0.001)

        tampered = [dict(c) for c in h4]
        for j in range(cut, len(tampered)):
            tampered[j]["spread"] *= 50.0
            if "real_volume" in tampered[j]:
                tampered[j]["real_volume"] *= 50.0

        visible_after = ta.closed_slice(tampered, "H4", at)
        after = mf.build_microstructure_features(visible_after, timestamp=at, atr=0.001)
        assert before == after

    def test_mutating_the_past_does_change_the_features(self):
        """Non-vacuity control for the test above."""
        h4 = candles(n=300)
        cut = 200
        from analysis.features import timeframe_alignment as ta

        at = ta.decision_time(h4, cut - 1, "H4")
        visible = ta.closed_slice(h4, "H4", at)
        before = mf.build_microstructure_features(visible, timestamp=at, atr=0.001)

        tampered = [dict(c) for c in h4]
        for j in range(cut):
            tampered[j]["spread"] *= 50.0

        visible_after = ta.closed_slice(tampered, "H4", at)
        after = mf.build_microstructure_features(visible_after, timestamp=at, atr=0.001)
        assert before != after
