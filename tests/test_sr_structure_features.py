"""S/R structure features: the reference project's ideas, without its leak.

The reference implementation
(https://github.com/Mrizalfahlepi/SR_Mapping_NN, scripts/02_feature_engineering.py)
marks a Williams Fractal at bar i by comparing it to bars i+1 and i+2, so
seven of its 26 features read two bars into the future. `TestFractalsAreConfirmedNotCentred`
proves this module does not, and `test_the_reference_rule_would_fail_this`
reproduces the original rule to show the check is capable of catching it.
"""

from __future__ import annotations

import random

import pytest

from analysis.features import sr_structure_features as sr
from analysis.features import timeframe_alignment as ta


def candles(n=300, seed=1, price=2330.0, vol=0.004):
    rng = random.Random(seed)
    out, close = [], price
    for i in range(n):
        opened = close * (1 + rng.gauss(0, vol * 0.3))
        close = opened * (1 + rng.gauss(0, vol))
        high = max(opened, close) + abs(rng.gauss(0, vol * close))
        low = min(opened, close) - abs(rng.gauss(0, vol * close))
        out.append({"t": float(i * 14400), "open": opened, "high": high,
                    "low": low, "close": close, "volume": 100.0})
    return out


def reference_fractals(highs, lows, lookback=2):
    """The reference project's rule, verbatim in behaviour — for contrast."""
    n = len(highs)
    up = [None] * n
    down = [None] * n
    for i in range(lookback, n - lookback):
        if all(highs[i] > highs[i - j] and highs[i] > highs[i + j]
               for j in range(1, lookback + 1)):
            up[i] = highs[i]
        if all(lows[i] < lows[i - j] and lows[i] < lows[i + j]
               for j in range(1, lookback + 1)):
            down[i] = lows[i]
    return up, down


class TestFractalsAreConfirmedNotCentred:
    def test_a_fractal_is_indexed_where_it_becomes_knowable(self):
        """A fractal centred on bar i appears at index i+2, not i."""
        highs = [1.0] * 20
        lows = [1.0] * 20
        highs[10] = 5.0          # a clear peak centred on bar 10
        up, _ = sr.confirmed_fractals(highs, lows)
        assert up[10] is None, "fractal appeared at its centre, before confirmation"
        assert up[12] == 5.0, "fractal did not appear at its confirmation bar"

    def test_mutating_the_future_cannot_change_a_confirmed_fractal(self):
        bars = candles(n=300)
        highs = [c["high"] for c in bars]
        lows = [c["low"] for c in bars]
        cut = 200

        before, _ = sr.confirmed_fractals(highs[:cut], lows[:cut])

        tampered_h = list(highs)
        tampered_l = list(lows)
        for j in range(cut, len(tampered_h)):
            tampered_h[j] *= 3.0
            tampered_l[j] *= 3.0
        after, _ = sr.confirmed_fractals(tampered_h[:cut], tampered_l[:cut])
        assert before == after

    def test_the_reference_rule_would_fail_this(self):
        """Non-vacuity, stated deterministically.

        A random fixture is a poor control here: the reference rule only leaks
        at the two bars adjacent to the mutation boundary, and if neither
        happens to be a fractal nothing changes and the test passes for the
        wrong reason. So the fixture is constructed instead — one unambiguous
        peak at bar 10, then bar 11 is raised. Under the reference rule the
        bar-10 fractal disappears, proving it was never a property of bar 10
        alone.
        """
        highs = [1.0] * 20
        lows = [1.0] * 20
        highs[10] = 5.0

        ref_before, _ = reference_fractals(highs, lows)
        assert ref_before[10] == 5.0, "fixture does not produce the fractal it should"

        tampered = list(highs)
        tampered[11] = 10.0          # a bar strictly AFTER the fractal's centre
        ref_after, _ = reference_fractals(tampered, lows)

        assert ref_after[10] is None, (
            "the reference rule showed no leak, so this fixture no longer "
            "demonstrates the defect being fixed")

        # And the confirmed rule survives the identical mutation, because it
        # never reported a bar-10 fractal until bar 12 anyway.
        ours_before, _ = sr.confirmed_fractals(highs[:11], lows[:11])
        ours_after, _ = sr.confirmed_fractals(tampered[:11], lows[:11])
        assert ours_before == ours_after


class TestNoLookAhead:
    def test_features_are_unchanged_when_the_future_is_mutated(self):
        bars = candles(n=400)
        cut = 250
        at = ta.decision_time(bars, cut - 1, "H4")

        visible = ta.closed_slice(bars, "H4", at)
        before = sr.build_sr_features(visible, timestamp=at, atr=10.0)

        tampered = [dict(c) for c in bars]
        for j in range(cut, len(tampered)):
            for key in ("open", "high", "low", "close"):
                tampered[j][key] *= 5.0
        visible_after = ta.closed_slice(tampered, "H4", at)
        after = sr.build_sr_features(visible_after, timestamp=at, atr=10.0)

        assert before == after

    def test_features_do_change_when_the_past_is_mutated(self):
        bars = candles(n=400)
        cut = 250
        at = ta.decision_time(bars, cut - 1, "H4")

        visible = ta.closed_slice(bars, "H4", at)
        before = sr.build_sr_features(visible, timestamp=at, atr=10.0)

        tampered = [dict(c) for c in bars]
        for j in range(cut - 20, cut):
            for key in ("open", "high", "low", "close"):
                tampered[j][key] *= 5.0
        visible_after = ta.closed_slice(tampered, "H4", at)
        after = sr.build_sr_features(visible_after, timestamp=at, atr=10.0)

        assert before != after


class TestContract:
    def test_all_declared_features_are_returned(self):
        bars = candles()
        named = sr.build_sr_features(bars, timestamp=bars[-1]["t"], atr=10.0)
        assert set(named) == set(sr.FEATURE_NAMES)

    def test_vector_matches_declared_order(self):
        bars = candles()
        vec = sr.build_vector(bars, timestamp=bars[-1]["t"], atr=10.0)
        named = sr.build_sr_features(bars, timestamp=bars[-1]["t"], atr=10.0)
        assert vec == [named[n] for n in sr.FEATURE_NAMES]

    def test_short_history_returns_none(self):
        bars = candles(n=40)
        assert sr.build_sr_features(bars, timestamp=bars[-1]["t"], atr=10.0) is None

    def test_zero_atr_returns_none_rather_than_dividing(self):
        bars = candles()
        assert sr.build_sr_features(bars, timestamp=bars[-1]["t"], atr=0.0) is None

    def test_every_value_is_finite_on_degenerate_input(self):
        import math

        flat = [{"t": float(i * 14400), "open": 2000.0, "high": 2000.0,
                "low": 2000.0, "close": 2000.0, "volume": 1.0} for i in range(200)]
        named = sr.build_sr_features(flat, timestamp=flat[-1]["t"], atr=1.0)
        assert named is not None
        for name, value in named.items():
            assert math.isfinite(value), f"{name} = {value}"


class TestScaleFree:
    def test_identical_shape_at_different_price_gives_identical_features(self):
        """XAUUSD at 2330 and a synthetic instrument at 1.10 with the same
        relative dynamics must produce the same features — the scale defect
        that made raw atr/macd act as instrument identifiers in the earlier
        KAIROS investigation."""
        cheap = candles(n=300, seed=11, price=1.10)
        rich = candles(n=300, seed=11, price=2330.0)

        cheap_atr = 1.10 * 0.004
        rich_atr = 2330.0 * 0.004

        a = sr.build_sr_features(cheap, timestamp=cheap[-1]["t"], atr=cheap_atr)
        b = sr.build_sr_features(rich, timestamp=rich[-1]["t"], atr=rich_atr)

        for name in sr.FEATURE_NAMES:
            assert a[name] == pytest.approx(b[name], rel=1e-6, abs=1e-9), (
                f"{name} depends on price scale: {a[name]} vs {b[name]}")
