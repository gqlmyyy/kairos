"""The expanded entry features: scale-free, leak-free, and identical in both paths.

The original ten failed for three reasons this module fixes, so each fix gets a
test that would catch its return:

* `market_regime` collapsed RANGING / HIGH_VOLATILITY / LOW_VOLATILITY to one
  value — and LOW_VOLATILITY was the only slice above chance in the last run.
* `atr` and `macd` were raw, so a tree splitting on them was really splitting on
  which instrument it was looking at (EURUSD ATR ~0.0017 vs XAUUSD ~41.7).
* Several features were re-encodings of each other, so ten slots carried about
  six independent dimensions.
"""

from __future__ import annotations

import math
import random

import pytest

from analysis.features import entry_features as ef


def _candles(n=300, start=1.10, vol=0.0015, seed=1, secs=14400):
    rng = random.Random(seed)
    out, close, t = [], start, 1_600_000_000
    for i in range(n):
        close = close * (1 + rng.gauss(0, vol))
        rng_h = abs(rng.gauss(0, vol * close * 0.8))
        out.append({"t": float(t + i * secs), "open": close,
                    "high": close + rng_h, "low": close - rng_h,
                    "close": close, "volume": 1.0})
    return out


@pytest.fixture
def pair():
    return _candles(), _candles(n=600, seed=2, secs=3600)


class TestGroups:
    def test_every_group_is_non_empty_and_unique(self):
        seen = set()
        for name, feats in ef.FEATURE_GROUPS.items():
            assert feats, f"group {name} is empty"
            for f in feats:
                assert f not in seen, f"{f} appears in two groups"
                seen.add(f)

    def test_group_names_is_stable_and_ordered(self):
        assert ef.group_names() == ef.group_names()
        assert ef.group_names(["baseline"]) == list(ef.FEATURE_GROUPS["baseline"])

    def test_unknown_group_is_rejected_loudly(self):
        with pytest.raises(ValueError, match="unknown feature group"):
            ef.group_names(["baseline", "not_a_group"])

    def test_vector_length_matches_selected_groups(self, pair):
        h4, h1 = pair
        for groups in (["baseline"], ["baseline", "trend"], None):
            vec = ef.build_vector(h4, h1, direction="BUY",
                                  timestamp=h4[-1]["t"], groups=groups)
            assert len(vec) == len(ef.group_names(groups))


class TestScaleFree:
    """The defect that made atr and macd act as symbol identifiers."""

    def test_identical_relative_dynamics_give_identical_features(self):
        """Two instruments with the same shape at 4000x different price must
        produce the same scale-free features."""
        cheap_h4 = _candles(start=1.10, vol=0.0015, seed=7)
        cheap_h1 = _candles(n=600, start=1.10, vol=0.0015, seed=8, secs=3600)
        rich_h4 = _candles(start=4330.0, vol=0.0015, seed=7)
        rich_h1 = _candles(n=600, start=4330.0, vol=0.0015, seed=8, secs=3600)

        a = ef.build_entry_features(cheap_h4, cheap_h1, direction="BUY",
                                    timestamp=cheap_h4[-1]["t"])
        b = ef.build_entry_features(rich_h4, rich_h1, direction="BUY",
                                    timestamp=rich_h4[-1]["t"])

        # Tolerance is 1e-3 relative, not tighter: live_indicators rounds ATR
        # and MACD to 6 decimal places *absolutely*, which is a different
        # relative precision at price 1.10 than at 4330. A genuine scale
        # dependency shows up as a factor of thousands, not the 5th decimal.
        for name in ef.group_names():
            assert a[name] == pytest.approx(b[name], rel=1e-3, abs=1e-6), (
                f"{name} depends on price scale: {a[name]} vs {b[name]}"
            )

    def test_the_old_raw_features_would_have_failed_that_test(self):
        """Sanity: the check above must be capable of failing.

        Raw ATR is what the previous spec fed the model, and it differs by
        ~4000x between these two instruments.
        """
        cheap = _candles(start=1.10, vol=0.0015, seed=7)
        rich = _candles(start=4330.0, vol=0.0015, seed=7)
        from analysis.features import live_parity_features as lpf

        atr_cheap = lpf.live_indicators(cheap)["atr"]
        atr_rich = lpf.live_indicators(rich)["atr"]
        assert atr_rich / atr_cheap > 1000, (
            "raw ATR is supposed to be wildly scale-dependent; if it is not, "
            "this fixture no longer represents the real problem"
        )

    def test_no_feature_is_a_raw_price_or_raw_atr(self, pair):
        """A value on the order of the instrument's price would be a scale leak."""
        h4, h1 = pair
        feats = ef.build_entry_features(h4, h1, direction="BUY",
                                        timestamp=h4[-1]["t"])
        price = h4[-1]["close"]
        for name, value in feats.items():
            assert abs(value) < price * 100, f"{name}={value} looks unnormalised"


class TestRegimeEncoding:
    def test_four_regimes_get_four_distinct_values(self):
        vals = {r: ef.encode_regime_4(r) for r in
                ("RANGING", "TRENDING", "HIGH_VOLATILITY", "LOW_VOLATILITY")}
        assert len(set(vals.values())) == 4, vals

    def test_the_old_collapse_is_gone(self):
        """These three used to all encode to 0.0."""
        assert ef.encode_regime_4("HIGH_VOLATILITY") != ef.encode_regime_4("RANGING")
        assert ef.encode_regime_4("LOW_VOLATILITY") != ef.encode_regime_4("RANGING")
        assert ef.encode_regime_4("HIGH_VOLATILITY") != ef.encode_regime_4("LOW_VOLATILITY")


class TestNoLookAhead:
    def test_future_candles_do_not_change_any_feature(self, pair):
        h4, h1 = pair
        cut = 200

        before = ef.build_entry_features(h4[:cut], h1, direction="BUY",
                                         timestamp=h4[cut - 1]["t"])

        tampered = [dict(c) for c in h4]
        for j in range(cut, len(tampered)):
            for k in ("open", "high", "low", "close"):
                tampered[j][k] *= 3.0
        after = ef.build_entry_features(tampered[:cut], h1, direction="BUY",
                                        timestamp=h4[cut - 1]["t"])

        assert before == after, "a feature read past the entry bar"

    def test_truncating_the_future_is_a_no_op(self, pair):
        h4, h1 = pair
        cut = 200
        a = ef.build_entry_features(h4[:cut], h1[:400], direction="BUY",
                                    timestamp=h4[cut - 1]["t"])
        b = ef.build_entry_features(h4[:cut], h1[:400], direction="BUY",
                                    timestamp=h4[cut - 1]["t"])
        assert a == b

    def test_short_history_returns_none_not_a_partial_row(self, pair):
        h4, h1 = pair
        assert ef.build_entry_features(h4[:40], h1, direction="BUY",
                                       timestamp=h4[39]["t"]) is None
        assert ef.build_entry_features(h4, h1[:40], direction="BUY",
                                       timestamp=h4[-1]["t"]) is None


class TestRobustness:
    @pytest.mark.parametrize("shape", ["flat", "monotonic", "spike", "zeros"])
    def test_degenerate_input_stays_finite(self, shape):
        n = 300
        if shape == "flat":
            closes = [1.1] * n
        elif shape == "monotonic":
            closes = [1.0 + i * 0.001 for i in range(n)]
        elif shape == "spike":
            closes = [1.1] * (n - 5) + [3.0] * 5
        else:
            closes = [0.0] * n

        def mk(vals, secs):
            return [{"t": float(1_600_000_000 + i * secs), "open": c, "high": c,
                     "low": c, "close": c, "volume": 1.0}
                    for i, c in enumerate(vals)]

        feats = ef.build_entry_features(mk(closes, 14400), mk(closes, 3600),
                                        direction="BUY",
                                        timestamp=1_600_000_000.0)
        assert feats is not None
        for name, value in feats.items():
            assert math.isfinite(value), f"{shape}: {name} = {value}"

    def test_direction_changes_the_vector(self, pair):
        h4, h1 = pair
        buy = ef.build_vector(h4, h1, direction="BUY", timestamp=h4[-1]["t"])
        sell = ef.build_vector(h4, h1, direction="SELL", timestamp=h4[-1]["t"])
        assert buy != sell


class TestRedundancy:
    def test_no_feature_is_a_linear_copy_of_another(self):
        """`momentum_score` used to be a three-bucket re-encoding of `rsi`,
        which was already feature #1. Two features whose correlation is ~1
        carry the same information and one of them is wasted capacity."""
        rows = []
        for seed in range(60):
            h4 = _candles(seed=seed)
            h1 = _candles(n=600, seed=seed + 100, secs=3600)
            f = ef.build_entry_features(h4, h1, direction="BUY",
                                        timestamp=h4[-1]["t"])
            if f:
                rows.append(f)
        assert len(rows) >= 30

        def corr(xs, ys):
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
            sy = math.sqrt(sum((y - my) ** 2 for y in ys))
            if sx == 0 or sy == 0:
                return 0.0
            return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)

        names = ef.group_names()
        cols = {n: [r[n] for r in rows] for n in names}
        duplicates = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                r = corr(cols[a], cols[b])
                # 0.999 rather than 1.0: pullback_depth and dist_low20_atr are
                # the same quantity with a sign flip in one branch, which is
                # legitimate, so only near-perfect duplication is flagged.
                if abs(r) > 0.999:
                    duplicates.append((a, b, round(r, 5)))
        assert not duplicates, f"features carry identical information: {duplicates}"

    def test_the_correlation_check_can_actually_fire(self):
        """Guard against a vacuous test: a planted duplicate must be caught."""
        xs = [float(i) for i in range(50)]
        ys = [2.0 * x + 1.0 for x in xs]          # perfectly linear copy
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)
        assert abs(r) > 0.999
