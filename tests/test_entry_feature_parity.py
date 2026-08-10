"""Training vs live feature parity — value-level, not just shape.

The previous entry model expected 65 features while the live path sent 10, and
XGBoost accepted the short vector silently (absent columns are treated as
`missing` and follow default branch directions). Every probability it produced
was unrelated to the trade — BUY and SELL received identical values.

A count check alone would not have caught the deeper version of that bug: the
same ten slots filled with values computed by different formulas. So these
tests compare the actual numbers, produced by the actual live functions, against
what the training pipeline would build from the same candles.

Four things are pinned:
  1. the contract itself (names, order, arity)
  2. encodings agree between training and live
  3. the recomputed indicators equal what the live client computes
  4. the two constant features stay constant, and are documented as such
"""

from __future__ import annotations

import random

import pytest

from analysis.features import live_parity_features as lpf
from analysis.models import entry_feature_spec as spec


def _make_candles(n=200, start=1.1000, seed=11, scale=0.0012):
    """A deterministic, realistic-looking series. Not market data — the point
    is that both paths see the *same* input, whatever it is."""
    rng = random.Random(seed)
    candles = []
    close = start
    t = 1_700_000_000
    for i in range(n):
        close = close * (1 + rng.gauss(0, scale) / max(start, 1e-9))
        rng_h = abs(rng.gauss(0, scale * 0.7))
        candles.append({
            "t": t + i * 3600,
            "open": close, "high": close + rng_h, "low": close - rng_h,
            "close": close, "volume": 1000.0,
        })
    return candles


class TestTheContract:
    def test_exactly_ten_features(self):
        assert spec.FEATURE_COUNT == 10
        assert len(spec.FEATURE_NAMES) == 10

    def test_names_and_order_are_the_documented_ones(self):
        assert spec.FEATURE_NAMES == (
            "rsi", "atr", "macd", "trend_strength", "trend_score",
            "momentum_score", "volatility_score", "market_regime",
            "session", "direction",
        )

    def test_live_module_reexports_the_spec_not_its_own_copy(self):
        """A second hardcoded list in the inference module is how the original
        drift happened."""
        from analysis.models.xgboost_v2_inference import LIVE_FEATURE_NAMES

        assert LIVE_FEATURE_NAMES is spec.FEATURE_NAMES

    def test_builder_returns_the_right_arity(self):
        vec = spec.build_feature_vector(
            rsi=55.0, atr=0.0018, macd=0.0002, trend_strength=0.0,
            trend_score=70.0, momentum_score=65.0, volatility_score=55.0,
            market_regime="TRENDING", session="london", direction="BUY",
        )
        assert len(vec) == spec.FEATURE_COUNT
        assert all(isinstance(v, float) for v in vec)

    def test_as_named_dict_round_trips(self):
        vec = spec.build_feature_vector(
            rsi=55.0, atr=0.0018, macd=0.0002, trend_strength=0.0,
            trend_score=70.0, momentum_score=65.0, volatility_score=55.0,
            market_regime="TRENDING", session="london", direction="BUY",
        )
        named = spec.as_named_dict(vec)
        assert list(named.keys()) == list(spec.FEATURE_NAMES)
        assert named["direction"] == 1.0
        assert named["rsi"] == 55.0

    def test_as_named_dict_rejects_a_wrong_length_vector(self):
        with pytest.raises(ValueError):
            spec.as_named_dict([1.0] * 9)


class TestEncodingParity:
    @pytest.mark.parametrize("regime,expected", [
        ("TRENDING", 1.0), ("trending", 1.0),
        ("RANGING", 0.0), ("ranging", 0.0),
        # Both fall to the default — an inherited collision, pinned so it stays
        # identical on both sides rather than being fixed on one only.
        ("HIGH_VOLATILITY", 0.0), ("LOW_VOLATILITY", 0.0),
        ("UNKNOWN", 0.0), (None, 0.0), ("", 0.0),
    ])
    def test_regime_encoding(self, regime, expected):
        assert spec.encode_regime(regime) == expected

    @pytest.mark.parametrize("session,expected", [
        ("asia", 0.0), ("london", 1.0), ("new_york", 2.0),
        ("Asia", 0.0), ("London", 1.0), ("NY", 2.0),
        ("weekend", 0.0), (None, 0.0),
    ])
    def test_session_encoding(self, session, expected):
        assert spec.encode_session(session) == expected

    @pytest.mark.parametrize("direction,expected", [
        ("BUY", 1.0), ("SELL", 0.0), ("buy", 0.0), (None, 0.0),
    ])
    def test_direction_encoding(self, direction, expected):
        """Case sensitivity is inherited and deliberate: main.py passes upper
        case. Pinned so training encodes it the same way."""
        assert spec.encode_direction(direction) == expected

    @pytest.mark.parametrize("hour,expected", [
        (0, "asia"), (3, "asia"), (6, "asia"),
        (7, "london"), (10, "london"), (12, "london"),
        (13, "new_york"), (20, "new_york"), (23, "new_york"),
    ])
    def test_session_boundaries(self, hour, expected):
        assert spec.session_from_hour(hour) == expected

    def test_session_from_timestamp_matches_session_from_hour(self):
        from datetime import datetime, timezone

        ts = 1_700_000_000
        hour = datetime.fromtimestamp(ts, tz=timezone.utc).hour
        assert spec.session_from_timestamp(ts) == spec.session_from_hour(hour)


class TestIndicatorParity:
    """The recomputed indicators must equal what the live client computes."""

    def test_rsi_matches_the_live_client(self):
        candles = _make_candles()
        closes = [c["close"] for c in candles]

        # The exact arithmetic from mt5_client.get_indicators, inlined here so
        # the test does not merely compare the module against itself.
        gains, losses = [], []
        for i in range(1, 15):
            diff = closes[-i] - closes[-i - 1]
            (gains if diff > 0 else losses).append(abs(diff))
        avg_gain = sum(gains) / 14 if gains else 0.001
        avg_loss = sum(losses) / 14 if losses else 0.001
        expected = 100 - (100 / (1 + avg_gain / avg_loss))

        assert lpf.live_rsi(closes) == pytest.approx(expected)

    def test_macd_matches_the_live_client(self):
        closes = [c["close"] for c in _make_candles()]
        expected = sum(closes[-12:]) / 12 - sum(closes[-26:]) / 26
        assert lpf.live_macd(closes) == pytest.approx(expected)

    def test_atr_matches_the_live_client(self):
        candles = _make_candles()
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]

        trs = []
        for i in range(1, 15):
            trs.append(max(highs[-i] - lows[-i],
                           abs(highs[-i] - closes[-i - 1]),
                           abs(lows[-i] - closes[-i - 1])))
        assert lpf.live_atr(highs, lows, closes) == pytest.approx(sum(trs) / 14)

    def test_indicator_dict_has_the_live_keys(self):
        out = lpf.live_indicators(_make_candles())
        assert set(out.keys()) == {
            "rsi", "atr", "macd", "ma_trend", "volatility", "atr_ratio", "close",
        }

    def test_too_few_candles_returns_none_rather_than_a_fallback_row(self):
        """Live substitutes a static fallback table here; such a row must never
        become training data."""
        assert lpf.live_indicators(_make_candles(n=30)) is None

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_ma_trend_matches_live_classification(self, seed):
        from config import MA_TREND_FLAT_ATR_MULT

        candles = _make_candles(seed=seed)
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        atr = lpf.live_atr(highs, lows, closes)

        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50
        price = closes[-1]
        if abs(price - ma20) <= atr * MA_TREND_FLAT_ATR_MULT:
            expected = "sideways"
        elif price > ma20 > ma50:
            expected = "strong uptrend"
        elif price > ma20:
            expected = "uptrend"
        elif price < ma20 < ma50:
            expected = "strong downtrend"
        elif price < ma20:
            expected = "downtrend"
        else:
            expected = "sideways"
        assert lpf.live_ma_trend(closes, atr) == expected


class TestScoreBucketParity:
    """The discretised scores must match the live snapshot functions."""

    @pytest.mark.parametrize("ma_trend,rsi,expected", [
        ("strong uptrend", 50, (85, "bullish")),
        ("uptrend", 50, (70, "bullish")),
        ("strong downtrend", 50, (85, "bearish")),
        ("downtrend", 50, (70, "bearish")),
        ("sideways", 70, (75, "bullish")),
        ("sideways", 60, (65, "bullish")),
        ("sideways", 30, (75, "bearish")),
        ("sideways", 40, (65, "bearish")),
        ("sideways", 50, (40, "neutral")),
    ])
    def test_trend_score_matches_live(self, ma_trend, rsi, expected):
        from analysis.technical.indicators import get_trend_score_from_snapshot
        from data.market.market_snapshot import MarketSnapshot
        from config import TF_TREND

        data = {"ma_trend": ma_trend, "rsi": rsi, "atr": 0.001, "macd": 0.0, "close": 1.1}
        snap = MarketSnapshot(data={"X": {TF_TREND: data}})

        assert lpf.trend_score_from_indicators(data) == expected
        assert get_trend_score_from_snapshot(snap, "X") == expected

    @pytest.mark.parametrize("rsi,expected", [
        (25, (85, "bullish")), (75, (85, "bearish")),
        (40, (65, "bearish")), (60, (65, "bullish")),
        (50, (40, "neutral")),
    ])
    def test_momentum_score_matches_live(self, rsi, expected):
        from analysis.technical.indicators import get_momentum_score_from_snapshot
        from data.market.market_snapshot import MarketSnapshot
        from config import TF_DECISION

        data = {"ma_trend": "sideways", "rsi": rsi, "atr": 0.001, "macd": 0.0, "close": 1.1}
        snap = MarketSnapshot(data={"X": {TF_DECISION: data}})

        assert lpf.momentum_score_from_indicators(data) == expected
        assert get_momentum_score_from_snapshot(snap, "X", timeframe=TF_DECISION) == expected

    def test_regime_matches_live(self):
        from analysis.technical.regime import get_market_regime_from_snapshot
        from data.market.market_snapshot import MarketSnapshot
        from config import TF_TREND, TF_DECISION

        for ma_trend, volatility, expected in [
            ("strong uptrend", "normal", "TRENDING"),
            ("strong downtrend", "normal", "TRENDING"),
            ("sideways", "normal", "RANGING"),
            ("sideways", "very high", "HIGH_VOLATILITY"),
            ("sideways", "low", "LOW_VOLATILITY"),
        ]:
            data = {"ma_trend": ma_trend, "rsi": 50, "atr": 0.001, "macd": 0.0,
                    "volatility": volatility, "close": 1.1}
            snap = MarketSnapshot(data={"X": {TF_TREND: data, TF_DECISION: data}})

            _, direction = lpf.trend_score_from_indicators(data)
            mine = lpf.regime_from_scores(
                direction, lpf.volatility_score_from_indicators(data)
            )

            assert mine == expected, f"{ma_trend}/{volatility}: got {mine}"
            assert get_market_regime_from_snapshot(snap, "X") == expected


class TestPreviouslyConstantFeaturesAreNowDynamic:
    """All three were frozen in production (KNOWN_ISSUES #13). These prove they
    respond to real input now, and that the fix reaches both paths identically."""

    # --- trend_strength -------------------------------------------------
    @pytest.mark.parametrize("strength,expected", [
        ("weak", 25.0), ("moderate", 60.0), ("strong", 100.0),
    ])
    def test_trend_strength_encodes_each_level_distinctly(self, strength, expected):
        assert spec.encode_trend_strength(strength) == expected

    def test_trend_strength_levels_are_ordered_and_distinct(self):
        w = spec.encode_trend_strength("weak")
        m = spec.encode_trend_strength("moderate")
        s = spec.encode_trend_strength("strong")
        assert w < m < s, "strength levels must be ordered"
        assert len({w, m, s}) == 3

    @pytest.mark.parametrize("raw", ["STRONG", " Strong ", "Moderate"])
    def test_trend_strength_is_case_and_whitespace_tolerant(self, raw):
        assert spec.encode_trend_strength(raw) == spec.encode_trend_strength(
            raw.strip().lower()
        )

    def test_unknown_strength_is_distinct_from_weak(self):
        """"not measured" and "measured, weak" must not collide."""
        assert spec.encode_trend_strength("bogus") == spec.TREND_STRENGTH_DEFAULT
        assert spec.encode_trend_strength(None) == spec.TREND_STRENGTH_DEFAULT
        assert spec.TREND_STRENGTH_DEFAULT != spec.encode_trend_strength("weak")

    def test_numeric_strength_passes_through(self):
        """A future numeric source needs no change here."""
        assert spec.encode_trend_strength(42.5) == 42.5

    def test_the_real_analyser_output_encodes_to_a_real_number(self):
        """End to end against the actual MTF analyser, not a literal."""
        from analysis.multi_timeframe.analyzer_snapshot import (
            get_multi_timeframe_analysis_from_snapshot,
        )
        from data.market.market_snapshot import MarketSnapshot
        from config import TF_TREND, TF_DECISION, TF_TIMING

        # All three timeframes agreeing bullish -> "strong".
        bull = {"ma_trend": "strong uptrend", "rsi": 62, "atr": 0.001,
                "macd": 0.0, "volatility": "normal", "close": 1.1}
        snap = MarketSnapshot(data={"X": {TF_TREND: bull, TF_DECISION: bull, TF_TIMING: bull}})
        mtf = get_multi_timeframe_analysis_from_snapshot(snap, "X")

        assert isinstance(mtf.strength, str)
        assert spec.encode_trend_strength(mtf.strength) > 0.0, (
            "the real analyser output still encodes to the not-measured default"
        )

    def test_main_py_no_longer_discards_the_strength_string(self):
        """The isinstance guard that caused the bug must be gone."""
        import os

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo, "main.py"), encoding="utf-8-sig") as fh:
            source = fh.read()

        assert "mtf.strength if isinstance(mtf.strength, (int, float))" not in source, (
            "main.py still guards a string with isinstance((int, float)), which "
            "always falls through and feeds the model a constant"
        )

    # --- volatility_score -----------------------------------------------
    @pytest.mark.parametrize("volatility,expected", [
        ("very high", 20.0), ("high", 35.0), ("low", 80.0), ("normal", 55.0),
    ])
    def test_volatility_score_responds_to_the_bucket(self, volatility, expected):
        from analysis.technical.indicators import get_volatility_score_from_snapshot
        from data.market.market_snapshot import MarketSnapshot
        from config import TF_DECISION

        data = {"ma_trend": "uptrend", "rsi": 50, "atr": 0.001, "macd": 0.0,
                "volatility": volatility, "close": 1.1}
        snap = MarketSnapshot(data={"X": {TF_DECISION: data}})

        assert get_volatility_score_from_snapshot(snap, "X") == expected
        assert lpf.volatility_score_from_indicators(data) == expected

    def test_volatility_score_is_not_constant_across_regimes(self):
        """The actual defect: one value for every market condition."""
        seen = set()
        for ratio in (0.5, 1.0, 1.3, 2.0):
            bucket = lpf.live_volatility_bucket(ratio)
            seen.add(lpf.volatility_score_from_indicators({"volatility": bucket}))
        assert len(seen) == 4, f"volatility_score barely moves: {sorted(seen)}"

    def test_volatility_is_symbol_scale_free(self):
        """An absolute ATR% cut pinned EURUSD (~0.14% of price) and XAUUSD
        (~0.96%) to different permanent buckets. The ratio must not."""
        import random

        def series(price, vol_rel, n=100, seed=3):
            rng = random.Random(seed)
            closes = [price]
            for _ in range(n):
                closes.append(closes[-1] * (1 + rng.gauss(0, vol_rel)))
            closes = closes[1:]
            highs, lows = [], []
            rng2 = random.Random(seed + 1)
            for c in closes:
                r = abs(rng2.gauss(0, vol_rel * c * 0.8))
                highs.append(c + r)
                lows.append(c - r)
            return highs, lows, closes

        # Same *relative* volatility, wildly different price scales.
        buckets = set()
        for price in (1.10, 1.27, 4330.0):
            highs, lows, closes = series(price, 0.0015)
            atr = lpf.live_atr(highs, lows, closes)
            buckets.add(lpf.live_volatility_bucket(
                lpf.live_atr_ratio(highs, lows, closes, atr)))
        assert len(buckets) == 1, (
            f"identical relative volatility gave different buckets by price "
            f"scale: {buckets}"
        )

    def test_there_is_only_one_atr_ratio_implementation(self):
        """There used to be a copy in mt5_client and another in
        live_parity_features, kept in step by a test. They are now the same
        function, so drift is impossible rather than merely detected."""
        import inspect
        from data.market import mt5_client

        assert not hasattr(mt5_client, "_atr_ratio"), (
            "mt5_client grew its own _atr_ratio again"
        )
        source = inspect.getsource(mt5_client.get_indicators)
        assert "lpf.live_atr_ratio" in source

    def test_get_indicators_emits_the_volatility_key(self):
        """The key whose absence froze the score at 55."""
        out = lpf.live_indicators(_make_candles())
        assert "volatility" in out
        assert out["volatility"] in {"very high", "high", "normal", "low"}

    def test_fallback_indicator_rows_carry_the_key_too(self):
        """A fallback row missing the key would silently reintroduce the bug."""
        from data.market.mt5_client import FALLBACK_INDICATORS

        for symbol, row in FALLBACK_INDICATORS.items():
            assert "volatility" in row, f"{symbol} fallback lacks 'volatility'"

    # --- market_regime ---------------------------------------------------
    def test_sideways_is_reachable_with_a_real_band(self):
        """Was only reachable when price == ma20 exactly."""
        from config import MA_TREND_FLAT_ATR_MULT

        closes = [1.1000] * 60          # perfectly flat -> price == ma20
        atr = 0.0010
        assert lpf.live_ma_trend(closes, atr) == "sideways"

        # And just inside the band, where the old code said "uptrend".
        closes = [1.1000] * 59 + [1.1000 + atr * MA_TREND_FLAT_ATR_MULT * 0.5]
        assert lpf.live_ma_trend(closes, atr) == "sideways"

    def test_outside_the_band_still_trends(self):
        """The band must not swallow genuine trends."""
        from config import MA_TREND_FLAT_ATR_MULT

        atr = 0.0010
        closes = [1.1000] * 59 + [1.1000 + atr * MA_TREND_FLAT_ATR_MULT * 3]
        assert lpf.live_ma_trend(closes, atr) != "sideways"

    def test_regime_reaches_more_than_one_value(self):
        """The actual defect: always TRENDING."""
        from analysis.technical.regime import get_market_regime_from_snapshot
        from data.market.market_snapshot import MarketSnapshot
        from config import TF_TREND, TF_DECISION

        seen = set()
        cases = [
            ("strong uptrend", "normal"),
            ("sideways", "normal"),
            ("sideways", "very high"),
            ("sideways", "low"),
        ]
        for ma_trend, volatility in cases:
            data = {"ma_trend": ma_trend, "rsi": 50, "atr": 0.001, "macd": 0.0,
                    "volatility": volatility, "close": 1.1}
            snap = MarketSnapshot(data={"X": {TF_TREND: data, TF_DECISION: data}})
            live = get_market_regime_from_snapshot(snap, "X")

            _, direction = lpf.trend_score_from_indicators(data)
            mine = lpf.regime_from_scores(
                direction, lpf.volatility_score_from_indicators(data)
            )
            assert mine == live, f"training/live regime disagree for {ma_trend}/{volatility}"
            seen.add(live)

        assert len(seen) >= 3, f"regime is still nearly constant: {seen}"
        assert "RANGING" in seen, "RANGING is still unreachable"

    def test_spec_records_no_remaining_constants(self):
        assert spec.LIVE_CONSTANT_FEATURES == {}, (
            "a feature is frozen again; document it or fix it"
        )


class TestEndToEndVectorParity:
    """The whole point: same candles in, same ten numbers out."""

    @pytest.mark.parametrize("seed", [1, 7, 42])
    @pytest.mark.parametrize("direction", ["BUY", "SELL"])
    def test_training_vector_equals_live_vector(self, seed, direction):
        from analysis.models.entry_feature_spec import build_feature_vector

        h4 = _make_candles(seed=seed, n=200)
        h1 = _make_candles(seed=seed + 100, n=200)

        h4_ind = lpf.live_indicators(h4)
        h1_ind = lpf.live_indicators(h1)
        assert h4_ind is not None and h1_ind is not None

        trend_score, trend_dir = lpf.trend_score_from_indicators(h4_ind)
        momentum_score, mom_dir = lpf.momentum_score_from_indicators(h1_ind)
        vol = lpf.volatility_score_from_indicators(h1_ind)
        regime = lpf.regime_from_scores(trend_dir, vol)
        strength = lpf.mtf_strength_from_directions(trend_dir, mom_dir, mom_dir)
        bar_ts = h4[-1]["t"]

        # --- what training would build ---
        training_vec = build_feature_vector(
            rsi=h1_ind["rsi"], atr=h4_ind["atr"], macd=h1_ind["macd"],
            trend_strength=strength,
            trend_score=trend_score, momentum_score=momentum_score,
            volatility_score=vol, market_regime=regime,
            session=spec.session_from_timestamp(bar_ts), direction=direction,
        )

        # --- what live would build from the same snapshot ---
        from data.market.market_snapshot import MarketSnapshot
        from analysis.technical.indicators import (
            get_trend_score_from_snapshot, get_momentum_score_from_snapshot,
            get_volatility_score_from_snapshot,
        )
        from analysis.technical.regime import get_market_regime_from_snapshot
        from config import TF_TREND, TF_DECISION, TF_TIMING

        snap = MarketSnapshot(data={"X": {
            TF_TREND: h4_ind, TF_DECISION: h1_ind, TF_TIMING: h1_ind,
        }})
        live_trend_score, _ = get_trend_score_from_snapshot(snap, "X")
        live_momentum, _ = get_momentum_score_from_snapshot(snap, "X", timeframe=TF_DECISION)
        live_vol = get_volatility_score_from_snapshot(snap, "X")
        live_regime = get_market_regime_from_snapshot(snap, "X")

        from analysis.multi_timeframe.analyzer_snapshot import (
            get_multi_timeframe_analysis_from_snapshot,
        )
        live_mtf = get_multi_timeframe_analysis_from_snapshot(snap, "X")

        live_vec = build_feature_vector(
            rsi=h1_ind["rsi"], atr=h4_ind["atr"], macd=h1_ind["macd"],
            trend_strength=live_mtf.strength,
            trend_score=live_trend_score, momentum_score=live_momentum,
            volatility_score=live_vol, market_regime=live_regime,
            session=spec.session_from_timestamp(bar_ts), direction=direction,
        )

        assert training_vec == live_vec, (
            "training and live disagree:\n"
            f"  training: {spec.as_named_dict(training_vec)}\n"
            f"  live    : {spec.as_named_dict(live_vec)}"
        )

    def test_buy_and_sell_produce_different_vectors(self):
        """The defect in the deployed model: BUY and SELL scored identically.
        Whatever else is true, the vectors themselves must differ."""
        from analysis.models.entry_feature_spec import build_feature_vector

        common = dict(
            rsi=55.0, atr=0.0018, macd=0.0002, trend_strength=0.0,
            trend_score=70.0, momentum_score=65.0, volatility_score=55.0,
            market_regime="TRENDING", session="london",
        )
        buy = build_feature_vector(**common, direction="BUY")
        sell = build_feature_vector(**common, direction="SELL")

        assert buy != sell
        assert buy[9] == 1.0 and sell[9] == 0.0


class TestDegenerateRealWorldInput:
    """Shapes that real MT5 data contains and that used to crash the pipeline.

    A flat stretch of 14 bars — ordinary over weekends, holidays and thin
    sessions — raised ZeroDivisionError in `live_rsi`. The guard read
    ``if losses`` (is the list empty?) when it needed to ask whether the list
    *sums* to zero: ``diff > 0`` sends an exact 0.0 to `losses`, so fourteen
    zeros made a non-empty list with a zero average.

    In live this raised inside get_indicators' try/except and silently returned
    FALLBACK_INDICATORS — which is where the rsi=50.0 / atr=0.001 constants that
    polluted the recorded dataset came from (KNOWN_ISSUES #3). It surfaced as a
    hard crash only when the training pipeline called the same code without a
    catch-all around it.
    """

    FLAT = [1.1000] * 20

    def test_flat_series_does_not_raise(self):
        assert lpf.live_rsi(self.FLAT) == pytest.approx(50.0)

    def test_flat_series_scores_neutral_not_extreme(self):
        """50 is the meaningful answer for "price did not move"."""
        assert lpf.live_rsi(self.FLAT) == pytest.approx(50.0)

    def test_monotonic_series_still_reaches_the_extremes(self):
        """The epsilon must not flatten genuine one-sided moves."""
        up = lpf.live_rsi([1.0 + i * 0.01 for i in range(20)])
        down = lpf.live_rsi([1.2 - i * 0.01 for i in range(20)])
        assert up > 85, up
        assert down < 15, down
        assert up > down

    def test_normal_input_is_bit_identical_to_the_old_formula(self):
        """The fix must only affect inputs that previously raised."""
        import random

        rng = random.Random(1)
        closes = [1.1]
        for _ in range(40):
            closes.append(closes[-1] * (1 + rng.gauss(0, 0.001)))
        closes = closes[1:]

        gains, losses = [], []
        for i in range(1, 15):
            diff = closes[-i] - closes[-i - 1]
            (gains if diff > 0 else losses).append(abs(diff))
        avg_gain = sum(gains) / 14 if gains else 0.001
        avg_loss = sum(losses) / 14 if losses else 0.001
        old = 100 - (100 / (1 + avg_gain / avg_loss))

        assert lpf.live_rsi(closes) == pytest.approx(old, abs=1e-12)

    @pytest.mark.parametrize("name,closes", [
        ("fully flat", [1.1] * 120),
        ("flat then jump", [1.1] * 100 + [1.2] * 20),
        ("zero prices", [0.0] * 120),
        ("single tick move", [1.1] * 119 + [1.1001]),
        ("huge gap", [1.1] * 60 + [50.0] * 60),
        ("alternating", [1.1 + (0.01 if i % 2 else -0.01) for i in range(120)]),
    ])
    def test_full_feature_vector_survives_degenerate_input(self, name, closes):
        candles = [{"t": 1_600_000_000 + i * 14400, "open": c, "high": c,
                    "low": c, "close": c, "volume": 1.0}
                   for i, c in enumerate(closes)]

        ind = lpf.live_indicators(candles)
        assert ind is not None

        trend_score, trend_dir = lpf.trend_score_from_indicators(ind)
        momentum, mom_dir = lpf.momentum_score_from_indicators(ind)
        vol = lpf.volatility_score_from_indicators(ind)

        vec = spec.build_feature_vector(
            rsi=ind["rsi"], atr=ind["atr"], macd=ind["macd"],
            trend_strength=lpf.mtf_strength_from_directions(trend_dir, mom_dir, mom_dir),
            trend_score=trend_score, momentum_score=momentum,
            volatility_score=vol,
            market_regime=lpf.regime_from_scores(trend_dir, vol),
            session="london", direction="BUY",
        )
        assert len(vec) == spec.FEATURE_COUNT
        assert all(isinstance(v, float) for v in vec)
        import math
        assert all(math.isfinite(v) for v in vec), f"{name}: {vec}"

    def test_zero_atr_yields_no_trade_rather_than_a_bad_label(self):
        """A flat window gives ATR 0, so SL/TP distances collapse; such a bar
        must be dropped, not labelled."""
        import sys
        import os
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
        import train_entry_model as trainer

        candles = [{"t": 1_600_000_000 + i * 14400, "open": 1.1, "high": 1.1,
                    "low": 1.1, "close": 1.1, "volume": 1.0} for i in range(140)]
        assert trainer.simulate_trade(candles, 100, "BUY", 0.0, 20) is None


class TestSingleIndicatorImplementation:
    """The live client must not keep its own copy of the arithmetic."""

    def test_mt5_client_delegates_to_the_shared_module(self):
        import inspect
        from data.market.mt5_client import get_indicators

        source = inspect.getsource(get_indicators)
        assert "lpf.live_rsi" in source
        assert "lpf.live_atr" in source
        assert "lpf.live_macd" in source
        # And no second copy of the RSI arithmetic.
        assert "avg_gain" not in source, (
            "mt5_client has its own RSI again; two copies of the same formula "
            "is how training and serving drift apart"
        )
