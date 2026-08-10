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
        assert set(out.keys()) == {"rsi", "atr", "macd", "ma_trend", "close"}

    def test_too_few_candles_returns_none_rather_than_a_fallback_row(self):
        """Live substitutes a static fallback table here; such a row must never
        become training data."""
        assert lpf.live_indicators(_make_candles(n=30)) is None

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_ma_trend_matches_live_classification(self, seed):
        closes = [c["close"] for c in _make_candles(seed=seed)]
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50
        price = closes[-1]
        if price > ma20 > ma50:
            expected = "strong uptrend"
        elif price > ma20:
            expected = "uptrend"
        elif price < ma20 < ma50:
            expected = "strong downtrend"
        elif price < ma20:
            expected = "downtrend"
        else:
            expected = "sideways"
        assert lpf.live_ma_trend(closes) == expected


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

    def test_regime_matches_live_given_the_production_volatility_score(self):
        from analysis.technical.regime import get_market_regime_from_snapshot
        from data.market.market_snapshot import MarketSnapshot
        from config import TF_TREND, TF_DECISION

        for ma_trend, expected in [
            ("strong uptrend", "TRENDING"),
            ("strong downtrend", "TRENDING"),
            ("sideways", "RANGING"),
        ]:
            data = {"ma_trend": ma_trend, "rsi": 50, "atr": 0.001, "macd": 0.0, "close": 1.1}
            snap = MarketSnapshot(data={"X": {TF_TREND: data, TF_DECISION: data}})

            _, direction = lpf.trend_score_from_indicators(data)
            mine = lpf.regime_from_scores(direction, lpf.volatility_score_live())

            assert mine == expected
            assert get_market_regime_from_snapshot(snap, "X") == expected


class TestConstantFeatures:
    """Two of the ten features are frozen in production. Pinned so the fact is
    recorded, and so training keeps reproducing them as constants."""

    def test_volatility_score_is_always_55_in_live(self):
        from analysis.technical.indicators import get_volatility_score_from_snapshot
        from data.market.market_snapshot import MarketSnapshot
        from config import TF_DECISION

        # get_indicators never emits a "volatility" key on any path.
        for ma_trend in ("strong uptrend", "sideways", "strong downtrend"):
            data = {"ma_trend": ma_trend, "rsi": 50, "atr": 0.05, "macd": 0.0, "close": 1.1}
            snap = MarketSnapshot(data={"X": {TF_DECISION: data}})
            assert get_volatility_score_from_snapshot(snap, "X") == 55

        assert lpf.volatility_score_live() == 55.0

    def test_trend_strength_is_always_zero_in_live(self):
        """mtf.strength is a string, so main.py's isinstance check never passes."""
        from analysis.multi_timeframe.analyzer_snapshot import (
            get_multi_timeframe_analysis_from_snapshot,
        )
        from data.market.market_snapshot import MarketSnapshot
        from config import TF_TREND, TF_DECISION, TF_TIMING

        data = {"ma_trend": "strong uptrend", "rsi": 62, "atr": 0.001, "macd": 0.0, "close": 1.1}
        snap = MarketSnapshot(data={"X": {TF_TREND: data, TF_DECISION: data, TF_TIMING: data}})
        mtf = get_multi_timeframe_analysis_from_snapshot(snap, "X")

        assert isinstance(mtf.strength, str)
        fed_to_model = mtf.strength if isinstance(mtf.strength, (int, float)) else 0.0
        assert fed_to_model == 0.0
        assert lpf.trend_strength_live() == 0.0

    def test_market_regime_is_always_trending_in_live(self):
        """ma_trend can only be "sideways" when price == ma20 exactly, so the
        H4 trend direction is never "neutral" and the regime never leaves
        TRENDING. Confirmed against production logs."""
        from analysis.technical.regime import get_market_regime_from_snapshot
        from data.market.market_snapshot import MarketSnapshot
        from config import TF_TREND, TF_DECISION

        # These are the only ma_trend values get_indicators can emit.
        reachable = ["strong uptrend", "uptrend", "strong downtrend", "downtrend"]
        for ma_trend in reachable:
            data = {"ma_trend": ma_trend, "rsi": 50, "atr": 0.001, "macd": 0.0, "close": 1.1}
            snap = MarketSnapshot(data={"X": {TF_TREND: data, TF_DECISION: data}})
            assert get_market_regime_from_snapshot(snap, "X") == "TRENDING"

    def test_sideways_requires_exact_float_equality(self):
        """The one ma_trend value that would unlock a non-TRENDING regime."""
        import random

        rng = random.Random(3)
        hits = 0
        for _ in range(20000):
            closes = [1.1 + rng.gauss(0, 0.002) for _ in range(60)]
            if lpf.live_ma_trend(closes) == "sideways":
                hits += 1
        assert hits == 0, f"expected 'sideways' to be unreachable, saw {hits}"

    def test_the_spec_records_all_three_constants(self):
        assert spec.LIVE_CONSTANT_FEATURES == {
            "trend_strength": 0.0,
            "volatility_score": 55.0,
            "market_regime": 1.0,
        }

    def test_only_seven_features_carry_information(self):
        informative = set(spec.FEATURE_NAMES) - set(spec.LIVE_CONSTANT_FEATURES)
        assert informative == {
            "rsi", "atr", "macd", "trend_score", "momentum_score",
            "session", "direction",
        }
        assert len(informative) == 7


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
        momentum_score, _ = lpf.momentum_score_from_indicators(h1_ind)
        vol = lpf.volatility_score_live()
        regime = lpf.regime_from_scores(trend_dir, vol)
        bar_ts = h4[-1]["t"]

        # --- what training would build ---
        training_vec = build_feature_vector(
            rsi=h1_ind["rsi"], atr=h4_ind["atr"], macd=h1_ind["macd"],
            trend_strength=lpf.trend_strength_live(),
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

        live_vec = build_feature_vector(
            rsi=h1_ind["rsi"], atr=h4_ind["atr"], macd=h1_ind["macd"],
            trend_strength=0.0,
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
