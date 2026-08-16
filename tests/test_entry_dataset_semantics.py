"""What a training row means, pinned against the four defects that invalidated
the previous dataset.

Each class here corresponds to one finding in ENTRY_PIPELINE_AUDIT.md:

  entry price   — resolved to h4_ema_50 in 100% of rows, displacing the
                  barriers by 0.92 ATR against barriers of 1.0/1.5 ATR
  timeframe     — "H4" indicators computed over a repeated H1 grid
  look-ahead    — the attached H4 candle closed three hours after the decision
  direction     — no direction column, so every row was a BUY

A test that only checks the fix is half a test, so most classes also assert
that the check can fail — otherwise a later refactor can make them vacuous
without anyone noticing.
"""

from __future__ import annotations

import random
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from analysis.features import timeframe_alignment as ta  # noqa: E402
from analysis.features import live_parity_features as lpf  # noqa: E402

trainer = pytest.importorskip("train_entry_model")


def candles(timeframe, n, seed, price=1.10, vol=0.0012):
    """A realistic series: every bar opens with a small gap from the last close.

    The gap matters. Generating `open == previous close` makes "next bar's
    open" and "this bar's close" the same number, and a test that cannot tell
    them apart cannot catch the entry-price defect it exists to catch.
    """
    span = ta.duration(timeframe)
    rng = random.Random(seed)
    out, close = [], price
    for i in range(n):
        opened = close * (1 + rng.gauss(0, vol * 0.3))
        close = opened * (1 + rng.gauss(0, vol))
        high = max(opened, close) + abs(rng.gauss(0, vol * close))
        low = min(opened, close) - abs(rng.gauss(0, vol * close))
        out.append({"t": float(i * span), "open": opened, "high": high,
                    "low": low, "close": close, "volume": 1.0})
    return out


@pytest.fixture(scope="module")
def built():
    h4 = candles("H4", 900, seed=1)
    h1 = candles("H1", 3600, seed=2)
    X, y, meta = trainer.build_dataset({"EURUSD": {"H4": h4, "H1": h1}}, horizon=24)
    return h4, h1, X, y, meta


class TestEntryPriceIsExecutable:
    def test_entry_price_is_the_next_bars_open(self, built):
        """Not this bar's close, and emphatically not an EMA.

        The close is what told us to trade; it is history by the time we know
        it. The next bar's open is the first price a market order could have
        received.
        """
        h4, _, _, _, meta = built
        span = ta.duration("H4")
        for row in meta[:500]:
            decision_index = int(row["t"] / span) - 1
            assert row["entry_price"] == pytest.approx(
                h4[decision_index + 1]["open"], rel=0, abs=0)

    def test_entry_price_is_never_the_decision_bars_close(self, built):
        h4, _, _, _, meta = built
        span = ta.duration("H4")
        matches = sum(
            1 for row in meta[:500]
            if row["entry_price"] == h4[int(row["t"] / span) - 1]["close"])
        assert matches == 0, (
            f"{matches} rows priced the fill at the decision bar's close")

    def test_entry_price_never_resolves_to_a_moving_average(self, built):
        """The entry_v2 defect: a guard that was always false sent entry_price
        to h4_ema_50 in 100.0000% of rows."""
        h4, _, _, _, meta = built
        span = ta.duration("H4")
        for row in meta[:200]:
            decision_index = int(row["t"] / span) - 1
            closes = [c["close"] for c in h4[: decision_index + 1]]
            for period in (12, 26, 50, 200):
                if len(closes) >= period:
                    sma = sum(closes[-period:]) / period
                    assert row["entry_price"] != pytest.approx(sma, rel=1e-9)

    def test_a_decision_on_the_last_bar_yields_no_row(self):
        """There is no next bar to fill against, so the row must not exist."""
        h4 = candles("H4", 200, seed=5)
        assert trainer.simulate_trade(h4, len(h4) - 1, "BUY", 0.001, 24) is None


class TestDecisionTimeIsACloseTime:
    def test_the_timestamp_is_the_bars_close_not_its_open(self, built):
        h4, _, _, _, meta = built
        span = ta.duration("H4")
        opens = {c["t"] for c in h4}
        stamps = {row["t"] for row in meta}
        assert not (stamps & opens) or min(stamps) >= min(opens) + span
        for row in meta[:100]:
            assert row["t"] % span == 0
            assert row["t"] - span in opens

    def test_the_session_is_taken_at_the_decision_moment(self, built):
        """A four-hour bar can open in one session and close in another."""
        from analysis.models import entry_feature_spec as spec

        _, _, _, _, meta = built
        for row in meta[:200]:
            assert row["session"] == spec.session_from_timestamp(row["t"])


class TestNoLookAhead:
    def test_mutating_the_future_leaves_every_feature_unchanged(self):
        h4 = candles("H4", 900, seed=1)
        h1 = candles("H1", 3600, seed=2)
        cut = 600

        tampered = [dict(c) for c in h4]
        for j in range(cut, len(tampered)):
            for key in ("open", "high", "low", "close"):
                tampered[j][key] *= 7.0

        Xb, _, mb = trainer.build_dataset({"S": {"H4": h4, "H1": h1}}, horizon=24)
        Xm, _, mm = trainer.build_dataset({"S": {"H4": tampered, "H1": h1}}, horizon=24)

        boundary = ta.decision_time(h4, cut - 1, "H4")
        before = {(m["t"], m["direction"]): tuple(x)
                  for x, m in zip(Xb, mb) if m["t"] <= boundary}
        after = {(m["t"], m["direction"]): tuple(x)
                 for x, m in zip(Xm, mm) if m["t"] <= boundary}

        common = set(before) & set(after)
        assert len(common) > 100, "too few overlapping rows to prove anything"
        differing = [k for k in common if before[k] != after[k]]
        assert not differing, f"{len(differing)} feature vectors read the future"

    def test_mutating_the_past_does_change_features(self):
        """Non-vacuity: the comparison above must be capable of failing."""
        h4 = candles("H4", 900, seed=1)
        h1 = candles("H1", 3600, seed=2)
        cut = 600

        tampered = [dict(c) for c in h4]
        for j in range(cut):
            for key in ("open", "high", "low", "close"):
                tampered[j][key] *= 7.0

        Xb, _, mb = trainer.build_dataset({"S": {"H4": h4, "H1": h1}}, horizon=24)
        Xm, _, mm = trainer.build_dataset({"S": {"H4": tampered, "H1": h1}}, horizon=24)

        boundary = ta.decision_time(h4, cut - 1, "H4")
        before = {(m["t"], m["direction"]): tuple(x)
                  for x, m in zip(Xb, mb) if m["t"] <= boundary}
        after = {(m["t"], m["direction"]): tuple(x)
                 for x, m in zip(Xm, mm) if m["t"] <= boundary}
        common = set(before) & set(after)
        assert any(before[k] != after[k] for k in common)


class TestHigherTimeframeIndicatorsUseHigherTimeframeCandles:
    """The entry_v2 defect measured directly.

    Its "h4_rsi_14" matched an RSI over the forward-filled H1 grid to 0.0000
    and a true H4 RSI not at all (mean error ~12.5 points). The lookback was 14
    hours wearing the name of 14 H4 candles.
    """

    def test_the_h4_indicator_matches_a_true_h4_computation(self):
        h4 = candles("H4", 400, seed=11)
        at = ta.decision_time(h4, 300, "H4")

        from_pipeline = lpf.live_indicators(ta.closed_slice(h4, "H4", at))
        from_real_h4 = lpf.live_indicators(h4[:301])
        assert from_pipeline is not None
        assert from_pipeline["rsi"] == pytest.approx(from_real_h4["rsi"], rel=1e-12)
        assert from_pipeline["atr"] == pytest.approx(from_real_h4["atr"], rel=1e-12)

    def test_an_oversampled_grid_would_give_a_different_answer(self):
        """Non-vacuity: repeating each H4 candle four times must change the result,
        otherwise the test above proves nothing."""
        h4 = candles("H4", 400, seed=11)
        oversampled = [dict(c) for c in h4[:301] for _ in range(4)]

        true_rsi = lpf.live_indicators(h4[:301])["rsi"]
        fake_rsi = lpf.live_indicators(oversampled)["rsi"]
        assert abs(true_rsi - fake_rsi) > 1.0, (
            "oversampling did not change the indicator; this fixture no longer "
            "reproduces the original defect")

    def test_h1_and_h4_indicators_are_computed_from_separate_series(self, built):
        """They must not be two views of one forward-filled grid."""
        h4, h1, _, _, _ = built
        at = ta.decision_time(h4, 400, "H4")
        h4_ind = lpf.live_indicators(ta.closed_slice(h4, "H4", at))
        h1_ind = lpf.live_indicators(ta.closed_slice(h1, "H1", at))
        assert h4_ind["atr"] != h1_ind["atr"]


class TestDirectionIsRepresented:
    def test_both_directions_are_labelled(self, built):
        _, _, _, _, meta = built
        directions = {row["direction"] for row in meta}
        assert directions == {"BUY", "SELL"}

    def test_the_two_directions_are_roughly_balanced(self, built):
        _, _, _, _, meta = built
        buys = sum(1 for m in meta if m["direction"] == "BUY")
        sells = len(meta) - buys
        assert 0.4 < buys / (buys + sells) < 0.6, (buys, sells)

    def test_direction_reaches_the_feature_vector(self, built):
        """Not merely recorded in metadata — the model must be able to see it."""
        from analysis.models import entry_feature_spec as spec

        _, _, X, _, meta = built
        index = spec.FEATURE_NAMES.index("direction")
        by_direction = {}
        for row, info in zip(X, meta):
            by_direction.setdefault(info["direction"], set()).add(row[index])
        assert by_direction["BUY"] != by_direction["SELL"]

    def test_buy_and_sell_labels_are_not_forced_opposites(self, built):
        """Both can lose: TP and SL are asymmetric, so a bar can miss both."""
        _, _, _, y, meta = built
        paired = {}
        for label, info in zip(y, meta):
            paired.setdefault((info["symbol"], info["t"]), {})[info["direction"]] = label
        both = [v for v in paired.values() if len(v) == 2]
        assert both, "no timestamp produced both directions"
        same = sum(1 for v in both if v["BUY"] == v["SELL"])
        assert same > 0, (
            "every paired BUY/SELL had opposite labels, which would mean the "
            "label is a direction predictor rather than a trade-quality one")


class TestLabelSemantics:
    def test_unresolved_trades_are_dropped_not_guessed(self):
        """entry_v2 invented a label for 9% of rows by proximity to the close."""
        flat = [{"t": float(i * 14400), "open": 1.0, "high": 1.0,
                 "low": 1.0, "close": 1.0, "volume": 1.0} for i in range(200)]
        assert trainer.simulate_trade(flat, 50, "BUY", 0.01, 24) is None

    def test_a_touched_tp_is_a_win(self):
        bars = [{"t": float(i * 14400), "open": 1.0, "high": 1.0,
                 "low": 1.0, "close": 1.0, "volume": 1.0} for i in range(60)]
        bars[12]["high"] = 1.05
        out = trainer.simulate_trade(bars, 10, "BUY", 0.01, 24)
        assert out["label"] == 1.0 and out["reason"] == "tp_first"

    def test_a_touched_sl_is_a_loss(self):
        bars = [{"t": float(i * 14400), "open": 1.0, "high": 1.0,
                 "low": 1.0, "close": 1.0, "volume": 1.0} for i in range(60)]
        bars[12]["low"] = 0.95
        out = trainer.simulate_trade(bars, 10, "BUY", 0.01, 24)
        assert out["label"] == 0.0 and out["reason"] == "sl_first"

    def test_both_barriers_in_one_bar_counts_as_a_loss(self):
        """Without tick data the order is unknowable; assume the worse one."""
        bars = [{"t": float(i * 14400), "open": 1.0, "high": 1.0,
                 "low": 1.0, "close": 1.0, "volume": 1.0} for i in range(60)]
        bars[12]["high"], bars[12]["low"] = 1.05, 0.95
        out = trainer.simulate_trade(bars, 10, "BUY", 0.01, 24)
        assert out["label"] == 0.0 and out["reason"] == "both_same_bar"

    def test_the_horizon_is_respected_in_bars(self):
        bars = [{"t": float(i * 14400), "open": 1.0, "high": 1.0,
                 "low": 1.0, "close": 1.0, "volume": 1.0} for i in range(80)]
        bars[40]["high"] = 1.05           # 29 bars after entry at index 11
        assert trainer.simulate_trade(bars, 10, "BUY", 0.01, 24) is None
        assert trainer.simulate_trade(bars, 10, "BUY", 0.01, 40)["label"] == 1.0

    def test_holding_time_never_exceeds_the_horizon(self, built):
        _, _, _, _, meta = built
        assert max(row["bars"] for row in meta) <= 24
        assert min(row["bars"] for row in meta) >= 1
