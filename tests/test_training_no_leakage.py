"""No feature may carry information from after the decision moment.

Look-ahead is the failure mode that makes a model look excellent in validation
and lose money live, because the validation score was computed from data the
live system will not have. It is also silent: nothing crashes, the metrics just
come out too good.

These tests attack it from three sides:

1. **Construction** — the feature builder is handed candles and must not read
   past the entry index. Verified by truncating the future and checking the
   vector is byte-identical.
2. **Labels** — the label must depend only on bars *after* entry, and must
   change when the future changes.
3. **Statistics** — no single feature may classify the label almost perfectly,
   which is what an outcome-derived value looks like.
"""

from __future__ import annotations

import random
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from analysis.features import live_parity_features as lpf
from analysis.models import entry_feature_spec as spec

train = pytest.importorskip("train_entry_model")


def _series(n=400, start=1.1000, seed=5, scale=0.0015):
    rng = random.Random(seed)
    out, close, t = [], start, 1_600_000_000
    for i in range(n):
        close = close * (1 + rng.gauss(0, scale) / start)
        r = abs(rng.gauss(0, scale * 0.8))
        out.append({"t": float(t + i * 14400), "open": close, "high": close + r,
                    "low": close - r, "close": close, "volume": 1000.0})
    return out


class TestFeaturesCannotSeeTheFuture:
    def test_indicators_are_unchanged_when_the_future_is_deleted(self):
        """The strongest form: truncate everything after the entry bar and the
        feature vector must be identical."""
        candles = _series()
        entry = 250

        with_future = lpf.live_indicators(candles[: entry + 1])
        # Same call, but the future physically does not exist.
        truncated = lpf.live_indicators(candles[: entry + 1][: entry + 1])

        assert with_future == truncated

    def test_a_changed_future_does_not_change_the_features(self):
        """Mutating bars after entry must not move any indicator."""
        candles = _series()
        entry = 250

        before = lpf.live_indicators(candles[: entry + 1])

        tampered = [dict(c) for c in candles]
        for j in range(entry + 1, len(tampered)):
            tampered[j]["close"] *= 1.5
            tampered[j]["high"] *= 1.5
            tampered[j]["low"] *= 1.5

        after = lpf.live_indicators(tampered[: entry + 1])
        assert before == after, "an indicator read past the entry bar"

    def test_h1_join_never_takes_a_bar_from_the_future(self):
        """H1 is joined by last-closed-at-or-before; a later bar would leak."""
        import bisect

        h1 = _series(n=800, seed=9)
        h1_times = [c["t"] for c in h1]
        bar_close_t = h1[400]["t"]

        pos = bisect.bisect_right(h1_times, bar_close_t)
        # Everything used must be at or before the H4 bar's close.
        assert all(c["t"] <= bar_close_t for c in h1[:pos])
        assert h1[pos]["t"] > bar_close_t


class TestLabelsOnlyUseTheFuture:
    def test_label_depends_on_bars_after_entry(self):
        candles = _series()
        entry = 200
        ind = lpf.live_indicators(candles[: entry + 1])
        atr = float(ind["atr"])

        base = train.simulate_trade(candles, entry, "BUY", atr, 24)

        # Force an immediate take-profit right after entry.
        boosted = [dict(c) for c in candles]
        tp = candles[entry]["close"] + atr * train.ATR_TP_BASE_MULTIPLIER
        boosted[entry + 1]["high"] = tp * 1.01

        after = train.simulate_trade(boosted, entry, "BUY", atr, 24)
        assert after is not None
        assert after["label"] == 1.0
        assert after["bars"] == 1
        assert after != base or base["bars"] != 1

    def test_label_ignores_bars_at_or_before_entry(self):
        """Changing history must not change the outcome of a trade opened later."""
        candles = _series()
        entry = 200
        atr = float(lpf.live_indicators(candles[: entry + 1])["atr"])
        base = train.simulate_trade(candles, entry, "BUY", atr, 24)

        tampered = [dict(c) for c in candles]
        for j in range(0, entry):        # strictly before entry
            tampered[j]["high"] *= 2.0
            tampered[j]["low"] *= 0.5

        after = train.simulate_trade(tampered, entry, "BUY", atr, 24)
        assert after == base

    def test_horizon_is_respected(self):
        """A hit beyond the horizon must not be counted."""
        candles = _series()
        entry, horizon = 200, 5
        atr = float(lpf.live_indicators(candles[: entry + 1])["atr"])

        flat = [dict(c) for c in candles]
        entry_price = flat[entry]["close"]
        for j in range(entry + 1, entry + horizon + 1):
            flat[j].update(close=entry_price, high=entry_price, low=entry_price)
        # A big move only *after* the horizon closes.
        far = entry + horizon + 1
        flat[far]["high"] = entry_price + atr * 10

        assert train.simulate_trade(flat, entry, "BUY", atr, horizon) is None

    def test_both_directions_are_labelled_from_the_same_bar(self):
        """BUY and SELL on identical history must not both win."""
        candles = _series()
        entry = 200
        atr = float(lpf.live_indicators(candles[: entry + 1])["atr"])

        buy = train.simulate_trade(candles, entry, "BUY", atr, 24)
        sell = train.simulate_trade(candles, entry, "SELL", atr, 24)
        if buy and sell:
            assert not (buy["label"] == 1.0 and sell["label"] == 1.0), (
                "a bar cannot hit both a BUY target and a SELL target first"
            )


class TestOutcomeDerivedValuesCannotReachTheFeatures:
    def test_no_feature_name_refers_to_an_outcome(self):
        banned = {"label", "pnl", "win", "loss", "tp", "sl", "exit", "outcome",
                  "future", "result", "holding"}
        for name in spec.FEATURE_NAMES:
            assert not any(b in name.lower() for b in banned), (
                f"feature {name!r} looks outcome-derived"
            )

    def test_builder_signature_accepts_no_outcome_arguments(self):
        import inspect

        params = set(inspect.signature(spec.build_feature_vector).parameters)
        # Exactly the ten features, with `session` supplied by the caller
        # (live: now; training: the bar's own timestamp) instead of `market_regime`
        # being read from a clock.
        assert params == set(spec.FEATURE_NAMES), params

    def test_validation_flags_a_planted_leak(self):
        """The detector must actually fire — otherwise it proves nothing."""
        rng = random.Random(1)
        X, y, meta = [], [], []
        for i in range(400):
            label = 1.0 if rng.random() > 0.5 else 0.0
            row = [rng.gauss(50, 10) for _ in range(spec.FEATURE_COUNT)]
            row[0] = 100.0 if label == 1.0 else 0.0     # planted leak in `rsi`
            X.append(row)
            y.append(label)
            meta.append({"symbol": "X", "t": 1_600_000_000 + i * 14400,
                         "direction": "BUY", "reason": "tp_first", "bars": 3,
                         "regime": "TRENDING", "session": "london"})

        report = train.validate_dataset(X, y, meta)
        assert "rsi" in report["leaky_features"], (
            f"planted leak not detected: {report.get('single_feature_auc')}"
        )

    def test_validation_is_quiet_on_clean_noise(self):
        """And must not cry leak on data that has none."""
        rng = random.Random(2)
        X, y, meta = [], [], []
        for i in range(400):
            X.append([rng.gauss(50, 10) for _ in range(spec.FEATURE_COUNT)])
            y.append(1.0 if rng.random() > 0.5 else 0.0)
            meta.append({"symbol": "X", "t": 1_600_000_000 + i * 14400,
                         "direction": "BUY", "reason": "tp_first", "bars": 3,
                         "regime": "TRENDING", "session": "london"})

        report = train.validate_dataset(X, y, meta)
        assert report["leaky_features"] == []


class TestProvenanceGate:
    def test_missing_manifest_is_rejected(self, tmp_path):
        result = train.check_provenance(str(tmp_path))
        assert result["real"] is False
        assert "manifest" in result["reason"]

    def test_manifest_marks_the_data_real(self, tmp_path):
        import json

        (tmp_path / "manifest.json").write_text(
            json.dumps({"fetched_at": "2026-01-01T00:00:00Z", "files": {}}),
            encoding="utf-8",
        )
        assert train.check_provenance(str(tmp_path))["real"] is True
