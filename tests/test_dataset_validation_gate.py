"""The checks that must stop a training run before a tree is grown.

A validation gate that reports problems but does not block is decoration. These
tests pin both halves: that each critical check detects its defect, and that a
clean dataset still passes — a gate which rejects everything is as useless as
one which rejects nothing.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from analysis.features import timeframe_alignment as ta  # noqa: E402
from analysis.models import entry_feature_spec as spec  # noqa: E402

trainer = pytest.importorskip("train_entry_model")


def candles(timeframe, n, seed, price=1.10, vol=0.0012):
    span = ta.duration(timeframe)
    rng = random.Random(seed)
    out, close = [], price
    for i in range(n):
        opened = close * (1 + rng.gauss(0, vol * 0.3))
        close = opened * (1 + rng.gauss(0, vol))
        out.append({"t": float(i * span), "open": opened,
                    "high": max(opened, close) + abs(rng.gauss(0, vol * close)),
                    "low": min(opened, close) - abs(rng.gauss(0, vol * close)),
                    "close": close, "volume": 1.0})
    return out


@pytest.fixture(scope="module")
def built():
    h4 = candles("H4", 900, seed=3)
    h1 = candles("H1", 3600, seed=4)
    return trainer.build_dataset({"EURUSD": {"H4": h4, "H1": h1}}, horizon=24)


class TestCandleSeriesGate:
    def test_a_clean_series_is_accepted(self):
        report = trainer.validate_candles(
            {"EURUSD": {"H4": candles("H4", 400, 1), "H1": candles("H1", 1600, 2)}})
        assert report["blockers"] == []

    def test_duplicate_timestamps_block(self):
        series = candles("H4", 400, 1)
        series.insert(10, dict(series[10]))
        report = trainer.validate_candles({"EURUSD": {"H4": series}})
        assert any("duplicate_timestamps" in b for b in report["blockers"])

    def test_unsorted_candles_block(self):
        series = candles("H4", 400, 1)
        series[5], series[9] = series[9], series[5]
        report = trainer.validate_candles({"EURUSD": {"H4": series}})
        assert any("unsorted" in b for b in report["blockers"])

    def test_an_off_grid_series_blocks(self):
        """A broker timezone offset shifts every alignment by a constant."""
        series = [{**c, "t": c["t"] + 137.0} for c in candles("H4", 400, 1)]
        report = trainer.validate_candles({"EURUSD": {"H4": series}})
        assert any("grid" in b for b in report["blockers"])

    def test_impossible_ohlc_blocks(self):
        series = candles("H4", 400, 1)
        series[7]["high"] = series[7]["low"] - 1.0
        report = trainer.validate_candles({"EURUSD": {"H4": series}})
        assert any("bad_ohlc" in b for b in report["blockers"])


class TestDatasetGate:
    def test_a_clean_dataset_passes_every_critical_check(self, built):
        X, y, meta = built
        report = trainer.validate_dataset(X, y, meta)
        assert report.get("fatal") is None
        assert report["non_finite_values"] == 0
        assert report["wrong_width_rows"] == 0
        assert report["out_of_order_rows"] == 0
        assert report["duplicate_decisions"] == 0
        assert report["impossible_entry_prices"] == 0
        assert report["leaky_features"] == []
        assert report["overall"]["balance_ok"] is True

    def test_an_empty_dataset_is_fatal(self):
        assert trainer.validate_dataset([], [], []).get("fatal")

    def test_out_of_order_rows_are_detected(self, built):
        X, y, meta = built
        shuffled = list(meta)
        shuffled[10], shuffled[400] = shuffled[400], shuffled[10]
        report = trainer.validate_dataset(X, y, shuffled)
        assert report["out_of_order_rows"] > 0

    def test_duplicate_decisions_are_detected(self, built):
        X, y, meta = built
        report = trainer.validate_dataset(
            X + X[:5], y + y[:5], meta + meta[:5])
        assert report["duplicate_decisions"] == 5

    def test_an_impossible_entry_price_is_detected(self, built):
        X, y, meta = built
        damaged = [dict(m) for m in meta]
        damaged[3]["entry_price"] = -1.0
        damaged[9]["entry_price"] = float("nan")
        assert trainer.validate_dataset(X, y, damaged)["impossible_entry_prices"] == 2

    def test_a_planted_leak_is_caught(self, built):
        """A feature copied from the label must trip the single-feature probe."""
        X, y, meta = built
        leaked = [list(row) for row in X]
        for row, label in zip(leaked, y):
            row[0] = label * 10.0 + row[0] * 1e-9
        report = trainer.validate_dataset(leaked, y, meta)
        assert spec.FEATURE_NAMES[0] in report["leaky_features"]


class TestTheLeakageProbeHandlesTies:
    """The probe used ordinal ranks, which manufactures a leak from a tie.

    `sorted(zip(col, y))` breaks ties on the second element, so inside every
    group of equal feature values the y=1 rows sort last and take the highest
    ranks. On an encoded categorical that produced ~0.90 for columns whose true
    AUC is ~0.50 — and every encoded feature in the contract is such a column,
    so promoting the probe to a blocker would have rejected every honest
    dataset while catching nothing.
    """

    def test_a_low_cardinality_feature_is_not_reported_as_leaky(self, built):
        X, y, meta = built
        report = trainer.validate_dataset(X, y, meta)
        for name in ("market_regime", "session", "direction", "volatility_score"):
            auc = report["single_feature_auc"].get(name)
            if auc is not None:
                assert auc < 0.75, (
                    f"{name} scored {auc}; a two-to-four-valued encoding cannot "
                    f"separate the label that well — this is the tie bug")

    def test_a_constant_predictor_scores_one_half(self):
        import numpy as np

        y = np.asarray([1.0] * 50 + [0.0] * 50)
        constant = np.zeros(100)
        assert trainer._auc_with_ties(y, constant, 50, 50) == pytest.approx(0.5)

    def test_a_perfect_predictor_still_scores_one(self):
        import numpy as np

        y = np.asarray([1.0] * 50 + [0.0] * 50)
        perfect = np.concatenate([np.ones(50), np.zeros(50)])
        assert trainer._auc_with_ties(y, perfect, 50, 50) == pytest.approx(1.0)

    def test_a_half_tied_column_is_scored_between_the_extremes(self):
        """Sanity that ties are averaged rather than ordered by label."""
        import numpy as np

        y = np.asarray([1.0] * 50 + [0.0] * 50)
        # Half the positives share a value with half the negatives.
        col = np.concatenate([np.ones(25), np.zeros(25), np.zeros(50)])
        auc = trainer._auc_with_ties(y, col, 50, 50)
        assert 0.5 < auc < 1.0
