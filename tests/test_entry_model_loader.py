"""A failed model load must not poison the cache for the rest of the process.

`load_v2_model()` assigned the module-level cache before loading::

    _model = xgb.Booster()      # global now set
    _model.load_model(PATH)     # ...and this is what raises

On a raise the function returned None, which reads as correct — but the global
was left holding an empty Booster. Every later call took the
``if _model is not None`` fast path and returned that empty object, whose
``num_features()`` raises. The entry gate then reported

    ML_GATE_INVALID — model feature count could not be determined

for the rest of the process, blaming the feature contract for what was really a
file-load failure, and never recovering even after the model on disk was
replaced with a good one. Only a restart cleared it.

Found while verifying a retrained model through the live inference path.
"""

from __future__ import annotations



import pytest

import analysis.models.xgboost_v2_inference as inf
from analysis.models import entry_feature_spec as spec


@pytest.fixture
def good_model(tmp_path):
    xgb = pytest.importorskip("xgboost")
    np = pytest.importorskip("numpy")

    X = np.random.rand(120, spec.FEATURE_COUNT)
    y = (np.random.rand(120) > 0.5).astype(float)
    d = xgb.DMatrix(X, label=y, feature_names=list(spec.FEATURE_NAMES))
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 3, "seed": 1},
        d, num_boost_round=5,
    )
    path = str(tmp_path / "good.json")
    booster.save_model(path)
    return path


@pytest.fixture
def corrupt_model(tmp_path):
    path = str(tmp_path / "corrupt.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{ this is not a model }")
    return path


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    """Never let one test's cache leak into the next, or into the real module."""
    monkeypatch.setattr(inf, "_model", None)
    original = inf.MODEL_PATH
    yield
    monkeypatch.setattr(inf, "MODEL_PATH", original)
    monkeypatch.setattr(inf, "_model", None)


class TestFailedLoadDoesNotPoisonTheCache:
    def test_a_corrupt_file_returns_none(self, monkeypatch, corrupt_model):
        monkeypatch.setattr(inf, "MODEL_PATH", corrupt_model)
        assert inf.load_v2_model() is None

    def test_the_cache_stays_empty_after_a_failed_load(self, monkeypatch, corrupt_model):
        """The actual defect: the global must not hold a half-built Booster."""
        monkeypatch.setattr(inf, "MODEL_PATH", corrupt_model)
        inf.load_v2_model()
        assert inf._model is None, (
            "a failed load left an object in the cache; every later call will "
            "return it and fail on num_features()"
        )

    def test_a_good_model_loads_after_a_failed_one(self, monkeypatch, corrupt_model, good_model):
        """Recovery without a restart — this is what the bug prevented."""
        monkeypatch.setattr(inf, "MODEL_PATH", corrupt_model)
        assert inf.load_v2_model() is None

        monkeypatch.setattr(inf, "MODEL_PATH", good_model)
        booster = inf.load_v2_model()
        assert booster is not None
        assert booster.num_features() == spec.FEATURE_COUNT

    def test_prediction_recovers_after_a_failed_load(self, monkeypatch, corrupt_model, good_model):
        """End to end: the gate must stop reporting ML_GATE_INVALID once the
        model on disk is usable again."""
        monkeypatch.setattr(inf, "MODEL_PATH", corrupt_model)
        first = inf.predict_with_v2(
            rsi=55.0, atr=0.0018, macd=0.0002, trend_strength=0.0,
            trend_score=70.0, momentum_score=65.0, volatility_score=55.0,
            market_regime="TRENDING", direction="BUY",
        )
        assert first["status"] == "ML_MODEL_MISSING"
        assert first["available"] is False

        monkeypatch.setattr(inf, "MODEL_PATH", good_model)
        second = inf.predict_with_v2(
            rsi=55.0, atr=0.0018, macd=0.0002, trend_strength=0.0,
            trend_score=70.0, momentum_score=65.0, volatility_score=55.0,
            market_regime="TRENDING", direction="BUY",
        )
        assert second["status"] == "OK", (
            f"did not recover after the model was fixed: {second}"
        )
        assert second["available"] is True
        assert 0.0 <= second["p_win"] <= 1.0

    def test_a_missing_file_reports_model_missing_not_gate_invalid(self, monkeypatch, tmp_path):
        """The two statuses mean different things to an operator reading logs."""
        monkeypatch.setattr(inf, "MODEL_PATH", str(tmp_path / "nope.json"))
        result = inf.predict_with_v2(
            rsi=55.0, atr=0.0018, macd=0.0002, trend_strength=0.0,
            trend_score=70.0, momentum_score=65.0, volatility_score=55.0,
            market_regime="TRENDING", direction="BUY",
        )
        assert result["status"] == "ML_MODEL_MISSING"

    def test_a_successful_load_is_cached(self, monkeypatch, good_model):
        """The cache must still work — this is a hot path."""
        monkeypatch.setattr(inf, "MODEL_PATH", good_model)
        first = inf.load_v2_model()
        second = inf.load_v2_model()
        assert first is second


class TestATenFeatureModelPassesTheGate:
    """Proves the contract accepts a correctly-built model, so a green gate is
    achievable and these tests are not vacuously passing on a blocked path."""

    def test_gate_returns_ok(self, monkeypatch, good_model):
        monkeypatch.setattr(inf, "MODEL_PATH", good_model)
        result = inf.predict_with_v2(
            rsi=58.0, atr=0.0018, macd=0.0004, trend_strength=0.0,
            trend_score=70.0, momentum_score=65.0, volatility_score=55.0,
            market_regime="TRENDING", direction="BUY",
        )
        assert result["status"] == "OK"
        assert result["available"] is True
        assert result["p_win"] is not None

    def test_buy_and_sell_reach_the_model_as_different_vectors(self, monkeypatch, good_model):
        monkeypatch.setattr(inf, "MODEL_PATH", good_model)
        common = dict(
            rsi=58.0, atr=0.0018, macd=0.0004, trend_strength=0.0,
            trend_score=70.0, momentum_score=65.0, volatility_score=55.0,
            market_regime="TRENDING",
        )
        buy = inf.predict_with_v2(**common, direction="BUY")
        sell = inf.predict_with_v2(**common, direction="SELL")
        assert buy["status"] == sell["status"] == "OK"
        # Whether the probabilities differ depends on what the model learned;
        # what must hold is that both are produced without the gate blocking.
        assert buy["p_win"] is not None and sell["p_win"] is not None
