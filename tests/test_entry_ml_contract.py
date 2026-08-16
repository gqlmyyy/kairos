"""C-01 regression: the entry ML gate must never predict on a mismatched vector.

The deployed artifact expects 65 features; live inference supplies 10. XGBoost
accepts that silently — absent columns become `missing` and each tree follows
its default branch — so it returned plausible probabilities that had no
relationship to the trade. Measured on the real artifact: BUY and SELL received
the *identical* p_win, and five of the ten supplied inputs moved it not at all.

These tests pin the contract check that now blocks such a prediction, and prove
a blocked gate cannot reach an MT5 order.
"""

from __future__ import annotations

import math

import pytest

from analysis.models import entry_feature_contract as contract
from analysis.models.entry_feature_contract import (
    FeatureContract,
    STATUS_INVALID,
    STATUS_MODEL_MISSING,
    STATUS_OK,
    STATUS_PREDICTION_ERROR,
)

NAMES_10 = ("rsi", "atr", "macd", "trend_strength", "trend_score",
            "momentum_score", "volatility_score", "market_regime",
            "session", "direction")
VALUES_10 = [50.0, 0.0012, 0.0, 0.0, 50.0, 50.0, 50.0, 1.0, 1.0, 1.0]


def contract_of(count, names=()):
    return FeatureContract(model_version="test", feature_count=count, feature_names=tuple(names))


class TestFeatureCountContract:
    def test_1_matching_count_is_allowed(self):
        assert contract.validate_features(VALUES_10, contract_of(10)) is None

    def test_2_sixtyfive_expected_ten_supplied_is_rejected(self):
        """The exact production defect."""
        reason = contract.validate_features(VALUES_10, contract_of(65))
        assert reason is not None
        assert "65" in reason and "10" in reason

    def test_5_extra_feature_is_rejected(self):
        reason = contract.validate_features(VALUES_10 + [1.0], contract_of(10))
        assert reason is not None
        assert "count mismatch" in reason

    def test_4_missing_feature_is_rejected(self):
        reason = contract.validate_features(VALUES_10[:-1], contract_of(10))
        assert reason is not None

    def test_unknown_model_feature_count_is_rejected(self):
        assert contract.validate_features(VALUES_10, contract_of(-1)) is not None


class TestFeatureOrderContract:
    def test_3_wrong_order_is_rejected(self):
        swapped = list(NAMES_10)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        reason = contract.validate_features(
            VALUES_10, contract_of(10, NAMES_10), supplied_names=swapped
        )
        assert reason is not None
        assert "order mismatch" in reason

    def test_correct_order_passes(self):
        assert contract.validate_features(
            VALUES_10, contract_of(10, NAMES_10), supplied_names=NAMES_10
        ) is None

    def test_renamed_feature_is_rejected(self):
        renamed = list(NAMES_10)
        renamed[3] = "trend_str"
        reason = contract.validate_features(
            VALUES_10, contract_of(10, NAMES_10), supplied_names=renamed
        )
        assert reason is not None

    def test_order_check_skipped_when_model_has_no_names(self):
        """The deployed artifact carries feature_names=None; count still guards."""
        assert contract.validate_features(
            VALUES_10, contract_of(10), supplied_names=NAMES_10
        ) is None


class TestNumericContract:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_6_7_nan_and_inf_features_are_rejected(self, bad):
        values = list(VALUES_10)
        values[1] = bad
        reason = contract.validate_features(values, contract_of(10, NAMES_10))
        assert reason is not None
        assert "atr" in reason

    def test_non_numeric_feature_is_rejected(self):
        values = list(VALUES_10)
        values[7] = "TRENDING"
        assert contract.validate_features(values, contract_of(10, NAMES_10)) is not None

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.5, "x"])
    def test_invalid_probability_is_rejected(self, bad):
        assert contract.validate_probability(bad) is not None

    @pytest.mark.parametrize("good", [0.0, 0.5, 1.0])
    def test_valid_probability_accepted(self, good):
        assert contract.validate_probability(good) is None


class TestGateResultSemantics:
    def test_invalid_never_reports_available(self):
        result = contract.invalid("mismatch")
        assert result.status == STATUS_INVALID
        assert result.p_win is None
        assert result.allowed is False
        assert result.available is False

    def test_10_model_missing_blocks(self):
        result = contract.model_missing("no artifact")
        assert result.status == STATUS_MODEL_MISSING
        assert result.available is False

    def test_prediction_error_blocks(self):
        result = contract.prediction_error("boom")
        assert result.status == STATUS_PREDICTION_ERROR
        assert result.available is False

    def test_8_ok_result_is_usable(self):
        result = contract.ok(0.72, contract_of(10))
        assert result.status == STATUS_OK
        assert result.allowed is True
        assert result.p_win == pytest.approx(0.72)

    def test_ok_with_none_probability_is_still_blocked(self):
        """Defensive: allowed requires an actual number, not just status OK."""
        from analysis.models.entry_feature_contract import GateResult

        assert GateResult(status=STATUS_OK, p_win=None).allowed is False


class TestLivePathBlocksTheRealArtifact:
    """End-to-end against the actual deployed model file."""

    def test_real_incident_inputs_are_now_blocked(self):
        from analysis.models.xgboost_v2_inference import predict_with_v2

        result = predict_with_v2(
            rsi=50.0, atr=47.35571, macd=0.0, trend_strength=0.0,
            trend_score=50.0, momentum_score=50.0, volatility_score=50.0,
            market_regime="TRENDING", direction="BUY",
        )
        # Before the fix this returned p_win=0.6228 and opened a position.
        assert result["available"] is False
        assert result["p_win"] is None
        # The block now happens one step earlier than it used to. The deployed
        # artifact carries no metadata sidecar, so the loader refuses it before
        # a vector is ever built and the gate reports ML_MODEL_MISSING rather
        # than ML_GATE_INVALID. Both are blocking; what this test pins is that
        # these inputs cannot produce a tradeable number.
        assert result["status"] in {STATUS_INVALID, "ML_MODEL_MISSING"}

    def test_9_live_feature_names_match_the_vector_built(self):
        """Schema drift guard: the declared names must match the code."""
        import inspect

        from analysis.models import xgboost_v2_inference as inf

        assert len(inf.LIVE_FEATURE_NAMES) == 10
        source = inspect.getsource(inf.predict_with_v2)
        # Every declared name must appear in the function that builds the vector.
        for name in inf.LIVE_FEATURE_NAMES:
            assert name in source, f"{name} declared but not built"

    def test_blocked_gate_cannot_produce_a_tradeable_decision(self):
        """PROPERTY 1: an invalid schema can never yield a trading decision.

        Mirrors main.py's gate: `if not model_available -> final_decision_valid = False`.
        """
        from analysis.models.xgboost_v2_inference import predict_with_v2

        result = predict_with_v2(
            rsi=70.0, atr=0.001, macd=0.0005, trend_strength=80.0,
            trend_score=90.0, momentum_score=85.0, volatility_score=60.0,
            market_regime="TRENDING", direction="BUY",
        )
        model_available = result["available"]
        final_decision_valid = bool(model_available)
        assert final_decision_valid is False

    def test_direction_no_longer_silently_ignored(self):
        """Both directions must be blocked identically — not silently equal."""
        from analysis.models.xgboost_v2_inference import predict_with_v2

        kwargs = dict(
            rsi=55.0, atr=0.0012, macd=0.0, trend_strength=0.0, trend_score=65.0,
            momentum_score=50.0, volatility_score=50.0, market_regime="TRENDING",
        )
        buy = predict_with_v2(**kwargs, direction="BUY")
        sell = predict_with_v2(**kwargs, direction="SELL")
        assert buy["available"] is False and sell["available"] is False
        assert buy["p_win"] is None and sell["p_win"] is None
