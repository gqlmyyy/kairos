"""One description of the features, shared by training and by live inference.

Three incompatible schemas reached `models/entry/entry_model.json` because
nothing forced the two paths to agree on what a feature *is*. The contract in
`entry_feature_spec.FEATURE_CONTRACT` is that description; these tests keep it
honest, since a contract nobody checks is a comment.
"""

from __future__ import annotations

import math

import pytest

from analysis.models import entry_feature_spec as spec
from analysis.models import entry_model_metadata as md


class TestContractCompleteness:
    def test_it_describes_exactly_the_deployed_vector_in_order(self):
        assert tuple(spec.FEATURE_CONTRACT) == spec.FEATURE_NAMES

    def test_every_feature_declares_every_required_field(self):
        spec.validate_contract()

    def test_an_undescribed_feature_fails_loudly(self, monkeypatch):
        """Non-vacuity: adding a feature without a contract entry must break."""
        monkeypatch.setattr(spec, "FEATURE_NAMES", spec.FEATURE_NAMES + ("invented",))
        with pytest.raises(RuntimeError, match="undescribed"):
            spec.validate_contract()

    def test_a_missing_field_fails_loudly(self, monkeypatch):
        damaged = {k: dict(v) for k, v in spec.FEATURE_CONTRACT.items()}
        damaged["rsi"].pop("availability")
        monkeypatch.setattr(spec, "FEATURE_CONTRACT", damaged)
        with pytest.raises(RuntimeError, match="missing contract fields"):
            spec.validate_contract()

    def test_required_history_is_a_real_bar_count(self):
        assert spec.REQUIRED_HISTORY_BARS >= 100
        for name, entry in spec.FEATURE_CONTRACT.items():
            assert isinstance(entry["required_history"], int), name
            assert entry["required_history"] >= 0, name


class TestContractSemantics:
    def test_every_feature_is_available_at_the_decision_timestamp(self):
        """The entry_v2 defect stated as a property.

        No feature may depend on a candle that closes after the decision. Any
        entry whose availability does not say so is the one to look at first.
        """
        for name, entry in spec.FEATURE_CONTRACT.items():
            assert "decision timestamp" in entry["availability"], (
                f"{name} does not declare availability at the decision moment: "
                f"{entry['availability']!r}")

    def test_no_feature_is_sourced_from_the_quarantined_pipeline(self):
        for name, entry in spec.FEATURE_CONTRACT.items():
            assert "entry_v2" not in entry["source"], name
            assert "entry_v2" not in entry["training_usage"], name

    def test_valid_ranges_are_ordered_pairs(self):
        for name, entry in spec.FEATURE_CONTRACT.items():
            low, high = entry["valid_range"]
            assert low <= high, name

    def test_built_vectors_land_inside_their_declared_ranges(self):
        vector = spec.build_feature_vector(
            rsi=55.0, atr=0.0018, macd=0.0004, trend_strength="strong",
            trend_score=70.0, momentum_score=65.0, volatility_score=55.0,
            market_regime="HIGH_VOLATILITY", session="london", direction="BUY",
        )
        for value, name in zip(vector, spec.FEATURE_NAMES):
            low, high = spec.FEATURE_CONTRACT[name]["valid_range"]
            assert low <= value <= high, f"{name}={value} outside ({low}, {high})"

    def test_missing_inputs_land_on_the_declared_defaults(self):
        vector = spec.as_named_dict(spec.build_feature_vector(
            rsi=None, atr=None, macd=None, trend_strength=None,
            trend_score=None, momentum_score=None, volatility_score=None,
            market_regime=None, session=None, direction=None,
        ))
        assert vector["rsi"] == spec.MISSING_DEFAULTS["rsi"]
        assert vector["trend_score"] == spec.MISSING_DEFAULTS["trend_score"]
        assert vector["market_regime"] == float(spec.REGIME_DEFAULT)
        assert vector["trend_strength"] == spec.TREND_STRENGTH_DEFAULT

    def test_unknown_is_not_confused_with_a_real_regime(self):
        """"Could not determine" and "ranging" are different statements."""
        unknown = spec.encode_regime(None)
        real = {spec.encode_regime(r) for r in
                ("RANGING", "TRENDING", "HIGH_VOLATILITY", "LOW_VOLATILITY")}
        assert unknown not in real
        assert len(real) == 4, "two regimes still share an encoding"

    def test_unmeasured_trend_strength_is_not_confused_with_weak(self):
        assert spec.encode_trend_strength(None) != spec.encode_trend_strength("weak")


class TestContractIsWiredIntoTheGate:
    def test_the_live_names_are_the_contract_names(self):
        from analysis.models import xgboost_v2_inference as inference

        assert tuple(inference.LIVE_FEATURE_NAMES) == tuple(spec.FEATURE_CONTRACT)

    def test_a_model_built_for_these_names_is_servable(self):
        payload = md.build(
            model_version="contract-test",
            training_pipeline_id="tests/test_feature_contract.py",
            feature_names=list(spec.FEATURE_NAMES),
            dataset_fingerprint="sha256:" + "c" * 64,
            training_start="2024-01-01T00:00:00+00:00",
            training_end="2024-06-01T00:00:00+00:00",
            validation_start="2024-06-01T00:00:00+00:00",
            validation_end="2024-09-01T00:00:00+00:00",
            test_start="2024-09-01T00:00:00+00:00",
            test_end="2024-12-01T00:00:00+00:00",
            symbol_scope=["EURUSD"], timeframe_scope=["H4", "H1"],
            target_definition="first touch of TP before SL",
            target_horizon=24, git_commit="d" * 40,
        )
        meta = md.parse(payload)
        assert md.validate_for_serving(
            meta, live_feature_names=spec.FEATURE_NAMES) is None

    def test_a_model_built_for_the_65_feature_schema_is_not(self):
        """The deployed artifact's schema, stated as a contract check."""
        payload = md.build(
            model_version="entry-v2-legacy",
            training_pipeline_id="analysis/entry_v2/entry_xgboost_trainer.py",
            feature_names=[f"h4_feature_{i}" for i in range(65)],
            dataset_fingerprint="sha256:" + "e" * 64,
            training_start="2024-01-01T00:00:00+00:00",
            training_end="2024-06-01T00:00:00+00:00",
            validation_start="2024-06-01T00:00:00+00:00",
            validation_end="2024-09-01T00:00:00+00:00",
            test_start="2024-09-01T00:00:00+00:00",
            test_end="2024-12-01T00:00:00+00:00",
            symbol_scope=["EURUSD"], timeframe_scope=["H4", "H1"],
            target_definition="tp-first on an EMA entry price",
            target_horizon=24, git_commit="f" * 40,
        )
        reason = md.validate_for_serving(
            md.parse(payload), live_feature_names=spec.FEATURE_NAMES)
        assert reason is not None and "name mismatch" in reason
