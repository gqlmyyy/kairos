"""How Exit Score actually behaves today, with the ML model fully disabled.

The exit model is off indefinitely (see KNOWN_ISSUES.md 2-3 and ROADMAP.md), so
the `probability` component supplies no reading. These tests pin what that means
in practice, because "47% of the weight" reading in tm_config is easy to
misinterpret as "47% of the score is now frozen".

It is not. The scorer takes a weighted mean over *available* components, so a
component with no data drops out and its weight is redistributed — the same
mechanism used for the disabled volume component.
"""

from __future__ import annotations

import pytest

from conftest import make_ctx
from trade_management import layer2_exit_score as es
from trade_management.layer2_exit_probability import (
    ProbabilityHistory,
    compute_exit_probability,
)


@pytest.fixture
def ml_off(settings):
    """Settings with the exit model disabled — the production configuration."""
    settings["ML_EXIT_ENABLED"] = False
    settings["ML_EXIT_SHADOW_MODE"] = False
    settings["EXIT_SCORE_THRESHOLD"] = 0.75
    return settings


@pytest.fixture
def disabled_probability(ml_off):
    return compute_exit_probability({}, ProbabilityHistory(), ml_off)


class TestProbabilityIsExcludedNotZeroed:
    """Answer to 'what value does probability contribute?': none at all."""

    def test_provider_reports_unavailable(self, disabled_probability):
        assert disabled_probability.available is False
        assert disabled_probability.probability is None
        assert disabled_probability.influences_decision is False
        assert disabled_probability.reason == "ml_exit_disabled"

    def test_component_is_absent_not_neutral_and_not_zero(self, ml_off, disabled_probability):
        ctx = make_ctx(current_price=101.0)
        breakdown = es.compute_exit_score(
            ctx, {"trend_score": 50.0, "momentum_score": 50.0},
            disabled_probability, ml_off,
        )
        # Not 0.5 (a neutral constant) and not 0.0 (a false "calm" vote):
        # the key simply is not there.
        assert "probability" not in breakdown.components
        assert "probability" not in breakdown.weights

    def test_weight_redistributes_over_the_two_live_components(self, ml_off, disabled_probability):
        ctx = make_ctx(current_price=101.0)
        breakdown = es.compute_exit_score(
            ctx, {"trend_score": 0.0, "momentum_score": 0.0},
            disabled_probability, ml_off,
        )
        total = sum(breakdown.weights.values())
        shares = {k: v / total for k, v in breakdown.weights.items()}
        assert shares["trend_reversal"] == pytest.approx(0.5555, abs=1e-3)
        assert shares["momentum_weakness"] == pytest.approx(0.4445, abs=1e-3)
        # Relative sizing is preserved: 0.2941 / 0.2353 before and after.
        assert shares["trend_reversal"] / shares["momentum_weakness"] == pytest.approx(
            0.2941 / 0.2353, abs=1e-3
        )


class TestThreeScenarios:
    """The three cases requested: strong, medium, weak — with ML off."""

    def test_weak_readings_score_zero_and_hold(self, ml_off, disabled_probability):
        """Market still supports the trade: nothing to act on."""
        ctx = make_ctx(current_price=101.0)
        breakdown = es.compute_exit_score(
            ctx, {"trend_score": 90.0, "momentum_score": 85.0},
            disabled_probability, ml_off,
        )
        assert breakdown.score == pytest.approx(0.0)
        assert not breakdown.should_close

    def test_medium_readings_hold(self, ml_off, disabled_probability):
        """Neutral market: well below the threshold, no exit."""
        ctx = make_ctx(current_price=101.0)
        breakdown = es.compute_exit_score(
            ctx, {"trend_score": 50.0, "momentum_score": 50.0},
            disabled_probability, ml_off,
        )
        assert breakdown.score == pytest.approx(0.0)
        assert not breakdown.should_close

    def test_strong_readings_still_close(self, ml_off, disabled_probability):
        """The key property: 0.75 remains reachable without the model."""
        ctx = make_ctx(current_price=101.0)
        breakdown = es.compute_exit_score(
            ctx, {"trend_score": 0.0, "momentum_score": 0.0},
            disabled_probability, ml_off,
        )
        assert breakdown.score == pytest.approx(1.0)
        assert breakdown.should_close

    def test_threshold_is_reachable_short_of_the_extremes(self, ml_off, disabled_probability):
        """Not only a perfect 1.0 clears it — genuinely strong readings do too."""
        ctx = make_ctx(current_price=101.0)
        breakdown = es.compute_exit_score(
            ctx, {"trend_score": 10.0, "momentum_score": 15.0},
            disabled_probability, ml_off,
        )
        assert breakdown.score == pytest.approx(0.756, abs=1e-3)
        assert breakdown.should_close

    def test_moderately_bearish_readings_do_not_close(self, ml_off, disabled_probability):
        """Guard against the threshold becoming trivially easy to trip."""
        ctx = make_ctx(current_price=101.0)
        breakdown = es.compute_exit_score(
            ctx, {"trend_score": 30.0, "momentum_score": 30.0},
            disabled_probability, ml_off,
        )
        assert breakdown.score == pytest.approx(0.4, abs=1e-3)
        assert not breakdown.should_close


class TestReEnablingRestoresTheWeight:
    """When a valid model exists, probability re-enters with no config change."""

    def test_available_probability_reclaims_its_share(self, settings):
        from trade_management.layer2_exit_probability import ProbabilityAssessment

        settings["EXIT_SCORE_THRESHOLD"] = 0.75
        ctx = make_ctx(current_price=101.0)
        live = ProbabilityAssessment(
            available=True, probability=0.9, qualified=True,
            weight_multiplier=1.0, influences_decision=True, reason="qualified",
        )
        breakdown = es.compute_exit_score(
            ctx, {"trend_score": 0.0, "momentum_score": 0.0}, live, settings
        )
        total = sum(breakdown.weights.values())
        assert "probability" in breakdown.components
        assert breakdown.weights["probability"] / total == pytest.approx(0.4706, abs=1e-3)

    def test_shadow_mode_still_does_not_influence(self, settings):
        from trade_management.layer2_exit_probability import ProbabilityAssessment

        ctx = make_ctx(current_price=101.0)
        shadow = ProbabilityAssessment(
            available=True, probability=0.99, qualified=True,
            weight_multiplier=0.0, influences_decision=False, reason="shadow_mode",
        )
        breakdown = es.compute_exit_score(
            ctx, {"trend_score": 50.0, "momentum_score": 50.0}, shadow, settings
        )
        assert "probability" not in breakdown.components
