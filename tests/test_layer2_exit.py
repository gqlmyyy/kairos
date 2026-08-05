"""Layer 2 tests: signal-flip hard override and the unified Exit Score."""

from __future__ import annotations

import pytest

from conftest import make_ctx
from trade_management import layer2_exit_score, layer2_signal_flip
from trade_management.layer2_exit_probability import (
    ProbabilityAssessment,
    ProbabilityHistory,
)


def _prob(value, qualified=True, multiplier=1.0, influences=True):
    return ProbabilityAssessment(
        available=True,
        probability=value,
        qualified=qualified,
        weight_multiplier=multiplier,
        influences_decision=influences,
        reason="test",
    )


# -------------------------------------------------------------- signal flip
class TestSignalFlip:
    def test_confirmed_opposite_signal_closes_immediately(self, settings):
        ctx = make_ctx(direction="buy")
        signal = {
            "direction": "SELL", "final_score": 60.0,
            "ai_confidence": 0.8, "mtf_aligned": True,
        }
        result = layer2_signal_flip.check_signal_flip(ctx, signal, settings)
        assert result.close_full
        assert result.terminal

    def test_same_direction_signal_is_ignored(self, settings):
        ctx = make_ctx(direction="buy")
        signal = {
            "direction": "BUY", "final_score": 90.0,
            "ai_confidence": 0.9, "mtf_aligned": True,
        }
        assert not layer2_signal_flip.check_signal_flip(ctx, signal, settings).close_full

    # --- edge cases: an unconfirmed flip must not close ---
    @pytest.mark.parametrize(
        "signal",
        [
            {"direction": "SELL", "final_score": 20.0, "ai_confidence": 0.9, "mtf_aligned": True},
            {"direction": "SELL", "final_score": 90.0, "ai_confidence": 0.2, "mtf_aligned": True},
            {"direction": "SELL", "final_score": 90.0, "ai_confidence": 0.9, "mtf_aligned": False},
        ],
        ids=["low_score", "low_confidence", "not_mtf_aligned"],
    )
    def test_unconfirmed_flip_does_not_close(self, settings, signal):
        ctx = make_ctx(direction="buy")
        assert not layer2_signal_flip.check_signal_flip(ctx, signal, settings).close_full

    def test_missing_signal_is_not_a_flip(self, settings):
        ctx = make_ctx(direction="buy")
        assert not layer2_signal_flip.check_signal_flip(ctx, None, settings).close_full

    def test_unparseable_signal_is_not_a_flip(self, settings):
        ctx = make_ctx(direction="buy")
        signal = {"direction": "SELL", "final_score": "abc", "ai_confidence": None}
        assert not layer2_signal_flip.check_signal_flip(ctx, signal, settings).close_full


# --------------------------------------------------------------- exit score
class TestExitScore:
    def test_all_components_calm_scores_low(self, settings):
        ctx = make_ctx(current_price=101.0)
        readings = {"trend_score": 80.0, "momentum_score": 75.0}
        breakdown = layer2_exit_score.compute_exit_score(
            ctx, readings, _prob(0.1), settings
        )
        assert breakdown.score < 0.3
        assert not breakdown.should_close

    def test_all_components_bearish_closes_a_long(self, settings):
        ctx = make_ctx(current_price=101.0)
        readings = {"trend_score": 2.0, "momentum_score": 2.0}
        breakdown = layer2_exit_score.compute_exit_score(
            ctx, readings, _prob(0.98), settings
        )
        assert breakdown.score > breakdown.threshold
        assert breakdown.should_close

    def test_weights_are_normalised_over_available_components(self, settings):
        """A missing component redistributes rather than scoring zero."""
        ctx = make_ctx(current_price=101.0)
        readings = {"trend_score": 0.0}  # fully reversed, nothing else available
        breakdown = layer2_exit_score.compute_exit_score(ctx, readings, None, settings)
        assert breakdown.score == pytest.approx(1.0)

    def test_volume_component_is_off_by_default(self, settings):
        ctx = make_ctx(current_price=101.0)
        readings = {"trend_score": 50.0, "volume_weakness": 1.0}
        breakdown = layer2_exit_score.compute_exit_score(ctx, readings, None, settings)
        assert "volume_weakness" not in breakdown.components

    def test_volume_component_activates_when_weighted(self, settings):
        """Documented re-enable path: give it a weight and supply the reading."""
        settings["EXIT_WEIGHT_VOLUME_WEAKNESS"] = 0.15
        ctx = make_ctx(current_price=101.0)
        readings = {"trend_score": 50.0, "volume_weakness": 1.0}
        breakdown = layer2_exit_score.compute_exit_score(ctx, readings, None, settings)
        assert "volume_weakness" in breakdown.components

    # --- edge cases ---
    def test_no_data_at_all_scores_zero(self, settings):
        breakdown = layer2_exit_score.compute_exit_score(make_ctx(), {}, None, settings)
        assert breakdown.score == 0.0
        assert not breakdown.should_close

    def test_score_exactly_at_threshold_does_not_close(self, settings):
        """Spec says close when score > threshold, not >=."""
        ctx = make_ctx(current_price=101.0)
        settings["EXIT_SCORE_THRESHOLD"] = 1.0
        readings = {"trend_score": 0.0}
        breakdown = layer2_exit_score.compute_exit_score(ctx, readings, None, settings)
        assert breakdown.score == pytest.approx(1.0)
        assert not breakdown.should_close

    def test_sell_side_reads_trend_inverted(self, settings):
        """A low trend score is bullish, so it opposes a short."""
        ctx = make_ctx(direction="sell", current_price=99.0, sl=101.0, initial_sl=101.0)
        breakdown = layer2_exit_score.compute_exit_score(
            ctx, {"trend_score": 0.0}, None, settings
        )
        assert breakdown.components["trend_reversal"] == pytest.approx(0.0)

    def test_unqualified_probability_is_damped(self, settings):
        ctx = make_ctx(current_price=101.0)
        readings = {"trend_score": 50.0, "momentum_score": 50.0}
        strong = layer2_exit_score.compute_exit_score(
            ctx, readings, _prob(1.0, qualified=True, multiplier=1.0), settings
        )
        damped = layer2_exit_score.compute_exit_score(
            ctx, readings, _prob(1.0, qualified=False, multiplier=0.25), settings
        )
        assert damped.score < strong.score

    def test_shadow_mode_probability_never_influences(self, settings):
        ctx = make_ctx(current_price=101.0)
        readings = {"trend_score": 50.0}
        shadow = _prob(1.0, influences=False)
        breakdown = layer2_exit_score.compute_exit_score(ctx, readings, shadow, settings)
        assert "probability" not in breakdown.components


# --------------------------------------------------- probability qualification
class TestProbabilityHistory:
    def test_counts_consecutive_declines(self):
        h = ProbabilityHistory()
        for v in (0.5, 0.4, 0.3):
            h.record(v)
        assert h.consecutive_declines() == 2

    def test_a_rise_resets_the_decline_streak(self):
        h = ProbabilityHistory()
        for v in (0.5, 0.4, 0.45):
            h.record(v)
        assert h.consecutive_declines() == 0

    def test_drop_from_entry_is_fractional(self):
        h = ProbabilityHistory()
        h.record(0.80)
        h.record(0.40)
        assert h.drop_from_entry() == pytest.approx(0.5)

    # --- edge cases ---
    def test_single_reading_has_no_streak(self):
        h = ProbabilityHistory()
        h.record(0.5)
        assert h.consecutive_declines() == 0
        assert h.drop_from_entry() == 0.0

    def test_improvement_reports_no_drop(self):
        h = ProbabilityHistory()
        h.record(0.4)
        h.record(0.9)
        assert h.drop_from_entry() == 0.0
