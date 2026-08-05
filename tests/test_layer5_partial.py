"""Layer 5 tests: partial TP ladder and MAE/MFE calibration."""

from __future__ import annotations

import pytest

from conftest import make_ctx
from trade_management import layer5_partial_tp as partial


class TestLadder:
    def test_nothing_due_below_first_level(self, settings):
        ctx = make_ctx(current_price=101.5)  # +1.5R, first level is +2R
        assert partial.next_due_level(ctx, settings) is None

    def test_first_level_due_at_2r(self, settings):
        ctx = make_ctx(current_price=102.0)
        level = partial.next_due_level(ctx, settings)
        assert level is not None
        assert level.index == 0
        assert level.fraction == pytest.approx(0.30)

    def test_second_level_due_at_3r(self, settings):
        ctx = make_ctx(current_price=103.0, partial_levels_done=(0,))
        level = partial.next_due_level(ctx, settings)
        assert level.index == 1

    def test_completed_levels_are_not_repeated(self, settings):
        ctx = make_ctx(current_price=102.5, partial_levels_done=(0,))
        assert partial.next_due_level(ctx, settings) is None

    def test_fraction_is_of_original_volume(self, settings):
        # Half already scaled out; the ladder still sizes off the original.
        ctx = make_ctx(current_price=102.0, initial_volume=2.0, volume=1.0)
        level = partial.next_due_level(ctx, settings)
        assert level.volume == pytest.approx(0.6)  # 30% of 2.0, not of 1.0

    def test_evaluate_emits_a_partial_close(self, settings):
        ctx = make_ctx(current_price=102.0)
        result = partial.evaluate(ctx, settings)
        assert result.close_fraction == pytest.approx(0.30)
        assert not result.close_full  # this layer never fully closes

    # --- edge cases ---
    def test_partial_below_min_lot_is_skipped(self, settings):
        ctx = make_ctx(current_price=102.0, initial_volume=0.01, volume=0.01)
        result = partial.evaluate(ctx, settings)
        assert result.close_fraction == 0.0

    def test_remainder_below_min_lot_is_skipped(self, settings):
        settings["MIN_REMAINING_VOLUME"] = 0.5
        ctx = make_ctx(current_price=102.0, initial_volume=1.0, volume=0.6)
        result = partial.evaluate(ctx, settings)
        assert result.close_fraction == 0.0
        assert "remainder_below_min_lot" in result.reasons[0]

    def test_unknown_risk_is_a_noop(self, settings):
        ctx = make_ctx(current_price=102.0, r_distance=0.0)
        assert partial.evaluate(ctx, settings).close_fraction == 0.0

    def test_disabled_layer_is_a_noop(self, settings):
        settings["PARTIAL_TP_ENABLED"] = False
        ctx = make_ctx(current_price=105.0)
        assert partial.evaluate(ctx, settings).close_fraction == 0.0

    def test_jumping_past_both_levels_takes_the_higher_one(self, settings):
        """A gap straight to +5R should not silently skip the ladder."""
        ctx = make_ctx(current_price=105.0)
        level = partial.next_due_level(ctx, settings)
        assert level.index == 1

    def test_sell_side_ladder_works(self, settings):
        ctx = make_ctx(direction="sell", current_price=98.0, sl=101.0, initial_sl=101.0)
        assert partial.evaluate(ctx, settings).close_fraction == pytest.approx(0.30)


class TestPullbackCalibration:
    def test_too_few_samples_keeps_the_default(self, settings):
        value = partial.compute_pullback_tolerance([0.1, 0.2, 0.3], settings)
        assert value == pytest.approx(settings["MFE_PULLBACK_TOLERANCE"])

    def test_enough_samples_uses_the_median(self, settings):
        settings["MFE_CALIBRATION_MIN_SAMPLES"] = 5
        value = partial.compute_pullback_tolerance([0.1, 0.2, 0.3, 0.4, 0.5], settings)
        assert value == pytest.approx(0.3)

    def test_median_resists_outliers(self, settings):
        settings["MFE_CALIBRATION_MIN_SAMPLES"] = 5
        value = partial.compute_pullback_tolerance([0.2, 0.2, 0.2, 0.2, 1.0], settings)
        assert value == pytest.approx(0.2)

    # --- edge cases ---
    def test_no_samples_keeps_the_default(self, settings):
        assert partial.compute_pullback_tolerance(None, settings) == pytest.approx(
            settings["MFE_PULLBACK_TOLERANCE"]
        )

    def test_out_of_range_and_junk_samples_are_discarded(self, settings):
        settings["MFE_CALIBRATION_MIN_SAMPLES"] = 3
        value = partial.compute_pullback_tolerance(
            [0.2, 0.3, 0.4, 5.0, -1.0, "junk", None], settings
        )
        assert value == pytest.approx(0.3)

    def test_calibration_never_emits_an_exit(self, settings):
        """MAE/MFE tunes trailing; it must not become an exit trigger."""
        ctx = make_ctx(current_price=102.0, mfe_r=5.0, mae_r=-3.0)
        result = partial.evaluate(ctx, settings)
        assert not result.close_full
