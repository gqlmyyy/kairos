"""Layer 3 tests: adaptive trailing (trend x volatility in one equation)."""

from __future__ import annotations

import pytest

from conftest import make_ctx
from trade_management import layer3_adaptive_trailing as trailing


class TestTrendFactor:
    def test_strong_trend_widens(self, settings):
        assert trailing.compute_trend_factor(90.0, settings) > 1.0

    def test_weak_trend_tightens(self, settings):
        assert trailing.compute_trend_factor(5.0, settings) < 1.0

    def test_factor_is_monotonic_in_trend_strength(self, settings):
        factors = [trailing.compute_trend_factor(s, settings) for s in (0, 25, 50, 70, 100)]
        assert factors == sorted(factors)

    # --- edge cases: clamped outside the band ---
    def test_clamped_below_and_above(self, settings):
        assert trailing.compute_trend_factor(-50.0, settings) == pytest.approx(
            trailing.compute_trend_factor(0.0, settings)
        )
        assert trailing.compute_trend_factor(500.0, settings) == pytest.approx(
            trailing.compute_trend_factor(100.0, settings)
        )


class TestVolatilityFactor:
    def test_rising_atr_widens(self, settings):
        assert trailing.compute_volatility_factor(1.5, 1.0, settings) > 1.0

    def test_falling_atr_tightens(self, settings):
        assert trailing.compute_volatility_factor(0.7, 1.0, settings) < 1.0

    def test_unchanged_atr_is_neutral(self, settings):
        assert trailing.compute_volatility_factor(1.0, 1.0, settings) == pytest.approx(1.0)

    # --- edge cases ---
    def test_extreme_spike_is_clamped(self, settings):
        assert trailing.compute_volatility_factor(100.0, 1.0, settings) == pytest.approx(
            settings["TRAILING_VOL_RATIO_MAX"]
        )

    def test_missing_entry_atr_is_neutral(self, settings):
        assert trailing.compute_volatility_factor(1.0, 0.0, settings) == pytest.approx(1.0)


class TestTrailingPlan:
    def test_inactive_below_activation_profit(self, settings):
        ctx = make_ctx(current_price=100.2)  # +0.2R
        assert not trailing.compute_trailing_plan(ctx, settings=settings).active

    def test_active_above_activation_profit(self, settings):
        ctx = make_ctx(current_price=102.0)  # +2R
        assert trailing.compute_trailing_plan(ctx, settings=settings).active

    def test_multiplier_stays_within_bounds(self, settings):
        ctx = make_ctx(current_price=110.0, trend_strength=100.0, atr_now=5.0, atr_at_entry=1.0)
        plan = trailing.compute_trailing_plan(ctx, settings=settings)
        assert settings["TRAILING_MIN_ATR_MULTIPLIER"] <= plan.atr_multiplier
        assert plan.atr_multiplier <= settings["TRAILING_MAX_ATR_MULTIPLIER"]

    def test_age_scale_tightens_the_trail(self, settings):
        ctx = make_ctx(current_price=105.0)
        wide = trailing.compute_trailing_plan(ctx, age_scale=1.0, settings=settings)
        tight = trailing.compute_trailing_plan(ctx, age_scale=0.5, settings=settings)
        assert tight.atr_multiplier < wide.atr_multiplier

    # --- edge cases ---
    def test_missing_atr_disables_the_layer(self, settings):
        ctx = make_ctx(current_price=105.0, atr_now=0.0)
        assert not trailing.compute_trailing_plan(ctx, settings=settings).active

    def test_disabled_flag_is_respected(self, settings):
        settings["ADAPTIVE_TRAILING_ENABLED"] = False
        ctx = make_ctx(current_price=105.0)
        assert not trailing.compute_trailing_plan(ctx, settings=settings).active


class TestTrailingEvaluate:
    def test_proposes_a_stop_behind_price(self, settings):
        ctx = make_ctx(current_price=105.0, sl=99.0)
        result = trailing.evaluate(ctx, settings=settings)
        assert result.new_sl is not None
        assert result.new_sl < ctx.current_price

    def test_sell_side_stop_sits_above_price(self, settings):
        ctx = make_ctx(direction="sell", current_price=95.0, sl=101.0, initial_sl=101.0)
        result = trailing.evaluate(ctx, settings=settings)
        assert result.new_sl is not None
        assert result.new_sl > ctx.current_price

    def test_never_moves_the_stop_backwards(self, settings):
        # Stop already tighter than anything the trail would propose.
        ctx = make_ctx(current_price=105.0, sl=104.9)
        assert trailing.evaluate(ctx, settings=settings).new_sl is None

    def test_open_ended_profile_clears_fixed_target(self, settings):
        settings["USE_FIXED_TP"] = False
        ctx = make_ctx(current_price=105.0, sl=99.0, tp=110.0)
        result = trailing.evaluate(ctx, settings=settings)
        assert result.new_tp == 0.0
        assert any("target_extended" in r for r in result.reasons)

    def test_fixed_tp_profile_leaves_target_alone(self, settings):
        settings["USE_FIXED_TP"] = True
        ctx = make_ctx(current_price=105.0, sl=99.0, tp=110.0)
        assert trailing.evaluate(ctx, settings=settings).new_tp is None


class TestCalibration:
    def test_higher_tolerance_loosens_the_trail(self, settings):
        ctx = make_ctx(current_price=105.0)
        loose = trailing.compute_calibration_factor(ctx, 0.60, settings)
        tight = trailing.compute_calibration_factor(ctx, 0.20, settings)
        assert loose > tight

    def test_adjustment_is_bounded(self, settings):
        ctx = make_ctx()
        extreme = trailing.compute_calibration_factor(ctx, 10.0, settings)
        assert extreme <= 1.0 + settings["MFE_CALIBRATION_MAX_ADJUST"] + 1e-9
