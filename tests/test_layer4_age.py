"""Layer 4 tests: trade age phases and the internal time-stop condition."""

from __future__ import annotations

import pytest

from conftest import make_ctx
from trade_management import layer4_trade_age as age


class TestPhases:
    @pytest.mark.parametrize(
        "bars,expected",
        [(0, age.PHASE_SETTLE), (5, age.PHASE_SETTLE),
         (6, age.PHASE_TRAIL), (12, age.PHASE_TRAIL),
         (13, age.PHASE_TIGHTEN), (15, age.PHASE_TIGHTEN)],
    )
    def test_phase_boundaries(self, settings, bars, expected):
        # Keep profit above the time-stop floor so only the phase is under test.
        ctx = make_ctx(bars_open=bars, current_price=102.0)
        assert age.assess_age(ctx, settings).phase == expected

    def test_settle_phase_blocks_large_changes(self, settings):
        ctx = make_ctx(bars_open=2, current_price=102.0)
        assert not age.assess_age(ctx, settings).allow_large_changes

    def test_trail_phase_allows_large_changes(self, settings):
        ctx = make_ctx(bars_open=8, current_price=102.0)
        assert age.assess_age(ctx, settings).allow_large_changes

    def test_scale_decreases_as_the_trade_ages(self, settings):
        scales = [
            age.assess_age(make_ctx(bars_open=b, current_price=102.0), settings).age_scale
            for b in (2, 8, 12, 14)
        ]
        assert scales == sorted(scales, reverse=True)


class TestTimeStop:
    def test_closes_a_stalled_trade(self, settings):
        # 11 bars, +0.1R -> below the 0.3R floor.
        ctx = make_ctx(bars_open=11, current_price=100.1)
        result = age.evaluate(ctx, settings)
        assert result.close_full
        assert any("time_stop" in r for r in result.reasons)

    def test_leaves_a_profitable_trade_alone(self, settings):
        # 11 bars, +1.5R -> comfortably above the floor.
        ctx = make_ctx(bars_open=11, current_price=101.5)
        assert not age.evaluate(ctx, settings).close_full

    def test_young_stalled_trade_is_untouched(self, settings):
        ctx = make_ctx(bars_open=3, current_price=100.05)
        assert not age.evaluate(ctx, settings).close_full

    # --- edge cases ---
    def test_hard_ceiling_closes_regardless_of_profit(self, settings):
        ctx = make_ctx(bars_open=20, current_price=110.0)  # +10R but expired
        assert age.evaluate(ctx, settings).close_full

    def test_exactly_at_min_bars_and_below_floor_closes(self, settings):
        ctx = make_ctx(bars_open=10, current_price=100.2)  # +0.2R
        assert age.evaluate(ctx, settings).close_full

    def test_exactly_at_the_profit_floor_survives(self, settings):
        ctx = make_ctx(bars_open=11, current_price=100.3)  # exactly +0.3R
        assert not age.evaluate(ctx, settings).close_full

    def test_losing_trade_past_min_bars_closes(self, settings):
        ctx = make_ctx(bars_open=11, current_price=99.5)  # -0.5R
        assert age.evaluate(ctx, settings).close_full

    def test_disabled_layer_never_closes(self, settings):
        settings["TRADE_AGE_ENABLED"] = False
        ctx = make_ctx(bars_open=50, current_price=100.0)
        assert not age.evaluate(ctx, settings).close_full

    def test_profile_can_extend_the_ceiling(self, settings):
        settings["TIME_STOP_MAX_BARS"] = 30
        settings["TIME_STOP_MIN_BARS"] = 25
        ctx = make_ctx(bars_open=20, current_price=100.1)
        assert not age.evaluate(ctx, settings).close_full

    def test_meta_carries_the_scale_for_layer3(self, settings):
        ctx = make_ctx(bars_open=8, current_price=102.0)
        result = age.evaluate(ctx, settings)
        assert "age_scale" in result.meta
        assert 0.0 < result.meta["age_scale"] <= 1.0
