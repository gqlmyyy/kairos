"""Layer 1 tests: initial protection, break-even, minimum modify distance."""

from __future__ import annotations

import pytest

from conftest import make_ctx
from trade_management import layer1_breakeven, layer1_min_modify_distance
from trade_management.layer1_initial_protection import compute_initial_protection
from trade_management.layer1_intrabar import IntrabarState, bars_since, can_open_new_entry
from trade_management.types import ModifyRequest


# --------------------------------------------------------------- protection
class TestInitialProtection:
    def test_normal_regime_uses_base_multipliers(self):
        p = compute_initial_protection("EURUSD", atr=0.0010, regime="normal", settings={})
        assert p.sl_multiplier == pytest.approx(1.5)
        assert p.tp_multiplier == pytest.approx(2.5)
        assert p.tp_distance == pytest.approx(0.0025)

    def test_volatile_regime_widens_stop(self):
        normal = compute_initial_protection("EURUSD", 0.0010, "normal", settings={})
        volatile = compute_initial_protection("EURUSD", 0.0010, "high_volatility", settings={})
        assert volatile.sl_multiplier > normal.sl_multiplier

    def test_trending_regime_extends_target(self):
        normal = compute_initial_protection("EURUSD", 0.0010, "normal", settings={})
        trending = compute_initial_protection("EURUSD", 0.0010, "trending", settings={})
        assert trending.tp_multiplier > normal.tp_multiplier

    def test_apply_to_buy_and_sell_are_mirrored(self):
        p = compute_initial_protection("EURUSD", 0.0010, "normal", settings={})
        buy_sl, buy_tp = p.apply_to(1.1000, "buy")
        sell_sl, sell_tp = p.apply_to(1.1000, "sell")
        assert buy_sl < 1.1000 < buy_tp
        assert sell_tp < 1.1000 < sell_sl

    # --- edge cases ---
    def test_zero_atr_yields_zero_distances(self):
        p = compute_initial_protection("EURUSD", atr=0.0, regime="normal", settings={})
        assert p.sl_distance == 0.0
        assert p.tp_distance == 0.0

    def test_unknown_regime_falls_back_to_base(self):
        p = compute_initial_protection("EURUSD", 0.0010, "not_a_regime", settings={})
        assert p.sl_multiplier == pytest.approx(1.5)

    def test_open_ended_profile_clears_take_profit(self):
        p = compute_initial_protection(
            "EURUSD", 0.0010, "trending", settings={"USE_FIXED_TP": False}
        )
        _sl, tp = p.apply_to(1.1000, "buy")
        assert tp == 0.0


# --------------------------------------------------------------- break-even
class TestBreakeven:
    def test_moves_stop_to_entry_at_trigger(self, settings):
        ctx = make_ctx(current_price=101.0)  # +1.0R
        result = layer1_breakeven.apply_breakeven(ctx, settings)
        assert result.new_sl is not None
        assert result.new_sl >= ctx.entry_price

    def test_below_trigger_does_nothing(self, settings):
        ctx = make_ctx(current_price=100.5)  # +0.5R
        assert layer1_breakeven.apply_breakeven(ctx, settings).new_sl is None

    def test_sell_side_moves_stop_down_to_entry(self, settings):
        ctx = make_ctx(direction="sell", current_price=99.0, sl=101.0, initial_sl=101.0)
        result = layer1_breakeven.apply_breakeven(ctx, settings)
        assert result.new_sl is not None
        assert result.new_sl <= ctx.entry_price

    # --- edge cases ---
    def test_runs_only_once(self, settings):
        ctx = make_ctx(current_price=101.0, breakeven_done=True)
        assert layer1_breakeven.apply_breakeven(ctx, settings).new_sl is None

    def test_unknown_risk_is_a_noop(self, settings):
        ctx = make_ctx(current_price=101.0, r_distance=0.0)
        assert layer1_breakeven.apply_breakeven(ctx, settings).new_sl is None

    def test_does_not_move_stop_backwards(self, settings):
        # Stop is already above the break-even target.
        ctx = make_ctx(current_price=101.0, sl=100.5)
        assert layer1_breakeven.apply_breakeven(ctx, settings).new_sl is None


# ------------------------------------------------------- min modify distance
class TestMinModifyDistance:
    def _req(self, ctx, sl=None, tp=None):
        return ModifyRequest(ctx.order_id, ctx.symbol, ctx.direction, sl, tp)

    def test_approves_a_large_enough_move(self, settings):
        ctx = make_ctx(current_price=102.0, sl=99.0, point_size=0.01)
        # 1.0 price unit = 100 points, well over the 15-point minimum.
        verdict = layer1_min_modify_distance.filter_modification(
            self._req(ctx, sl=100.0), ctx, settings
        )
        assert verdict.approved

    def test_rejects_a_move_below_the_threshold(self, settings):
        ctx = make_ctx(current_price=102.0, sl=99.0, point_size=0.01)
        # 0.05 price units = 5 points, under the minimum.
        verdict = layer1_min_modify_distance.filter_modification(
            self._req(ctx, sl=99.05), ctx, settings
        )
        assert not verdict.approved
        assert "too_small" in verdict.reason

    def test_rejects_backwards_stop(self, settings):
        ctx = make_ctx(current_price=102.0, sl=100.0, point_size=0.01)
        verdict = layer1_min_modify_distance.filter_modification(
            self._req(ctx, sl=98.0), ctx, settings
        )
        assert not verdict.approved
        assert verdict.reason == "sl_not_an_improvement"

    # --- edge cases ---
    def test_rejects_stop_on_wrong_side_of_price(self, settings):
        ctx = make_ctx(current_price=100.5, sl=99.0, point_size=0.01)
        verdict = layer1_min_modify_distance.filter_modification(
            self._req(ctx, sl=101.0), ctx, settings
        )
        assert not verdict.approved

    def test_rejects_stop_inside_broker_stop_level(self, settings):
        ctx = make_ctx(
            current_price=102.0, sl=99.0, point_size=0.01, broker_stop_level_points=200.0
        )
        # 101.9 is only 10 points from price, inside the 200-point stop level.
        verdict = layer1_min_modify_distance.filter_modification(
            self._req(ctx, sl=101.9), ctx, settings
        )
        assert not verdict.approved
        assert "broker_stop_level" in verdict.reason

    def test_empty_request_is_rejected(self, settings):
        ctx = make_ctx()
        verdict = layer1_min_modify_distance.filter_modification(self._req(ctx), ctx, settings)
        assert not verdict.approved

    def test_disabled_filter_passes_everything(self, settings):
        ctx = make_ctx(current_price=102.0, sl=99.0, point_size=0.01)
        settings["MIN_MODIFY_ENABLED"] = False
        verdict = layer1_min_modify_distance.filter_modification(
            self._req(ctx, sl=99.001), ctx, settings
        )
        assert verdict.approved


# ----------------------------------------------------------------- intrabar
class TestIntrabar:
    def test_new_candle_allows_entry(self):
        state = IntrabarState()
        decision = can_open_new_entry("EURUSD", state, "H1", candle_ts=1000)
        assert decision.allow_new_entry

    def test_same_candle_blocks_entry(self):
        state = IntrabarState()
        state.commit("EURUSD", 1000)
        decision = can_open_new_entry("EURUSD", state, "H1", candle_ts=1000)
        assert not decision.allow_new_entry
        assert decision.reason == "same_candle"

    # --- edge cases ---
    def test_missing_boundary_blocks_rather_than_guessing(self, monkeypatch):
        monkeypatch.setattr(
            "trade_management.layer1_intrabar._fetch_last_closed_candle_ts",
            lambda *_a, **_k: None,
        )
        decision = can_open_new_entry("EURUSD", IntrabarState(), "H1")
        assert not decision.allow_new_entry
        assert decision.reason == "candle_boundary_unavailable"

    def test_per_symbol_tracking_is_independent(self):
        state = IntrabarState()
        state.commit("EURUSD", 1000)
        assert can_open_new_entry("GBPUSD", state, "H1", candle_ts=1000).allow_new_entry

    @pytest.mark.parametrize(
        "elapsed,timeframe,expected",
        [(3600, "H1", 1), (7200, "H1", 2), (3599, "H1", 0), (14400, "H4", 1)],
    )
    def test_bars_since(self, elapsed, timeframe, expected):
        assert bars_since(1_000_000, 1_000_000 + elapsed, timeframe) == expected

    def test_bars_since_handles_missing_input(self):
        assert bars_since(None, 1_000_000, "H1") == 0
