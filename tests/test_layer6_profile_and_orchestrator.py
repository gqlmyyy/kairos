"""Layer 6 tests plus orchestrator ordering.

The orchestrator tests are what enforce the agreed execution order: a hard
override must beat everything below it, and the minimum-modify filter must be
the last thing between a decision and the broker.
"""

from __future__ import annotations

import pytest

from conftest import make_ctx
from trade_management import TradeManagementOrchestrator
from trade_management import layer6_trade_profile as profile


class TestProfileClassification:
    @pytest.mark.parametrize(
        "regime,expected",
        [
            ("trending", "trend"),
            ("strong uptrend", "trend"),
            ("breakout", "breakout"),
            ("high_volatility", "breakout"),
            ("mean_reversion", "mean_reversion"),
            ("ranging", "range"),
            ("sideways", "range"),
        ],
    )
    def test_regime_maps_to_profile(self, regime, expected):
        assert profile.classify_entry(regime, mtf_aligned=True) == expected

    def test_unaligned_trend_downgrades_to_breakout(self):
        assert profile.classify_entry("trending", mtf_aligned=False) == "breakout"

    # --- edge cases ---
    def test_unknown_regime_falls_back_to_default(self):
        from trade_management import tm_config as C

        assert profile.classify_entry("wat", mtf_aligned=True) == C.DEFAULT_PROFILE

    def test_none_regime_falls_back_to_default(self):
        from trade_management import tm_config as C

        assert profile.classify_entry(None) == C.DEFAULT_PROFILE


class TestProfileSettings:
    def test_trend_profile_is_open_ended(self):
        assert profile.resolve_settings("trend")["USE_FIXED_TP"] is False

    def test_mean_reversion_uses_a_fixed_target(self):
        assert profile.resolve_settings("mean_reversion")["USE_FIXED_TP"] is True

    def test_trend_trails_wider_than_range(self):
        trend = profile.resolve_settings("trend")
        rng = profile.resolve_settings("range")
        assert trend["TRAILING_BASE_ATR_MULTIPLIER"] > rng["TRAILING_BASE_ATR_MULTIPLIER"]

    def test_every_profile_resolves_all_tunables(self):
        for name in profile.C.VALID_PROFILES:
            s = profile.resolve_settings(name)
            for key in ("EXIT_SCORE_THRESHOLD", "TRAILING_BASE_ATR_MULTIPLIER",
                        "TIME_STOP_MAX_BARS", "PARTIAL_TP_LADDER", "BREAKEVEN_TRIGGER_R"):
                assert key in s, f"{name} missing {key}"

    def test_unknown_profile_falls_back_without_raising(self):
        assert profile.resolve_settings("nonsense")["PROFILE"] == profile.C.DEFAULT_PROFILE

    def test_stored_profile_wins_over_reclassification(self):
        name, _ = profile.profile_for_trade(stored_profile="range", regime="trending")
        assert name == "range"


class TestOrchestratorOrdering:
    def _orch(self):
        return TradeManagementOrchestrator()

    def test_signal_flip_beats_everything_below_it(self, settings):
        """Hard override wins even when lower layers also want to act."""
        ctx = make_ctx(current_price=102.0, bars_open=11)  # partial + trail eligible
        signal = {
            "direction": "SELL", "final_score": 90.0,
            "ai_confidence": 0.9, "mtf_aligned": True,
        }
        outcome = self._orch().manage_open_trade(ctx, settings, signal=signal)
        assert outcome.close_full
        assert outcome.close_fraction == 0.0
        assert outcome.modify is None
        assert any("signal_flip" in r for r in outcome.reasons)

    def test_exit_score_close_short_circuits_lower_layers(self, settings):
        ctx = make_ctx(current_price=102.0)
        readings = {"trend_score": 0.0, "momentum_score": 0.0}
        outcome = self._orch().manage_open_trade(ctx, settings, readings=readings)
        assert outcome.close_full
        assert outcome.modify is None

    def test_time_stop_close_short_circuits_lower_layers(self, settings):
        ctx = make_ctx(current_price=100.1, bars_open=11)
        outcome = self._orch().manage_open_trade(ctx, settings)
        assert outcome.close_full
        assert outcome.close_fraction == 0.0

    def test_settle_phase_suppresses_trailing_but_not_breakeven(self, settings):
        ctx = make_ctx(current_price=101.5, bars_open=2, sl=99.0)
        outcome = self._orch().manage_open_trade(ctx, settings)
        trailing = next(r for r in outcome.layer_results if r.layer == "adaptive_trailing")
        assert "settle" in trailing.reasons[0]
        assert outcome.modify is not None  # break-even still moved the stop

    def test_partial_and_trail_can_happen_in_one_pass(self, settings):
        ctx = make_ctx(current_price=102.0, bars_open=8, sl=99.0, breakeven_done=True)
        outcome = self._orch().manage_open_trade(ctx, settings)
        assert outcome.close_fraction > 0
        assert outcome.modify is not None

    def test_min_modify_filter_is_the_last_gate(self, settings):
        """A stop change under the threshold must never reach the broker."""
        settings["MIN_MODIFY_DISTANCE_POINTS"] = 1e9
        ctx = make_ctx(current_price=105.0, bars_open=8, sl=99.0)
        outcome = self._orch().manage_open_trade(ctx, settings)
        assert outcome.modify is None
        assert outcome.rejected_modify_reason

    def test_most_protective_stop_wins_among_proposals(self, settings):
        ctx = make_ctx(current_price=105.0, bars_open=8, sl=99.0)
        outcome = self._orch().manage_open_trade(ctx, settings)
        assert outcome.modify is not None
        # Break-even proposes ~100.05; trailing proposes higher. Tighter wins.
        assert outcome.modify.new_sl > ctx.entry_price

    def test_quiet_trade_produces_no_action(self, settings):
        ctx = make_ctx(current_price=100.1, bars_open=2)
        outcome = self._orch().manage_open_trade(ctx, settings)
        assert not outcome.has_action

    # --- edge cases ---
    def test_missing_inputs_do_not_raise(self, settings):
        ctx = make_ctx()
        outcome = self._orch().manage_open_trade(
            ctx, settings, signal=None, readings=None, exit_features=None
        )
        assert outcome is not None

    def test_forget_trade_clears_per_trade_state(self, settings):
        orch = self._orch()
        ctx = make_ctx(current_price=102.0)
        orch.manage_open_trade(ctx, settings)
        orch.forget_trade(ctx.order_id)
        assert ctx.order_id not in orch._probability_history

    def test_layer_results_cover_the_full_chain(self, settings):
        ctx = make_ctx(current_price=102.0, bars_open=8)
        outcome = self._orch().manage_open_trade(ctx, settings)
        layers = [r.layer for r in outcome.layer_results]
        assert layers[0] == "signal_flip"
        assert "exit_score" in layers
        assert "trade_age" in layers
        assert layers.index("breakeven") < layers.index("partial_tp")
        assert layers.index("partial_tp") < layers.index("adaptive_trailing")
