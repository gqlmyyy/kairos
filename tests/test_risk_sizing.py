"""Stop-distance and position-size regression tests.

These pin the fix for a live incident on 2026-08-07: a XAUUSD position was
opened and stopped out within the same second because the stop had been capped
to 0.497 price units against an ATR of 47.36 — roughly 1% of one ATR.

Two defects combined:

1. ``get_max_sl_distance`` shrank the stop so that trading MAX_LOT would risk
   under 5% of equity. It assumed 0.10 lots while the order was 0.01, and more
   fundamentally derived stop placement from an assumed position size rather
   than the other way round.

2. ``calculate_position_size`` used ``max(MIN_LOT, size)``, silently rounding an
   undersized position up to the broker minimum — which converts "this account
   cannot afford this trade" into "take a bigger one".

The second is the dangerous half: fixing only the stop would have produced a
0.01-lot XAUUSD trade risking $71 of a $99 account.
"""

from __future__ import annotations

import pytest

from config import PIP_VALUES
from trade_management.tm_config import MAX_SL_PIPS
from risk.position_sizing import MIN_LOT, calculate_position_size, get_pip_value_per_lot
from risk.symbol_info import get_max_sl_distance

# The exact conditions from the incident.
INCIDENT_EQUITY = 99.40
INCIDENT_ATR = {"XAUUSD": 47.35571, "EURUSD": 0.00175, "GBPUSD": 0.00226}


class TestStopDistanceIsNotCrushed:
    @pytest.mark.parametrize("symbol,atr", INCIDENT_ATR.items())
    def test_stop_is_a_meaningful_fraction_of_atr(self, symbol, atr):
        """The regression: the cap returned ~1% of ATR."""
        cap = get_max_sl_distance(
            symbol, max_sl_pips=MAX_SL_PIPS, atr=atr, account_equity=INCIDENT_EQUITY
        )
        assert cap > atr * 0.10, (
            f"{symbol}: stop ceiling {cap} is under 10% of ATR {atr} — "
            "this is the 2026-08-07 instant stop-out condition"
        )

    def test_xauusd_incident_value_no_longer_reproduces(self):
        cap = get_max_sl_distance(
            "XAUUSD", max_sl_pips=MAX_SL_PIPS,
            atr=INCIDENT_ATR["XAUUSD"], account_equity=INCIDENT_EQUITY,
        )
        assert cap != pytest.approx(0.497, abs=0.01)

    def test_equity_no_longer_influences_stop_distance(self):
        """Stop placement is a market decision; equity controls size instead."""
        kwargs = dict(symbol="XAUUSD", max_sl_pips=MAX_SL_PIPS, atr=INCIDENT_ATR["XAUUSD"])
        tiny = get_max_sl_distance(**kwargs, account_equity=50.0)
        large = get_max_sl_distance(**kwargs, account_equity=500_000.0)
        assert tiny == pytest.approx(large)

    def test_max_sl_pips_acts_as_a_ceiling_not_a_floor(self):
        """It was combined with max(), so an ATR spike overrode the setting."""
        pip = PIP_VALUES["XAUUSD"]
        cap = get_max_sl_distance("XAUUSD", max_sl_pips=100.0, atr=1000.0)
        assert cap <= 100.0 * pip + 1e-9, "MAX_SL_PIPS did not bound the stop"

    # --- edge cases ---
    def test_missing_atr_falls_back_to_the_pip_ceiling(self):
        cap = get_max_sl_distance("EURUSD", max_sl_pips=100.0, atr=None)
        assert cap == pytest.approx(100.0 * PIP_VALUES["EURUSD"])

    def test_zero_atr_is_treated_as_absent(self):
        cap = get_max_sl_distance("EURUSD", max_sl_pips=100.0, atr=0.0)
        assert cap > 0


class TestUndersizedPositionsAreRejected:
    """The half that matters most: refuse, do not round up."""

    @pytest.mark.parametrize("symbol,atr", INCIDENT_ATR.items())
    def test_incident_conditions_now_reject_the_trade(self, symbol, atr):
        size = calculate_position_size(
            INCIDENT_EQUITY, atr * 1.5, symbol, consecutive_losses=0, score=50
        )
        assert size == 0.0, f"{symbol} should be refused on a {INCIDENT_EQUITY} account"

    def test_rejection_blocks_entry_via_main_gate(self):
        """main.py gates on `position_size > 0`, so 0.0 stops the trade."""
        size = calculate_position_size(
            INCIDENT_EQUITY, INCIDENT_ATR["XAUUSD"] * 1.5, "XAUUSD", score=50
        )
        assert not (size is not None and size > 0)

    def test_minimum_lot_is_never_silently_substituted(self):
        """The old behaviour: 0.000035 lots became 0.01, risking 71% of equity."""
        symbol, atr = "XAUUSD", INCIDENT_ATR["XAUUSD"]
        sl = atr * 1.5
        size = calculate_position_size(INCIDENT_EQUITY, sl, symbol, score=50)

        pip = PIP_VALUES[symbol]
        risk_if_forced = (sl / pip) * get_pip_value_per_lot(symbol) * MIN_LOT
        assert risk_if_forced > INCIDENT_EQUITY * 0.5, "test premise no longer holds"
        assert size == 0.0

    def test_adequately_funded_account_still_trades(self):
        """The fix must not block legitimate trades on a larger account.

        At $50k the risk-correct size is ~0.0088 lots, just under the minimum.
        Rounding up risks $71 — 0.14% of equity, far below the hard ceiling —
        so the trade proceeds.
        """
        size = calculate_position_size(
            50_000.0, INCIDENT_ATR["XAUUSD"] * 1.5, "XAUUSD", score=75
        )
        assert size >= MIN_LOT

    def test_rounding_up_is_allowed_below_the_hard_ceiling(self):
        """Discrete lot sizes make small budget overshoot unavoidable."""
        from config import MAX_RISK_PER_TRADE_PCT

        equity, symbol = 50_000.0, "XAUUSD"
        sl = INCIDENT_ATR[symbol] * 1.5
        size = calculate_position_size(equity, sl, symbol, score=75)

        pip = PIP_VALUES[symbol]
        risk = (sl / pip) * get_pip_value_per_lot(symbol) * size
        assert size == MIN_LOT
        assert risk <= equity * MAX_RISK_PER_TRADE_PCT

    def test_hard_ceiling_is_what_separates_accept_from_reject(self):
        """Same symbol and stop; only equity differs."""
        symbol = "XAUUSD"
        sl = INCIDENT_ATR[symbol] * 1.5

        assert calculate_position_size(INCIDENT_EQUITY, sl, symbol, score=50) == 0.0
        assert calculate_position_size(50_000.0, sl, symbol, score=50) >= MIN_LOT

    def test_no_accepted_trade_ever_exceeds_the_hard_ceiling(self):
        """Sweep the incident conditions across a range of account sizes."""
        from config import MAX_RISK_PER_TRADE_PCT

        for symbol, atr in INCIDENT_ATR.items():
            sl = atr * 1.5
            pip = PIP_VALUES[symbol]
            pvl = get_pip_value_per_lot(symbol)
            for equity in (50, 99.40, 500, 1_000, 10_000, 100_000):
                size = calculate_position_size(equity, sl, symbol, score=50)
                if size == 0.0:
                    continue
                risk = (sl / pip) * pvl * size
                assert risk <= equity * MAX_RISK_PER_TRADE_PCT + 1e-6, (
                    f"{symbol} at equity {equity}: accepted a trade risking "
                    f"${risk:.2f}, above the ceiling"
                )

    def test_accepted_size_respects_the_risk_budget(self):
        equity, symbol = 50_000.0, "EURUSD"
        sl = INCIDENT_ATR[symbol] * 1.5
        size = calculate_position_size(equity, sl, symbol, score=75)

        pip = PIP_VALUES[symbol]
        actual_risk = (sl / pip) * get_pip_value_per_lot(symbol) * size
        # score 75 -> 0.5% budget; allow headroom for the 2dp lot rounding.
        assert actual_risk <= equity * 0.01

    # --- edge cases ---
    def test_zero_stop_distance_does_not_crash(self):
        assert calculate_position_size(10_000.0, 0.0, "EURUSD", score=70) >= 0.0

    def test_size_is_capped_at_the_symbol_maximum(self):
        from risk.position_sizing import MAX_LOT_PER_SYMBOL

        size = calculate_position_size(
            10_000_000.0, PIP_VALUES["EURUSD"] * 10, "EURUSD", score=95
        )
        assert size <= MAX_LOT_PER_SYMBOL["EURUSD"]

    def test_consecutive_losses_shrink_the_size(self):
        equity, sl = 50_000.0, INCIDENT_ATR["EURUSD"] * 1.5
        fresh = calculate_position_size(equity, sl, "EURUSD", consecutive_losses=0, score=75)
        bruised = calculate_position_size(equity, sl, "EURUSD", consecutive_losses=3, score=75)
        assert bruised < fresh


class TestLayerOneUsesTheFixedCap:
    """Layer 1 calls get_max_sl_distance, so the fix must reach it."""

    def test_initial_protection_no_longer_crushes_the_stop(self):
        from trade_management.layer1_initial_protection import compute_initial_protection

        protection = compute_initial_protection(
            "XAUUSD", atr=INCIDENT_ATR["XAUUSD"], regime="trending",
            account_equity=INCIDENT_EQUITY,
        )
        assert protection.sl_distance > INCIDENT_ATR["XAUUSD"] * 0.10

    def test_stop_and_target_stay_proportionate(self):
        """R:R was 1:310 during the incident."""
        from trade_management.layer1_initial_protection import compute_initial_protection

        protection = compute_initial_protection(
            "XAUUSD", atr=INCIDENT_ATR["XAUUSD"], regime="trending",
            account_equity=INCIDENT_EQUITY,
        )
        ratio = protection.tp_distance / protection.sl_distance
        assert ratio < 20, f"risk:reward is 1:{ratio:.0f}, stop is far too tight"
