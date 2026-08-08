"""M-03: NaN and Inf must never reach the broker.

The defect these tests pin was not a missing check — it was a check that read
as correct and did the opposite of its job:

.. code-block:: python

    if direction == "BUY":
        if sl >= live_price:      # NaN >= x  is False
            raise ValueError(...)
        if tp <= live_price:      # NaN <= x  is False
            raise ValueError(...)

Comparisons against NaN are always False, so the one input that could not be
recovered from — a stop loss that is not a number — was the one input that
passed every branch. The order reached MT5, MT5 accepted it, and the position
went live unprotected.

``test_nan_defeats_naive_comparisons`` below encodes *why* the old shape failed,
so nobody reintroduces it while reading the new code as equivalent.
"""

from __future__ import annotations

import math

import pytest

from execution.order_validation import (
    OrderValidationError,
    ValidationResult,
    is_finite_number,
    validate_market_data,
    validate_order_inputs,
    validate_order_prices,
)

NAN = float("nan")
INF = float("inf")
NEG_INF = float("-inf")

BAD_NUMBERS = [NAN, INF, NEG_INF]


class TestTheOriginalDefect:
    def test_nan_defeats_naive_comparisons(self):
        """The property that made the old check useless."""
        price = 1.1000
        assert (NAN >= price) is False
        assert (NAN <= price) is False
        assert (NAN > price) is False
        assert (NAN < price) is False
        assert (NAN == NAN) is False

    @pytest.mark.parametrize("direction", ["BUY", "SELL"])
    def test_nan_stop_loss_is_rejected(self, direction):
        result = validate_order_prices(1.1000, NAN, 1.1100 if direction == "BUY" else 1.0900, direction)
        assert not result.ok
        assert "sl" in result.reason

    @pytest.mark.parametrize("direction", ["BUY", "SELL"])
    def test_nan_take_profit_is_rejected(self, direction):
        sl = 1.0900 if direction == "BUY" else 1.1100
        result = validate_order_prices(1.1000, sl, NAN, direction)
        assert not result.ok
        assert "tp" in result.reason

    def test_nan_live_price_is_rejected(self):
        result = validate_order_prices(NAN, 1.0900, 1.1100, "BUY")
        assert not result.ok
        assert "live_price" in result.reason

    def test_mt5_direct_wrapper_raises_on_nan(self):
        """The wrapper open_trade actually calls must still raise."""
        from execution.mt5_direct import _validate_sl_tp_order

        with pytest.raises(ValueError, match="sl"):
            _validate_sl_tp_order(1.1000, NAN, 1.1100, "BUY")

    def test_mt5_direct_wrapper_accepts_a_valid_order(self):
        from execution.mt5_direct import _validate_sl_tp_order

        assert _validate_sl_tp_order(1.1000, 1.0900, 1.1100, "BUY") is True


class TestIsFiniteNumber:
    @pytest.mark.parametrize("value", [0, 1, -1, 0.5, -0.5, 1e10, 1e-10, "1.5", "0"])
    def test_accepts_real_numbers(self, value):
        assert is_finite_number(value) is True

    @pytest.mark.parametrize("value", BAD_NUMBERS + [None, "", "abc", [], {}, object()])
    def test_rejects_everything_else(self, value):
        assert is_finite_number(value) is False

    @pytest.mark.parametrize("value", [True, False])
    def test_rejects_booleans(self, value):
        """True is numerically 1.0; a bool here is a wiring bug, not a price."""
        assert is_finite_number(value) is False


class TestValidateOrderInputs:
    GOOD = dict(symbol="EURUSD", direction="BUY", size=0.10,
                sl_distance=0.0030, tp_distance=0.0050)

    def test_a_normal_order_passes(self):
        assert validate_order_inputs(**self.GOOD).ok

    @pytest.mark.parametrize("field", ["size", "sl_distance", "tp_distance"])
    @pytest.mark.parametrize("bad", BAD_NUMBERS)
    def test_non_finite_numbers_are_rejected(self, field, bad):
        kwargs = dict(self.GOOD)
        kwargs[field] = bad
        result = validate_order_inputs(**kwargs)
        assert not result.ok
        assert field in result.reason

    @pytest.mark.parametrize("field", ["size", "sl_distance", "tp_distance"])
    def test_none_is_rejected(self, field):
        kwargs = dict(self.GOOD)
        kwargs[field] = None
        result = validate_order_inputs(**kwargs)
        assert not result.ok
        assert "missing" in result.reason

    @pytest.mark.parametrize("size", [0, 0.0, -0.01])
    def test_non_positive_size_is_rejected(self, size):
        result = validate_order_inputs(**{**self.GOOD, "size": size})
        assert not result.ok
        assert "size" in result.reason

    @pytest.mark.parametrize("sl", [0, 0.0, -0.001])
    def test_non_positive_stop_distance_is_rejected(self, sl):
        """A zero stop distance opens the position unprotected."""
        result = validate_order_inputs(**{**self.GOOD, "sl_distance": sl})
        assert not result.ok
        assert "sl_distance" in result.reason

    def test_zero_take_profit_distance_is_allowed(self):
        """Trailing-only profiles (trend, breakout) run with no fixed target."""
        assert validate_order_inputs(**{**self.GOOD, "tp_distance": 0.0}).ok

    def test_negative_take_profit_distance_is_rejected(self):
        result = validate_order_inputs(**{**self.GOOD, "tp_distance": -0.001})
        assert not result.ok

    @pytest.mark.parametrize("direction", ["", None, "LONG", "buy_now", 0, "SEL"])
    def test_unknown_direction_is_rejected(self, direction):
        result = validate_order_inputs(**{**self.GOOD, "direction": direction})
        assert not result.ok
        assert "direction" in result.reason

    @pytest.mark.parametrize("direction", ["BUY", "SELL", "buy", " sell "])
    def test_direction_casing_and_padding_tolerated(self, direction):
        kwargs = {**self.GOOD, "direction": direction}
        assert validate_order_inputs(**kwargs).ok

    @pytest.mark.parametrize("symbol", ["", "   ", None])
    def test_empty_symbol_is_rejected(self, symbol):
        result = validate_order_inputs(**{**self.GOOD, "symbol": symbol})
        assert not result.ok
        assert "symbol" in result.reason


class TestValidateOrderPrices:
    def test_valid_buy(self):
        assert validate_order_prices(1.1000, 1.0950, 1.1100, "BUY").ok

    def test_valid_sell(self):
        assert validate_order_prices(1.1000, 1.1050, 1.0900, "SELL").ok

    def test_buy_with_stop_above_price_is_rejected(self):
        result = validate_order_prices(1.1000, 1.1050, 1.1100, "BUY")
        assert not result.ok
        assert "sl" in result.reason

    def test_sell_with_stop_below_price_is_rejected(self):
        result = validate_order_prices(1.1000, 1.0950, 1.0900, "SELL")
        assert not result.ok
        assert "sl" in result.reason

    def test_buy_with_target_below_price_is_rejected(self):
        result = validate_order_prices(1.1000, 1.0950, 1.0900, "BUY")
        assert not result.ok
        assert "tp" in result.reason

    def test_stop_exactly_at_price_is_rejected(self):
        """Zero stop distance is not protection."""
        assert not validate_order_prices(1.1000, 1.1000, 1.1100, "BUY").ok
        assert not validate_order_prices(1.1000, 1.1000, 1.0900, "SELL").ok

    @pytest.mark.parametrize("direction", ["BUY", "SELL"])
    def test_zero_take_profit_is_mt5_for_no_target(self, direction):
        sl = 1.0950 if direction == "BUY" else 1.1050
        assert validate_order_prices(1.1000, sl, 0.0, direction).ok

    @pytest.mark.parametrize("direction", ["BUY", "SELL"])
    def test_zero_take_profit_can_be_disallowed(self, direction):
        sl = 1.0950 if direction == "BUY" else 1.1050
        result = validate_order_prices(1.1000, sl, 0.0, direction, allow_zero_tp=False)
        assert not result.ok

    @pytest.mark.parametrize("price", [0, -1.0])
    def test_non_positive_price_is_rejected(self, price):
        result = validate_order_prices(price, 1.0, 2.0, "BUY")
        assert not result.ok
        assert "live_price" in result.reason

    @pytest.mark.parametrize("bad", BAD_NUMBERS)
    @pytest.mark.parametrize("slot", [0, 1, 2])
    def test_any_non_finite_slot_is_rejected(self, bad, slot):
        args = [1.1000, 1.0950, 1.1100]
        args[slot] = bad
        assert not validate_order_prices(*args, "BUY").ok

    def test_gold_scale_prices(self):
        """XAUUSD moves in whole dollars, not pips — the rules are scale-free."""
        assert validate_order_prices(2650.00, 2580.00, 2760.00, "BUY").ok
        assert not validate_order_prices(2650.00, 2680.00, 2760.00, "BUY").ok


class TestValidateMarketData:
    def test_all_absent_is_valid(self):
        """Every field is optional; a caller validates only what it holds."""
        assert validate_market_data().ok

    def test_normal_readings_pass(self):
        assert validate_market_data(
            atr=0.0018, spread=0.8, equity=5000.0,
            risk_amount=25.0, probability=0.63, confidence=0.71,
        ).ok

    @pytest.mark.parametrize(
        "field", ["atr", "spread", "equity", "risk_amount", "probability", "confidence"]
    )
    @pytest.mark.parametrize("bad", BAD_NUMBERS)
    def test_non_finite_is_rejected(self, field, bad):
        result = validate_market_data(**{field: bad})
        assert not result.ok
        assert field in result.reason

    @pytest.mark.parametrize("atr", [0, 0.0, -0.001])
    def test_non_positive_atr_is_rejected(self, atr):
        """A zero ATR yields a zero stop distance — an unprotected position."""
        result = validate_market_data(atr=atr)
        assert not result.ok
        assert "atr" in result.reason

    def test_negative_equity_is_rejected(self):
        assert not validate_market_data(equity=-1.0).ok
        assert not validate_market_data(equity=0.0).ok

    def test_negative_spread_is_rejected(self):
        assert not validate_market_data(spread=-0.1).ok

    def test_zero_spread_is_allowed(self):
        """Some feeds report 0 on an illiquid tick; it is not by itself invalid."""
        assert validate_market_data(spread=0.0).ok

    @pytest.mark.parametrize("value", [-0.01, 1.01, 2.0, -1.0])
    @pytest.mark.parametrize("field", ["probability", "confidence"])
    def test_probability_outside_zero_to_one_is_rejected(self, field, value):
        result = validate_market_data(**{field: value})
        assert not result.ok
        assert field in result.reason

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("field", ["probability", "confidence"])
    def test_probability_boundaries_are_inclusive(self, field, value):
        assert validate_market_data(**{field: value}).ok


class TestValidationResult:
    def test_raise_if_invalid_raises(self):
        with pytest.raises(OrderValidationError, match="because"):
            ValidationResult(False, "because").raise_if_invalid()

    def test_raise_if_invalid_is_silent_when_valid(self):
        ValidationResult(True).raise_if_invalid()


class TestOpenTradeRejectsBeforeTouchingTheBroker:
    """The guard must run before a session or symbol is touched."""

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            (dict(size=NAN), "size"),
            (dict(size=0), "size"),
            (dict(sl_distance=NAN), "sl_distance"),
            (dict(sl_distance=0), "sl_distance"),
            (dict(tp_distance=INF), "tp_distance"),
            (dict(direction="LONG"), "direction"),
            (dict(symbol=""), "symbol"),
        ],
    )
    def test_bad_input_is_refused(self, kwargs, expected, monkeypatch):
        import execution.mt5_direct as md

        def _must_not_run(*args, **kw):
            raise AssertionError("broker was contacted despite invalid inputs")

        monkeypatch.setattr(md, "_ensure_mt5_initialized", _must_not_run)
        monkeypatch.setattr(md, "_ensure_symbol_selected", _must_not_run)

        call = dict(symbol="EURUSD", direction="BUY", size=0.10,
                    sl_distance=0.0030, tp_distance=0.0050, reason="test")
        call.update(kwargs)

        result = md.open_trade(**call)
        assert result["status"] == "error"
        assert expected in result["error"]
        assert result["order_id"] is None

    def test_the_guard_is_not_vacuous(self):
        """Valid inputs must get past it — otherwise everything 'passes'.

        MetaTrader5 is not importable off Windows, so a valid order stops at the
        next check instead. What matters is *which* check stopped it: reaching a
        broker-availability error proves the input guard let it through.
        """
        import execution.mt5_direct as md

        result = md.open_trade("EURUSD", "BUY", 0.10, 0.0030, 0.0050, "test")
        assert result["status"] == "error"
        assert "invalid order inputs" not in result["error"], (
            "the input guard rejected a valid order"
        )
