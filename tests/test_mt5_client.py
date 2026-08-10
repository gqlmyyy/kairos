"""Tests for the MT5 market data client.

The critical property here is that the indicator formulas are byte-for-byte the
ones the entry model was trained against. If someone "fixes" the RSI to use
Wilder smoothing or the MACD to use real EMAs, these tests fail — which is the
point. Correcting them is a deliberate change that requires retraining, tracked
in ROADMAP.md.
"""

from __future__ import annotations

import pytest

from data.market import mt5_client


def make_candles(closes, highs=None, lows=None):
    """Build candle dicts from a list of closes."""
    highs = highs or [c * 1.001 for c in closes]
    lows = lows or [c * 0.999 for c in closes]
    return [
        {
            "time": 1_700_000_000 + i * 3600,
            "open": closes[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
            "tick_volume": 100 + i,
        }
        for i in range(len(closes))
    ]


@pytest.fixture(autouse=True)
def clean_cache():
    mt5_client.clear_cache()
    yield
    mt5_client.clear_cache()


class TestIndicatorFormulas:
    """Pin the exact arithmetic inherited from the QuantDinger client."""

    def _patch_candles(self, monkeypatch, candles):
        monkeypatch.setattr(mt5_client, "get_candles", lambda *a, **k: candles)

    def test_rsi_uses_simple_average_not_wilder(self, monkeypatch):
        """A steadily rising series gives RSI 100 under the simple formula.

        Wilder smoothing over 100 candles would give a different number; this
        test exists to catch an accidental "improvement".
        """
        closes = [100.0 + i for i in range(60)]
        self._patch_candles(monkeypatch, make_candles(closes))
        result = mt5_client.get_indicators("EURUSD", "H1")
        # All 14 differences are gains -> avg_loss falls back to 0.001 -> RSI ~100
        assert result["rsi"] == pytest.approx(100.0, abs=0.1)

    def test_macd_is_sma_difference_not_ema(self, monkeypatch):
        """MACD here is mean(last 12) - mean(last 26), not an EMA difference."""
        closes = [100.0 + i for i in range(60)]
        self._patch_candles(monkeypatch, make_candles(closes))
        result = mt5_client.get_indicators("EURUSD", "H1")

        expected = (sum(closes[-12:]) / 12) - (sum(closes[-26:]) / 26)
        assert result["macd"] == pytest.approx(round(expected, 6))

    def test_atr_averages_last_14_true_ranges(self, monkeypatch):
        # Alternating series: every bar's true range is high-low = 2.0, and the
        # RSI denominator stays non-zero (see the flat-series test below).
        closes = [100.0 + (i % 2) * 0.5 for i in range(60)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        self._patch_candles(monkeypatch, make_candles(closes, highs, lows))
        result = mt5_client.get_indicators("EURUSD", "H1")
        assert result["atr"] == pytest.approx(2.0, abs=1e-6)

    def test_perfectly_flat_series_is_computed_not_faked(self, monkeypatch):
        """Regression: this used to raise and fall back to fixed constants.

        With every close identical, all 14 differences are zero. Because
        `diff > 0` routes an exact 0.0 to `losses`, the list was non-empty but
        summed to zero, the `if losses` guard missed it, and
        `rs = avg_gain / avg_loss` raised ZeroDivisionError. The except clause
        swallowed it and returned FALLBACK_INDICATORS — which is where the
        rsi=50.0 / atr=0.001 constants polluting the recorded dataset came from
        (KNOWN_ISSUES #3).

        Flat stretches are ordinary in real data: weekends, holidays, thin
        sessions, frozen feeds. The epsilon now applies to a zero *average*
        rather than an empty list, so the window is scored honestly: RSI 50
        (price did not move) and a real ATR from the actual high/low range.
        """
        closes = [100.0] * 60
        self._patch_candles(monkeypatch, make_candles(closes, [101.0] * 60, [99.0] * 60))
        result = mt5_client.get_indicators("EURUSD", "H1")

        assert result != mt5_client.FALLBACK_INDICATORS["EURUSD"], (
            "a flat window still degrades to fallback constants"
        )
        assert result["rsi"] == pytest.approx(50.0), "no movement should read neutral"
        assert result["atr"] == pytest.approx(2.0), "ATR must come from the real range"
        assert result["close"] == 100.0
        assert result["ma_trend"] == "sideways"

    @pytest.mark.parametrize(
        "closes,expected",
        [
            ([100.0 + i for i in range(60)], "strong uptrend"),
            ([100.0 - i for i in range(60)], "strong downtrend"),
        ],
    )
    def test_ma_trend_classification(self, monkeypatch, closes, expected):
        self._patch_candles(monkeypatch, make_candles(closes))
        assert mt5_client.get_indicators("EURUSD", "H1")["ma_trend"] == expected

    # --- edge cases ---
    def test_too_few_candles_returns_fallback(self, monkeypatch):
        self._patch_candles(monkeypatch, make_candles([100.0] * 5))
        result = mt5_client.get_indicators("EURUSD", "H1")
        assert result == mt5_client.FALLBACK_INDICATORS["EURUSD"]

    def test_no_candles_returns_fallback(self, monkeypatch):
        self._patch_candles(monkeypatch, [])
        result = mt5_client.get_indicators("XAUUSD", "H4")
        assert result["rsi"] == 50.0
        assert result["atr"] == mt5_client.FALLBACK_INDICATORS["XAUUSD"]["atr"]

    def test_unknown_symbol_gets_generic_fallback(self, monkeypatch):
        self._patch_candles(monkeypatch, [])
        result = mt5_client.get_indicators("NZDCHF", "H1")
        assert result["rsi"] == 50.0
        assert result["ma_trend"] == "sideways"


class TestTimeframeMapping:
    @pytest.mark.parametrize(
        "alias,expected_attr",
        [
            ("H1", "TIMEFRAME_H1"), ("1H", "TIMEFRAME_H1"),
            ("H4", "TIMEFRAME_H4"), ("4H", "TIMEFRAME_H4"),
            ("M15", "TIMEFRAME_M15"), ("15M", "TIMEFRAME_M15"),
            ("15m", "TIMEFRAME_M15"),
        ],
    )
    def test_all_aliases_resolve(self, alias, expected_attr):
        """One map replaces the two divergent ones the old code carried."""
        assert mt5_client._TF_NAMES.get(alias) == expected_attr or \
               mt5_client._TF_NAMES.get(alias.upper()) == expected_attr

    def test_unknown_timeframe_returns_none(self):
        assert mt5_client._mt5_timeframe("NOT_A_TF") is None


class TestDerivedGetters:
    def test_getters_read_from_indicators(self, monkeypatch):
        monkeypatch.setattr(
            mt5_client, "get_indicators",
            lambda *a, **k: {"rsi": 61.5, "atr": 0.0012, "macd": -0.0003, "close": 1.09},
        )
        assert mt5_client.get_rsi("EURUSD") == pytest.approx(61.5)
        assert mt5_client.get_atr("EURUSD") == pytest.approx(0.0012)
        assert mt5_client.get_macd("EURUSD") == pytest.approx(-0.0003)
        assert mt5_client.get_price("EURUSD") == pytest.approx(1.09)

    def test_zero_atr_falls_back_to_symbol_default(self, monkeypatch):
        monkeypatch.setattr(mt5_client, "get_indicators", lambda *a, **k: {"atr": 0})
        assert mt5_client.get_atr("XAUUSD") == mt5_client.FALLBACK_ATR["XAUUSD"]


class TestEquity:
    def test_equity_comes_from_account_info(self, monkeypatch):
        class Account:
            balance, equity, margin = 500.0, 512.5, 20.0

        monkeypatch.setattr(mt5_client, "get_account_info", lambda: Account())
        assert mt5_client.get_equity() == pytest.approx(512.5)

    def test_equity_zero_when_session_unavailable(self, monkeypatch):
        """Same contract as the old client: 0.0, so risk checks behave alike."""
        monkeypatch.setattr(mt5_client, "get_account_info", lambda: None)
        assert mt5_client.get_equity() == 0.0


class TestShims:
    def test_client_shim_reexports_the_old_names(self):
        from data.market import client

        for name in ("get_candles", "get_indicators", "get_atr", "get_equity", "set_token"):
            assert hasattr(client, name), f"client shim lost {name}"

    def test_hybrid_shim_reexports_the_old_names(self):
        from data.market import hybrid_client

        for name in ("get_indicators_hybrid", "get_atr_hybrid", "get_atr", "get_candles"):
            assert hasattr(hybrid_client, name), f"hybrid shim lost {name}"

    def test_hybrid_raises_when_no_data(self, monkeypatch):
        """Callers depend on the exception rather than a degraded reading."""
        from data.market import hybrid_client

        monkeypatch.setattr(hybrid_client, "get_indicators", lambda *a, **k: {})
        with pytest.raises(RuntimeError):
            hybrid_client.get_indicators_hybrid("EURUSD", "H1")

    def test_set_token_is_a_harmless_noop(self):
        from data.market import client

        assert client.set_token("anything") is None
