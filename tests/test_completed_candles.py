"""H-02 regression: trading decisions must use completed candles only.

``copy_rates_from_pos(symbol, tf, 0, count)`` returns the currently forming bar
at the newest position. ``get_indicators`` then read ``closes[-1]``, so RSI/ATR/
MACD were computed from a bar that changes on every tick — irreproducible, and
not what a backtest on closed bars sees.

These tests prove the forming bar is dropped, and that candle-boundary
semantics hold around the edges.
"""

from __future__ import annotations

import pytest

from data.market import mt5_client


H1 = 3600
BASE = 1_786_100_400  # an exact H1 boundary


def make_bars(n, start=BASE, step=H1):
    """Structured-array-like list of bars, oldest -> newest."""
    return [
        {
            "time": start + i * step,
            "open": 100.0 + i,
            "high": 100.5 + i,
            "low": 99.5 + i,
            "close": 100.0 + i,
            "tick_volume": 10 + i,
        }
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def clean():
    mt5_client.clear_cache()
    yield
    mt5_client.clear_cache()


@pytest.fixture
def fake_mt5(monkeypatch):
    """Stub MT5 that records how many bars were requested."""
    state = {"requested": None, "bars": make_bars(11)}

    class Fake:
        TIMEFRAME_H1 = 16385

        def copy_rates_from_pos(self, symbol, tf, start, count):
            state["requested"] = count
            return state["bars"][-count:] if count <= len(state["bars"]) else state["bars"]

    monkeypatch.setattr(mt5_client, "mt5", Fake())
    monkeypatch.setattr(mt5_client, "ensure_session", lambda: True)
    monkeypatch.setattr(mt5_client, "ensure_symbol", lambda s: True)

    from contextlib import contextmanager

    @contextmanager
    def _noop():
        yield Fake()

    monkeypatch.setattr(mt5_client, "mt5_call", _noop)
    return state


class TestFormingCandleExcluded:
    def test_newest_returned_bar_is_not_the_forming_one(self, fake_mt5):
        all_bars = fake_mt5["bars"]
        newest_time = all_bars[-1]["time"]

        candles = mt5_client.get_candles("EURUSD", "H1", count=10)

        assert candles, "expected candles"
        assert candles[-1]["time"] != newest_time, "forming candle was not dropped"
        assert candles[-1]["time"] == all_bars[-2]["time"]

    def test_requests_one_extra_bar_to_preserve_count(self, fake_mt5):
        mt5_client.get_candles("EURUSD", "H1", count=10)
        assert fake_mt5["requested"] == 11

    def test_returned_count_matches_requested(self, fake_mt5):
        candles = mt5_client.get_candles("EURUSD", "H1", count=10)
        assert len(candles) == 10

    def test_indicators_do_not_see_the_forming_bar(self, fake_mt5, monkeypatch):
        """The forming bar's close must not influence the computed price."""
        bars = fake_mt5["bars"]
        # Make the forming bar wildly different; it must not show up.
        bars[-1]["close"] = 9_999.0
        bars[-1]["high"] = 10_000.0

        candles = mt5_client.get_candles("EURUSD", "H1", count=10)
        closes = [c["close"] for c in candles]
        assert 9_999.0 not in closes

    # --- edge cases ---
    def test_only_forming_bar_available_yields_nothing(self, fake_mt5):
        fake_mt5["bars"] = make_bars(1)
        assert mt5_client.get_candles("EURUSD", "H1", count=10) == []

    def test_two_bars_yield_one_completed(self, fake_mt5):
        fake_mt5["bars"] = make_bars(2)
        candles = mt5_client.get_candles("EURUSD", "H1", count=1)
        assert len(candles) == 1
        assert candles[0]["time"] == fake_mt5["bars"][0]["time"]

    def test_no_bars_yields_empty(self, fake_mt5):
        fake_mt5["bars"] = []
        assert mt5_client.get_candles("EURUSD", "H1", count=10) == []


class TestCandleBoundarySemantics:
    """Prove the M15/H1/H4 alignment stated in the remediation brief."""

    @pytest.mark.parametrize(
        "timeframe,seconds",
        [("M15", 900), ("H1", 3600), ("H4", 14400)],
    )
    def test_bars_since_counts_only_closed_bars(self, timeframe, seconds):
        from trade_management.layer1_intrabar import bars_since

        open_ts = BASE
        # One second before the first close -> zero completed bars.
        assert bars_since(open_ts, open_ts + seconds - 1, timeframe) == 0
        # Exactly at the boundary -> one completed bar.
        assert bars_since(open_ts, open_ts + seconds, timeframe) == 1
        # One second after -> still one.
        assert bars_since(open_ts, open_ts + seconds + 1, timeframe) == 1

    def test_m15_1245_scenario(self):
        """At 12:47 the usable M15 bar is the one that closed at 12:45."""
        from trade_management.layer1_intrabar import bars_since

        opened = BASE                     # 12:00
        now = BASE + 47 * 60              # 12:47
        # 12:00->12:15->12:30->12:45 = 3 completed 15m bars.
        assert bars_since(opened, now, "M15") == 3

    def test_h1_and_h4_alignment_at_the_same_instant(self):
        from trade_management.layer1_intrabar import bars_since

        opened = BASE
        now = BASE + 5 * H1 + 47 * 60     # 5h47m later
        assert bars_since(opened, now, "H1") == 5
        assert bars_since(opened, now, "H4") == 1

    def test_missing_timestamp_is_zero_not_a_guess(self):
        from trade_management.layer1_intrabar import bars_since

        assert bars_since(None, BASE, "H1") == 0
        assert bars_since(BASE, None, "H1") == 0

    def test_future_open_time_never_yields_negative_age(self):
        from trade_management.layer1_intrabar import bars_since

        assert bars_since(BASE + 10 * H1, BASE, "H1") == 0
