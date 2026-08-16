"""Cross-asset features: point-in-time availability, and no network required.

Everything here runs against injected fixtures — no real yfinance call. The
one contract worth pinning hard is the day-boundary rule: a daily bar dated D
must not be visible before (D+1) 00:00 UTC, however close the decision.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from analysis.features import cross_asset_features as ca

DAY = 86400


def daily_series(n=60, start_date="2024-01-01", start_price=100.0, step=0.1):
    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp()
    out, price = [], start_price
    for i in range(n):
        price += step * ((-1) ** i)
        out.append({"t": start + i * DAY, "close": price})
    return out


class TestAvailabilityRule:
    def test_a_bar_is_not_visible_on_its_own_date(self):
        series = daily_series()
        bar = series[10]
        at = bar["t"] + 3600 * 5  # same UTC day, five hours later
        assert ca._last_known(series, at) == series[9]["close"]

    def test_a_bar_is_visible_the_next_utc_day(self):
        series = daily_series()
        bar = series[10]
        at = bar["t"] + DAY
        assert ca._last_known(series, at) == bar["close"]

    def test_one_second_before_the_next_day_it_is_still_not_visible(self):
        series = daily_series()
        bar = series[10]
        at = bar["t"] + DAY - 1
        assert ca._last_known(series, at) == series[9]["close"]

    def test_nothing_is_visible_before_the_first_bar_closes(self):
        series = daily_series()
        assert ca._last_known(series, series[0]["t"]) is None


class TestBuildFeatures:
    def test_none_when_no_relevant_ticker_has_data(self):
        assert ca.build_cross_asset_features(
            {}, symbol="EURUSD", decision_timestamp=daily_series()[30]["t"]) is None

    def test_eurusd_only_uses_its_relevant_tickers(self):
        data = {"dxy": daily_series(), "silver": daily_series(start_price=25.0)}
        at = daily_series()[40]["t"] + DAY
        result = ca.build_cross_asset_features(data, symbol="EURUSD", decision_timestamp=at)
        assert any(k.startswith("dxy_") for k in result)
        assert not any(k.startswith("silver_") for k in result), (
            "EURUSD pulled in silver, which is not in its relevant-ticker list")

    def test_xauusd_can_use_silver_and_oil(self):
        data = {"silver": daily_series(start_price=25.0), "oil": daily_series(start_price=80.0)}
        at = daily_series()[40]["t"] + DAY
        result = ca.build_cross_asset_features(data, symbol="XAUUSD", decision_timestamp=at)
        assert any(k.startswith("silver_") for k in result)
        assert any(k.startswith("oil_") for k in result)

    def test_too_little_history_yields_no_features_for_that_ticker(self):
        data = {"dxy": daily_series()}
        at = daily_series()[2]["t"] + DAY  # only 3 bars visible
        result = ca.build_cross_asset_features(data, symbol="EURUSD", decision_timestamp=at)
        assert result is None


class TestNoLookAhead:
    def test_mutating_a_future_bar_does_not_change_the_feature(self):
        series = daily_series(n=60)
        cut = 40
        at = series[cut]["t"] + DAY
        data = {"dxy": series}
        before = ca.build_cross_asset_features(data, symbol="EURUSD", decision_timestamp=at)

        tampered = [dict(r) for r in series]
        for j in range(cut + 1, len(tampered)):
            tampered[j]["close"] *= 100.0
        after = ca.build_cross_asset_features(
            {"dxy": tampered}, symbol="EURUSD", decision_timestamp=at)
        assert before == after

    def test_mutating_a_past_bar_does_change_the_feature(self):
        series = daily_series(n=60)
        cut = 40
        at = series[cut]["t"] + DAY
        before = ca.build_cross_asset_features(
            {"dxy": series}, symbol="EURUSD", decision_timestamp=at)

        tampered = [dict(r) for r in series]
        for j in range(cut - 5, cut):
            tampered[j]["close"] *= 100.0
        after = ca.build_cross_asset_features(
            {"dxy": tampered}, symbol="EURUSD", decision_timestamp=at)
        assert before != after


class TestFetchNeverFabricates:
    def test_fetch_daily_series_returns_none_without_yfinance(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "yfinance":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        assert ca.fetch_daily_series("DX-Y.NYB", "2024-01-01", "2024-02-01") is None

    def test_fetch_all_reports_unavailable_rather_than_raising(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "yfinance":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        data, report = ca.fetch_all("2024-01-01", "2024-02-01")
        assert data == {}
        assert all("NOT AVAILABLE" in s for s in report.status.values())
