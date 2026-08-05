from __future__ import annotations

import time

import pytest

from execution.post_entry.xgboost_exit_model_adapter import XGBoostExitModelAdapter


def _base_snapshot() -> dict:
    return {
        "trade": {
            "symbol": "EURUSD",
            "time_open": time.time(),
            "spread": 15.0,
            "profit": 0.0,
            "order_id": "1",
            "volume": 0.6,
            "direction": "buy",
        },
        "expected_row": {
            "symbol": "EURUSD",
            "expected_session": "london",
            "expected_trend_h1": 0.0,
            "expected_trend_h4": 0.0,
            "expected_news_impact_score": 0.0,
            "p_win": 0.5,
            "mfe": 0.0,
            "mae": 0.0,
        },
        "market_regime": "TRENDING",
    }


# Patch model call so tests don't depend on the real XGBoost model.
@pytest.fixture(autouse=True)
def _patch_model(monkeypatch):
    import analysis.models.xgboost_exit_model as mod

    def fake_predict_exit_probability(features):
        return 0.4

    # adapter imported predict_exit_probability into its module scope,
    # so patch both the model module and the adapter's bound reference.
    monkeypatch.setattr(mod, "predict_exit_probability", fake_predict_exit_probability)
    monkeypatch.setattr(
        "execution.post_entry.xgboost_exit_model_adapter.predict_exit_probability",
        fake_predict_exit_probability,
    )


def test_predict_success_with_valid_quantdinger_data(monkeypatch):
    adapter = XGBoostExitModelAdapter()

    def fake_get_indicators_hybrid(symbol: str, timeframe: str = "4H"):
        if timeframe == "H4":
            return {"atr": 0.0008, "rsi": 55.0}
        if timeframe == "H1":
            return {"rsi": 55.0, "atr": 0.0008}
        return {"rsi": 55.0, "atr": 0.0008}

    monkeypatch.setattr(
        "execution.post_entry.xgboost_exit_model_adapter.get_indicators_hybrid",
        fake_get_indicators_hybrid,
    )

    monkeypatch.setattr(
        "execution.post_entry.xgboost_exit_model_adapter.get_candles",
        lambda symbol, timeframe="H4", count=50: [{"high": 1.1, "low": 1.0, "close": 1.05}] * 50,
    )

    monkeypatch.setattr(
        "execution.post_entry.xgboost_exit_model_adapter.calculate_adx",
        lambda candles, period=14: 20.0,
    )

    snap = _base_snapshot()
    snap["trade"]["time_open"] = time.time() - 60

    out = adapter.predict(snap, position_state=None)
    assert out["features_incomplete"] is False
    assert out["continue_probability"] is not None
    assert out["exit_probability"] is not None
    assert out["confidence"] is not None


def test_predict_fails_safely_when_quantdinger_unavailable(monkeypatch):
    adapter = XGBoostExitModelAdapter()

    def fake_get_indicators_hybrid(symbol: str, timeframe: str = "4H"):
        # adapter treats rsi==50.0 as placeholder -> must fail safely
        return {"rsi": 50.0, "atr": 0.0008}

    monkeypatch.setattr(
        "execution.post_entry.xgboost_exit_model_adapter.get_indicators_hybrid",
        fake_get_indicators_hybrid,
    )

    monkeypatch.setattr(
        "execution.post_entry.xgboost_exit_model_adapter.get_candles",
        lambda symbol, timeframe="H4", count=50: [{"high": 1.1, "low": 1.0, "close": 1.05}] * 50,
    )
    monkeypatch.setattr(
        "execution.post_entry.xgboost_exit_model_adapter.calculate_adx",
        lambda candles, period=14: 20.0,
    )

    import analysis.models.xgboost_exit_model as mod

    called = {"n": 0}

    def spy_predict_exit_probability(features):
        called["n"] += 1
        return 0.9

    monkeypatch.setattr(mod, "predict_exit_probability", spy_predict_exit_probability)
    monkeypatch.setattr(
        "execution.post_entry.xgboost_exit_model_adapter.predict_exit_probability",
        spy_predict_exit_probability,
    )

    snap = _base_snapshot()
    snap["trade"]["time_open"] = time.time() - 60
    snap["market_regime"] = "TRENDING"

    out = adapter.predict(snap, position_state=None)
    assert out["features_incomplete"] is True
    assert out["continue_probability"] is None
    assert out["exit_probability"] is None
    assert out["confidence"] is None
    assert called["n"] == 0


def test_adx_computed_from_quantdinger_candles(monkeypatch):
    adapter = XGBoostExitModelAdapter()

    def fake_get_indicators_hybrid(symbol: str, timeframe: str = "4H"):
        if timeframe == "H4":
            return {"atr": 0.0008, "rsi": 55.0}
        if timeframe == "H1":
            return {"rsi": 55.0, "atr": 0.0008}
        return {"rsi": 55.0, "atr": 0.0008}

    monkeypatch.setattr(
        "data.market.hybrid_client.get_indicators_hybrid",
        fake_get_indicators_hybrid,
    )

    expected_adx = 33.0
    dummy_candles = [{"high": 1.1, "low": 1.0, "close": 1.05}] * 50

    monkeypatch.setattr(
        "execution.post_entry.xgboost_exit_model_adapter.get_candles",
        lambda symbol, timeframe="H4", count=50: dummy_candles,
    )
    monkeypatch.setattr(
        "execution.post_entry.xgboost_exit_model_adapter.calculate_adx",
        lambda candles, period=14: expected_adx,
    )

    import analysis.models.xgboost_exit_model as mod

    captured = {"features": None}

    def capture_predict_exit_probability(features):
        captured["features"] = features
        return 0.5

    monkeypatch.setattr(mod, "predict_exit_probability", capture_predict_exit_probability)
    monkeypatch.setattr(
        "execution.post_entry.xgboost_exit_model_adapter.predict_exit_probability",
        capture_predict_exit_probability,
    )

    snap = _base_snapshot()
    snap["trade"]["time_open"] = time.time() - 60

    out = adapter.predict(snap, position_state=None)
    assert out["features_incomplete"] is False
    assert captured["features"] is not None
    assert float(captured["features"]["entry_adx"]) == expected_adx


def test_predict_fails_safely_when_market_regime_unknown(monkeypatch):
    adapter = XGBoostExitModelAdapter()

    def fake_get_indicators_hybrid(symbol: str, timeframe: str = "4H"):
        if timeframe == "H4":
            return {"atr": 0.0008, "rsi": 55.0}
        if timeframe == "H1":
            return {"rsi": 55.0, "atr": 0.0008}
        return {"rsi": 55.0, "atr": 0.0008}

    monkeypatch.setattr(
        "execution.post_entry.xgboost_exit_model_adapter.get_indicators_hybrid",
        fake_get_indicators_hybrid,
    )

    monkeypatch.setattr(
        "execution.post_entry.xgboost_exit_model_adapter.get_candles",
        lambda symbol, timeframe="H4", count=50: [{"high": 1.1, "low": 1.0, "close": 1.05}] * 50,
    )
    monkeypatch.setattr(
        "execution.post_entry.xgboost_exit_model_adapter.calculate_adx",
        lambda candles, period=14: 20.0,
    )

    snap = _base_snapshot()
    snap["trade"]["time_open"] = time.time() - 60
    snap["market_regime"] = "Unknown"

    out = adapter.predict(snap, position_state=None)
    assert out["features_incomplete"] is True
    assert out["continue_probability"] is None
    assert out["exit_probability"] is None
    assert out["confidence"] is None

