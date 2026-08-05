from __future__ import annotations

from typing import Any, Dict

from analysis.models.xgboost_exit_model import predict_exit_probability


def test_predict_exit_probability_range_and_no_exception() -> None:
    features: Dict[str, Any] = {
        "mfe": 10.8,
        "mae": -16.8,
        "atr": 4.42e-05,
        "rsi": 0.5474,
        "trade_health": 50.0,
        # adapter may pass this even if schema doesn't use it
        "profit_decay_pct": 104.8,
        "time_open_hours": 0.0,
        "spread": 15.0,
        "news_impact": 0.65,
        "market_regime": "TRENDING",
        "volume": 0.6,
        "direction": "buy",
        # IMPORTANT: no session here to avoid string->float issues
    }

    prob = predict_exit_probability(features)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

