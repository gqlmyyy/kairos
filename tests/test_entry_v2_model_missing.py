"""Regression: a missing entry_v2 booster must return ML_MODEL_MISSING, not crash.

Found by ruff (F821 undefined-name) during the M-04 static-analysis pass:

.. code-block:: python

    if booster is None:
        return _as_dict(contract.model_missing(
            f"entry_v2 booster not loadable from {artifacts.model_path}"
        ))

``artifacts`` is a local variable inside ``_load_everything()``; nothing by
that name exists in ``predict_with_entry_v2``'s scope. The exact case that
should degrade gracefully — no trained model present — raised a bare
``NameError`` instead. trade_gate's ML gate (risk/trade_gate.py) rejects on
``ml_available=False``, which this path is supposed to produce; a NameError
propagating out is a worse failure than the one being guarded against, not a
safe substitute for it.
"""

from __future__ import annotations

import analysis.entry_v2.inference as inference


def test_missing_booster_returns_model_missing_instead_of_raising(monkeypatch):
    monkeypatch.setitem(inference._cached, "loaded", True)
    monkeypatch.setitem(inference._cached, "booster", None)
    monkeypatch.setitem(inference._cached, "artifacts", None)

    result = inference.predict_with_entry_v2(
        rsi=55.0, atr=0.0018, macd=0.0002, trend_strength=60.0,
        trend_score=65.0, momentum_score=55.0, volatility_score=40.0,
        market_regime="trending", direction="BUY",
    )

    assert result["available"] is False
    assert result["p_win"] is None
    assert result["status"] == "ML_MODEL_MISSING"
    assert "unresolved" in result["reason"] or "not loadable" in result["reason"]


def test_missing_booster_with_a_resolved_path_names_it_in_the_reason(monkeypatch):
    """When artifacts *did* resolve but the file just isn't there, the
    message should say where it looked."""
    class _FakeArtifacts:
        model_path = "/fake/models/entry_v2/entry_model.json"

    monkeypatch.setitem(inference._cached, "loaded", True)
    monkeypatch.setitem(inference._cached, "booster", None)
    monkeypatch.setitem(inference._cached, "artifacts", _FakeArtifacts())

    result = inference.predict_with_entry_v2(
        rsi=55.0, atr=0.0018, macd=0.0002, trend_strength=60.0,
        trend_score=65.0, momentum_score=55.0, volatility_score=40.0,
        market_regime="trending", direction="BUY",
    )

    assert result["status"] == "ML_MODEL_MISSING"
    assert "/fake/models/entry_v2/entry_model.json" in result["reason"]
