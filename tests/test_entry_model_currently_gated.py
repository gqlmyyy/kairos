"""Pins the current, real state of the deployed entry model against the gate.

``models/entry/entry_model.json`` expects 65 features; the live inference path
sends 10 (see KNOWN_ISSUES.md item 0). This is a *known, accepted* state — the
bot is safe (nothing trades on an unverified prediction) but non-functional
(nothing trades at all) until the model is retrained or the entry_v2 feature
pipeline is finished.

This test exists so that state cannot change silently. If it starts failing,
one of two things happened, and both are worth knowing immediately:

- the model file was retrained/replaced (this test's assumptions are stale —
  update or delete it, and update KNOWN_ISSUES.md item 0), or
- the live feature vector was widened without updating this test — the more
  dangerous case, since it means someone touched entry-signal generation.

Skips (rather than fails) when the model artifact isn't present, so a fresh
checkout without ``models/`` populated doesn't report a false regression.
"""

from __future__ import annotations

import os

import pytest

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "entry", "entry_model.json",
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="models/entry/entry_model.json not present in this checkout",
)


def test_the_deployed_model_still_expects_65_features():
    """Documents the artifact's actual shape, so a retrain is visible here."""
    xgb = pytest.importorskip("xgboost")

    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    assert booster.num_features() == 65


def test_v1_path_sends_10_features_not_65():
    from analysis.models.xgboost_v2_inference import LIVE_FEATURE_NAMES

    assert len(LIVE_FEATURE_NAMES) == 10


def test_predict_with_v2_is_gated_shut_against_the_real_model():
    """The exact call main.py makes on the default (v1) entry path.

    This must return ML_GATE_INVALID, never a numeric p_win — a numeric
    result here would mean the contract check regressed and a live trade
    could be sized off a prediction the model was never trained to produce.
    """
    from analysis.models.xgboost_v2_inference import predict_with_v2

    result = predict_with_v2(
        rsi=55.0, atr=0.0012, macd=0.0, trend_strength=50.0, trend_score=50.0,
        momentum_score=50.0, volatility_score=50.0,
        market_regime="trending", direction="BUY",
    )

    assert result["available"] is False
    assert result["p_win"] is None
    # Blocked at load rather than at the contract check: the deployed artifact
    # has no metadata sidecar, so it is refused before a feature vector exists.
    # The guarantee under test is "no tradeable number", not which of the two
    # blocking statuses reports it.
    assert result["status"] in {"ML_GATE_INVALID", "ML_MODEL_MISSING"}
    assert "entry_model.json" in result["reason"]


def test_the_gate_result_would_be_rejected_by_trade_gate():
    """End-to-end: confirms trade_gate actually refuses on this output,
    not just that xgboost_v2_inference reports it correctly."""
    from analysis.models.xgboost_v2_inference import predict_with_v2
    from risk.trade_gate import GateDecision, TradeRequest, validate_trade_request

    ml_result = predict_with_v2(
        rsi=55.0, atr=0.0012, macd=0.0, trend_strength=50.0, trend_score=50.0,
        momentum_score=50.0, volatility_score=50.0,
        market_regime="trending", direction="BUY",
    )

    class _AllowingGovernor:
        def is_halted(self):
            return False

        def get_halt_reason(self):
            return ""

        def can_open_new_position(self, count):
            return True, ""

    request = TradeRequest(
        symbol="EURUSD", direction="BUY", final_score=80.0, ai_confidence=0.8,
        confidence=0.75, equity=5000.0, position_size=0.10, sl_distance=0.0030,
        tp_distance=0.0050, signal_is_valid=True,
        ml_available=ml_result["available"], ml_p_win=ml_result["p_win"],
        ml_threshold=0.55, ml_status=ml_result["status"],
    )
    gate = validate_trade_request(request, governor=_AllowingGovernor())

    assert gate.decision is GateDecision.REJECT
    assert "ml_unavailable" in gate.reason
