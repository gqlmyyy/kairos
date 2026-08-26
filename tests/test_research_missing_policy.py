"""No fallback may hide a missing input, and a real zero is not missing.

The legacy path collapses everything into a number: `float(value or default)`
turns a genuine 0.0 into the default, and MISSING_DEFAULTS substitutes 50.0
for an unmeasured score. A model cannot tell a substituted value from an
observed one, so the substitution does not degrade a prediction — it
invalidates it, quietly, with a plausible probability attached.

These tests pin the opposite behaviour: four distinct states, zero is VALID,
and a required feature that is not VALID blocks the prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.research import availability as av

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "models" / "research" / "registry.json"
CANDLES = ROOT / "tests" / "fixtures" / "research" / "candles"


# --- the four states ---------------------------------------------------------

def test_zero_is_valid_not_missing():
    """The single most important line in this file."""
    state = av.classify("rsi", 0.0)
    assert state.state == av.VALID
    assert state.value == 0.0
    assert state.usable


@pytest.mark.parametrize("value", [0, 0.0, -0.0])
def test_every_spelling_of_zero_is_valid(value):
    assert av.classify("x", value).state == av.VALID


def test_nan_before_warmup_is_missing_not_invalid():
    s = av.classify("rsi", float("nan"), warmed_up=False)
    assert s.state == av.MISSING and s.value is None


def test_nan_after_warmup_is_invalid_not_missing():
    """A NaN on complete data is a data problem, not a schedule."""
    s = av.classify("rsi", float("nan"), warmed_up=True)
    assert s.state == av.INVALID


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_infinity_is_invalid(value):
    assert av.classify("x", value).state == av.INVALID


def test_a_column_the_source_cannot_produce_is_unavailable():
    s = av.classify("spread_relative", None, available=False)
    assert s.state == av.UNAVAILABLE and s.value is None


def test_unavailable_outranks_everything_else():
    """There is no value to judge, so no other verdict applies."""
    s = av.classify("spread_relative", 1.23, available=False)
    assert s.state == av.UNAVAILABLE and s.value is None


def test_none_is_missing_not_zero():
    s = av.classify("x", None)
    assert s.state == av.MISSING and s.value is None


def test_a_non_numeric_value_is_invalid_not_coerced():
    assert av.classify("x", "50").state == av.INVALID


def test_vector_is_unusable_if_any_single_feature_is():
    state = av.build_vector_state(["a", "b", "c"], [1.0, float("nan"), 0.0])
    assert not state.usable
    assert [s.name for s in state.offenders()] == ["b"]
    assert state.states[2].state == av.VALID, "a zero must survive alongside a NaN"


def test_summary_names_what_is_wrong():
    state = av.build_vector_state(["a", "b"], [1.0, None], unavailable=["b"])
    assert "b=UNAVAILABLE" in state.summary()


# --- end to end through the model -------------------------------------------

pytestmark_models = pytest.mark.skipif(
    not REGISTRY.exists(), reason="research models not imported in this checkout")


@pytestmark_models
def test_a_source_without_spread_refuses_to_predict_rather_than_inventing_it():
    """KAIROS's stored candles carry no spread; every research model needs it.

    The correct outcome is a refusal naming the unavailable features — not a
    prediction computed from a fabricated spread of zero, which would look
    entirely normal and be entirely wrong.
    """
    from analysis.research import candles as cd
    from analysis.research import replay as rp

    source = cd.KairosHistoricalSource(ROOT / "data" / "historical")
    if not source.available("XAUUSD", "H1"):
        pytest.skip("no stored historical candles in this checkout")

    result = rp.replay("XAUUSD", "H1", source, start="2024-06-01", limit=3,
                       registry_path=REGISTRY)
    assert result.rows_scored == 0
    assert set(result.status_counts) == {"FEATURE_UNAVAILABLE"}
    assert "spread" in result.unavailable_columns
    assert "spread_relative" in result.unavailable_features
    assert result.predictions["p_win"].isna().all(), (
        "an unavailable feature must yield no probability at all")


@pytestmark_models
def test_an_unavailable_feature_blocks_even_when_every_other_feature_is_fine():
    from analysis.research import candles as cd
    from analysis.research import engine as E
    from analysis.research import inference as inf
    from analysis.research.model_loader import load_model

    model = load_model("XAUUSD", "H1", registry_path=REGISTRY)
    source = cd.JsonCandleSource(CANDLES)
    stack = cd.load_stack(source, "XAUUSD", ["H1", "H4"])
    frame = E.build_feature_frame("XAUUSD", "H1", stack, ["H4"])
    row = frame.iloc[-1].to_dict()

    ok = inf.predict_row(model, row, entry_direction="long")
    assert ok.status == "OK" and ok.p_win is not None

    blocked = inf.predict_row(model, row, entry_direction="long",
                              unavailable_columns=["spread"])
    assert blocked.status == "FEATURE_UNAVAILABLE"
    assert blocked.p_win is None
    assert "spread_relative" in blocked.reason


@pytestmark_models
def test_a_single_nan_feature_blocks_the_prediction():
    from analysis.research import candles as cd
    from analysis.research import engine as E
    from analysis.research import inference as inf
    from analysis.research.model_loader import load_model

    model = load_model("XAUUSD", "H1", registry_path=REGISTRY)
    source = cd.JsonCandleSource(CANDLES)
    frame = E.build_feature_frame("XAUUSD", "H1",
                                  cd.load_stack(source, "XAUUSD", ["H1", "H4"]), ["H4"])
    row = frame.iloc[-1].to_dict()
    row["rsi"] = float("nan")

    pred = inf.predict_row(model, row, entry_direction="long")
    assert pred.status == "FEATURE_INVALID" and pred.p_win is None
    assert "rsi" in pred.reason


@pytestmark_models
def test_a_zero_feature_does_not_block_the_prediction():
    """The complement of the test above: a real zero must still be servable."""
    from analysis.research import candles as cd
    from analysis.research import engine as E
    from analysis.research import inference as inf
    from analysis.research.model_loader import load_model

    model = load_model("XAUUSD", "H1", registry_path=REGISTRY)
    source = cd.JsonCandleSource(CANDLES)
    frame = E.build_feature_frame("XAUUSD", "H1",
                                  cd.load_stack(source, "XAUUSD", ["H1", "H4"]), ["H4"])
    row = frame.iloc[-1].to_dict()
    row["rsi"] = 0.0
    row["adx"] = 0.0

    pred = inf.predict_row(model, row, entry_direction="long")
    assert pred.status == "OK", pred.reason
    assert pred.p_win is not None


@pytestmark_models
def test_a_missing_entry_direction_is_refused_not_defaulted_to_buy():
    """`target=1` means something different per side, so there is no safe default."""
    from analysis.research import candles as cd
    from analysis.research import engine as E
    from analysis.research import inference as inf
    from analysis.research.model_loader import load_model

    model = load_model("XAUUSD", "H1", registry_path=REGISTRY)
    if "entry_direction" not in model.feature_names:
        pytest.skip("this model's feature set carries no entry_direction")

    source = cd.JsonCandleSource(CANDLES)
    frame = E.build_feature_frame("XAUUSD", "H1",
                                  cd.load_stack(source, "XAUUSD", ["H1", "H4"]), ["H4"])
    pred = inf.predict_row(model, frame.iloc[-1].to_dict(), entry_direction=None)
    assert pred.status == "MODEL_NOT_COMPATIBLE"
    assert pred.p_win is None


def test_an_unknown_direction_raises_rather_than_falling_back():
    from analysis.research.inference import FeatureVectorError, encode_entry_direction

    assert encode_entry_direction("BUY") == 1.0
    assert encode_entry_direction("SELL") == -1.0
    with pytest.raises(FeatureVectorError, match="unrecognised"):
        encode_entry_direction("sideways")
    with pytest.raises(FeatureVectorError):
        encode_entry_direction(0.0)


def test_an_unusable_vector_cannot_be_materialised_into_an_array():
    """There must be no way to get numbers out of a vector that has none."""
    from analysis.research.inference import FeatureVector, FeatureVectorError

    state = av.build_vector_state(["a", "b"], [1.0, None])
    vector = FeatureVector(names=("a", "b"), state=state)
    with pytest.raises(FeatureVectorError, match="unusable"):
        vector.as_array()


def test_spread_relative_keeps_a_real_zero_spread_as_an_observation():
    """A literal zero spread is common on these feeds and is real data.

    Under UNIT_WHEN_ALSO_ZERO a zero spread against a zero median is a ratio
    of exactly 1.0. A positive spread against a zero median stays NaN, because
    that ratio is genuinely unbounded and must not be invented.
    """
    from analysis.research.price_action import spread_relative

    zeros = pd.Series([0.0] * 250)
    out = spread_relative(zeros, 200)
    assert out.iloc[-1] == 1.0

    spiky = pd.Series([0.0] * 249 + [3.0])
    assert np.isnan(spread_relative(spiky, 200).iloc[-1])

    strict = spread_relative(zeros, 200, zero_median_policy="STRICT")
    assert np.isnan(strict.iloc[-1])


# --- spread availability is detected, not assumed ----------------------------

def test_kairos_source_detects_spread_from_the_files(tmp_path):
    """`fetch_training_candles` DOES record MT5's per-bar spread.

    A snapshot without it is simply older than that change, so the source
    probes the files instead of hard-coding an answer — otherwise a refreshed
    snapshot would go on being refused for no reason.
    """
    import json as _json

    from analysis.research.candles import KairosHistoricalSource

    def write(root, with_spread):
        root.mkdir(parents=True, exist_ok=True)
        for tf, minutes in (("H1", 3600), ("H4", 14400)):
            rows = []
            for i in range(5):
                row = {"t": 1700000000.0 + i * minutes, "open": 1.0, "high": 1.2,
                       "low": 0.9, "close": 1.1, "volume": 10.0}
                if with_spread:
                    row["spread"] = 2.0
                rows.append(row)
            (root / f"EURUSD_{tf}.json").write_text(_json.dumps(rows), encoding="utf-8")

    old = tmp_path / "old"
    write(old, with_spread=False)
    assert KairosHistoricalSource(old).provides_spread is False
    assert KairosHistoricalSource(old).unavailable_columns() == ["spread"]

    fresh = tmp_path / "fresh"
    write(fresh, with_spread=True)
    source = KairosHistoricalSource(fresh)
    assert source.provides_spread is True
    assert source.unavailable_columns() == []
    assert "spread" in source.load("EURUSD", "H1").columns


def test_a_partially_spread_bearing_snapshot_is_treated_as_absent(tmp_path):
    """A mixed stack is worse than no spread at all.

    If H1 carried spread and H4 did not, the entry row's own `spread_relative`
    would exist while `H4_spread_relative` did not — an input assembled from
    two different contracts. Refusing the whole source is the safe reading.
    """
    import json as _json

    from analysis.research.candles import KairosHistoricalSource

    root = tmp_path / "mixed"
    root.mkdir()
    base = {"t": 1700000000.0, "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1}
    (root / "EURUSD_H1.json").write_text(
        _json.dumps([{**base, "spread": 2.0}]), encoding="utf-8")
    (root / "EURUSD_H4.json").write_text(_json.dumps([base]), encoding="utf-8")

    assert KairosHistoricalSource(root).provides_spread is False
