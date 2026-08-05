"""Parity check: XGBClassifier.predict_proba vs Booster.predict.

Layer 2's probability provider loads the exit model through the sklearn wrapper
so ``predict_proba`` is available. That is only safe if it returns exactly what
the raw Booster path returned before. This test proves it on real historical
rows from execution_dataset, not synthetic data.

Skipped when xgboost, the model file, or the database are unavailable.
"""

from __future__ import annotations

import os

import pytest

MODEL_PATH = "models/exit/exit_model.json"
DB_PATH = "trading_bot_v3.db"
TOLERANCE = 1e-6

xgb = pytest.importorskip("xgboost", reason="xgboost not installed")
np = pytest.importorskip("numpy", reason="numpy not installed")


def _historical_rows(limit: int = 200):
    """Real execution_dataset rows, mapped into exit-model feature dicts."""
    import sqlite3

    if not os.path.exists(DB_PATH):
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT * FROM execution_dataset ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()

    mapped = []
    for row in rows:
        mapped.append(
            {
                "mfe": row.get("mfe") or 0.0,
                "mae": row.get("mae") or 0.0,
                "entry_atr": row.get("expected_atr") or 0.0,
                "entry_rsi": row.get("expected_rsi") or 0.0,
                "entry_adx": 0.0,
                "market_regime": row.get("expected_market_regime"),
                "trade_duration": 0.0,
                "spread": row.get("expected_spread") or 0.0,
                "volume": row.get("expected_volume") or 0.0,
                "session": row.get("expected_session"),
                "trend_h1": 0.0,
                "trend_h4": 0.0,
            }
        )
    return mapped


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="exit model artifact missing")
def test_predict_proba_matches_booster_predict():
    from analysis.models.feature_schema import FEATURE_ORDER, build_feature_vector

    rows = _historical_rows()
    if not rows:
        pytest.skip("no historical execution_dataset rows available")

    matrix = np.asarray([build_feature_vector(r) for r in rows], dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)

    # --- path A: raw Booster (what the deleted adapter used) ---
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)

    # Gate: the artifact must match the code's feature schema before parity can
    # mean anything. A mismatch here is an artifact/schema problem, not a
    # predict_proba problem — but it must still block enabling the path.
    assert booster.num_features() == len(FEATURE_ORDER), (
        f"Exit model artifact expects {booster.num_features()} features but "
        f"feature_schema.FEATURE_ORDER defines {len(FEATURE_ORDER)}. "
        f"Parity cannot be verified and ML_EXIT_ENABLED must stay False until "
        f"the deployed model and the schema are re-aligned."
    )

    booster_probs = booster.predict(xgb.DMatrix(matrix, feature_names=FEATURE_ORDER))
    booster_probs = np.asarray(booster_probs, dtype=np.float64).reshape(-1)

    # --- path B: sklearn wrapper (what Layer 2 uses) ---
    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_PATH)
    sklearn_probs = np.asarray(clf.predict_proba(matrix), dtype=np.float64)[:, 1]

    max_diff = float(np.max(np.abs(booster_probs - sklearn_probs)))
    mean_diff = float(np.mean(np.abs(booster_probs - sklearn_probs)))

    print(
        f"\nparity over {len(rows)} historical rows: "
        f"max_diff={max_diff:.3e} mean_diff={mean_diff:.3e}"
    )
    assert max_diff < TOLERANCE, (
        f"predict_proba diverges from booster.predict by {max_diff:.3e} "
        f"(tolerance {TOLERANCE:.0e}) — do NOT enable the sklearn path"
    )
