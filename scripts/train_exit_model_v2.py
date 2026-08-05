#!/usr/bin/env python3
"""Train an exit model against the canonical 12-feature schema.

Why this exists: the deployed models/exit/exit_model.json expects 13 features
while analysis/models/feature_schema.FEATURE_ORDER defines 12, and nothing in
the repository or its git history names the thirteenth. Rather than reverse
engineer a lost artifact, this trains a clean model on the schema the code
actually uses.

Everything goes through feature_schema.build_feature_vector so the training
vector is, by construction, the same vector inference builds.

The script refuses to save a model that does not clearly beat chance on a
held-out split. A model that cannot be validated must not reach shadow mode,
let alone production.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from analysis.models.feature_schema import FEATURE_ORDER, build_feature_vector

DB_PATH = "trading_bot_v3.db"
OUT_PATH = "models/exit/exit_model_v2.json"
REPORT_PATH = "models/exit/exit_model_v2_report.json"

# A model must clear these on the held-out split or it is not saved.
MIN_AUC = 0.60
MIN_SAMPLES = 100
MIN_MINORITY_CLASS = 15


def load_rows(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM execution_dataset "
                "WHERE actual_pnl IS NOT NULL AND actual_pnl != 0 "
                "ORDER BY dataset_created_at ASC"
            )
        ]
    finally:
        conn.close()
    return rows


def to_features(row: dict) -> dict:
    """Map a DB row onto the schema's feature names."""
    duration_raw = row.get("trade_duration")
    try:
        duration_hours = float(duration_raw or 0) / 60.0
    except (TypeError, ValueError):
        duration_hours = 0.0

    return {
        "mfe": row.get("mfe") or 0.0,
        "mae": row.get("mae") or 0.0,
        "entry_atr": row.get("expected_atr") or row.get("actual_atr") or 0.0,
        "entry_rsi": row.get("expected_rsi") or row.get("actual_rsi") or 50.0,
        "entry_adx": row.get("expected_adx") or 0.0,
        "market_regime": row.get("expected_market_regime"),
        "trade_duration": duration_hours,
        "spread": row.get("spread_at_entry") or row.get("expected_spread") or 0.0,
        "volume": row.get("expected_volume") or 1.0,
        "session": row.get("expected_session"),
        "trend_h1": row.get("expected_trend_strength") or 0.0,
        "trend_h4": row.get("expected_trend_strength") or 0.0,
    }


def make_label(row: dict) -> int:
    """1 = bad exit, 0 = good exit.

    Kept deliberately close to the previous trainer's definition so the target
    stays comparable, minus the branches that depended on an mfe/mae that the
    live path never populated.
    """
    actual_pnl = float(row.get("actual_pnl") or 0)
    expected_tp = float(row.get("expected_tp") or 0)
    exit_reason = str(row.get("exit_reason") or "").lower()

    if "take_profit" in exit_reason or exit_reason == "tp":
        if expected_tp > 0 and actual_pnl < expected_tp * 0.5:
            return 1
        if actual_pnl <= 0:
            return 1
    if actual_pnl <= 0:
        return 1
    return 0


def constant_features(matrix: np.ndarray) -> list:
    """Feature names whose value never varies — they carry no signal."""
    return [
        name
        for i, name in enumerate(FEATURE_ORDER)
        if float(np.std(matrix[:, i])) == 0.0
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument(
        "--force-save", action="store_true",
        help="save even if the quality gates fail (diagnostics only)",
    )
    args = parser.parse_args()

    rows = load_rows(args.db)
    print(f"[1] loaded {len(rows)} rows with a realised P&L")

    if len(rows) < MIN_SAMPLES:
        print(f"    FAIL: need at least {MIN_SAMPLES} samples")
        return 2

    X = np.asarray([build_feature_vector(to_features(r)) for r in rows], dtype=np.float32)
    y = np.asarray([make_label(r) for r in rows], dtype=np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    positives = int(y.sum())
    print(f"[2] label balance: bad_exit={positives} good_exit={len(y) - positives}")

    dead = constant_features(X)
    print(f"[3] constant (zero-variance) features: {dead or 'none'}")
    live = [f for f in FEATURE_ORDER if f not in dead]
    print(f"    features carrying signal: {len(live)}/{len(FEATURE_ORDER)} -> {live}")

    minority = min(positives, len(y) - positives)
    if minority < MIN_MINORITY_CLASS:
        print(f"    FAIL: minority class has {minority} samples, need {MIN_MINORITY_CLASS}")
        if not args.force_save:
            return 2

    # Time-ordered split: this is a time series, so a random split would leak
    # future information into training and inflate the score.
    split = int(len(X) * (1.0 - args.test_fraction))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"[4] time-ordered split: train={len(X_train)} test={len(X_test)}")

    if len(set(y_test.tolist())) < 2:
        print("    FAIL: the held-out split contains a single class; AUC is undefined")
        if not args.force_save:
            return 2

    from sklearn.metrics import accuracy_score, roc_auc_score
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("[5] trained")

    proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    acc = float(accuracy_score(y_test, (proba > 0.5).astype(int)))
    majority = max(y_test.mean(), 1 - y_test.mean())

    print(f"[6] held-out AUC={auc:.4f} accuracy={acc:.4f} "
          f"(majority-class baseline={majority:.4f})")
    print(f"    predict_proba: min={proba.min():.4f} max={proba.max():.4f} "
          f"mean={proba.mean():.4f} NaN={int(np.isnan(proba).sum())}")

    importance = dict(zip(FEATURE_ORDER, [float(v) for v in model.feature_importances_]))
    top = sorted(importance.items(), key=lambda kv: -kv[1])[:5]
    print(f"[7] top features: {top}")

    report = {
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_balance": {"bad_exit": positives, "good_exit": len(y) - positives},
        "feature_order": FEATURE_ORDER,
        "constant_features": dead,
        "auc": auc,
        "accuracy": acc,
        "majority_baseline": float(majority),
        "proba_stats": {
            "min": float(proba.min()), "max": float(proba.max()),
            "mean": float(proba.mean()), "nan_count": int(np.isnan(proba).sum()),
        },
        "feature_importance": importance,
        "passed_gates": bool(auc >= MIN_AUC and minority >= MIN_MINORITY_CLASS),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[8] report written to {REPORT_PATH}")

    if not report["passed_gates"] and not args.force_save:
        print(f"\nREFUSING TO SAVE: AUC {auc:.4f} < {MIN_AUC} — "
              f"this model is not distinguishable from chance.")
        return 1

    model.save_model(args.out)
    print(f"[9] saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
