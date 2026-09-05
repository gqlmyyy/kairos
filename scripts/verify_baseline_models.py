from __future__ import annotations

"""scripts/verify_baseline_models.py

Final verification for the baseline XGBoost models.

Single source of truth: models/baseline/<SYMBOL>/<TF>/model.json — nothing
else is read. The nine artifacts (EURUSD, GBPUSD, XAUUSD x M15, H1, H4) are
loaded and executed, and every load and every output is checked:

Per model:
  1. File integrity        — SHA-256, size, JSON parses, learner section present.
  2. Load integrity        — xgboost.Booster() loads the native JSON.
  3. Metadata sanity       — objective, num_feature == len(feature_names),
                             tree count, base_score.
  4. Execution (algorithm) — booster.predict on deterministic seeded inputs:
                             random normal (512 x F), zeros (8 x F), ones (8 x F),
                             with the model's own feature names on the DMatrix.
  5. Output correctness    — finite, inside [0, 1] for binary:logistic,
                             bit-exact determinism across repeated predicts,
                             non-constant response on varied input.

Cross-model: within each timeframe, the three symbol feature-name lists must
be identical (the schema is per-timeframe by design, not global).

The report is printed and written to reports/baseline_model_verification.json.
Exit code 0 only when every check on every model passes. Deterministic (seed 42).
"""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np  # type: ignore
import xgboost as xgb  # type: ignore

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINE_ROOT = Path(REPO_ROOT) / "models" / "baseline"
REPORT_PATH = Path(REPO_ROOT) / "reports" / "baseline_model_verification.json"

SYMBOLS = ("EURUSD", "GBPUSD", "XAUUSD")
TIMEFRAMES = ("M15", "H1", "H4")

RANDOM_SEED = 42
N_RANDOM_SAMPLES = 512
N_EXTREME_SAMPLES = 8

PASS = "PASS"
FAIL = "FAIL"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _predict(booster: xgb.Booster, X: np.ndarray,
             feature_names: "List[str] | None") -> np.ndarray:
    """Run the model on X with its own feature names pinned to the DMatrix."""
    if feature_names:
        dmat = xgb.DMatrix(X, feature_names=feature_names)
    else:
        dmat = xgb.DMatrix(X)
    return booster.predict(dmat)


def verify_one(model_path: Path) -> Dict[str, Any]:
    """Load one baseline artifact, execute it, and check every output."""
    report: Dict[str, Any] = {
        "model": str(model_path.relative_to(REPO_ROOT)),
        "exists": model_path.is_file(),
        "checks": {},
    }
    if not report["exists"]:
        report["status"] = FAIL
        report["error"] = "model.json not found"
        return report

    report["size_bytes"] = model_path.stat().st_size
    report["sha256"] = sha256_of(model_path)

    # ---- 1. JSON integrity -------------------------------------------------
    try:
        with open(model_path, encoding="utf-8") as f:
            raw = json.load(f)
        learner = raw["learner"]
        lmp = learner["learner_model_param"]
        report["checks"]["json_parses"] = True
    except Exception as exc:  # noqa: BLE001 - report a corrupt artifact, do not raise
        report["status"] = FAIL
        report["error"] = f"json/load error: {exc}"
        report["checks"]["json_parses"] = False
        return report

    objective = learner.get("objective", {}).get("name", "?")
    num_feature = int(lmp.get("num_feature", -1))
    feature_names: List[str] = list(learner.get("feature_names") or [])
    base_score = lmp.get("base_score", "?")
    n_trees = len(learner.get("gradient_booster", {}).get("model", {}).get("trees", []))

    report.update(
        {
            "objective": objective,
            "num_feature": num_feature,
            "n_feature_names": len(feature_names),
            "n_trees": n_trees,
            "base_score": base_score,
            "first_features": feature_names[:8],
        }
    )

    checks = report["checks"]
    checks["metadata_consistent"] = num_feature > 0 and (
        not feature_names or len(feature_names) == num_feature
    )

    # ---- 2. xgboost load integrity ----------------------------------------
    try:
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        checks["booster_loads"] = True
    except Exception as exc:  # noqa: BLE001
        report["status"] = FAIL
        report["error"] = f"xgboost load error: {exc}"
        report["checks"]["booster_loads"] = False
        return report

    # ---- 3/4. Execute the algorithm on deterministic inputs ---------------
    rng = np.random.default_rng(RANDOM_SEED)
    names = feature_names if feature_names else None

    X_random = rng.normal(size=(N_RANDOM_SAMPLES, num_feature))
    X_zeros = np.zeros((N_EXTREME_SAMPLES, num_feature))
    X_ones = np.ones((N_EXTREME_SAMPLES, num_feature))

    try:
        p_random = _predict(booster, X_random, names)
        p_random_repeat = _predict(booster, X_random, names)
        p_zeros = _predict(booster, X_zeros, names)
        p_ones = _predict(booster, X_ones, names)
        checks["inference_runs"] = True
    except Exception as exc:  # noqa: BLE001
        report["status"] = FAIL
        report["error"] = f"inference error: {exc}"
        report["checks"]["inference_runs"] = False
        return report

    # ---- 5. Output correctness --------------------------------------------
    all_outputs = np.concatenate([p_random, p_zeros, p_ones])
    checks["outputs_finite"] = bool(np.isfinite(all_outputs).all())
    if objective == "binary:logistic":
        checks["outputs_in_unit_interval"] = bool(
            (all_outputs >= 0.0).all() and (all_outputs <= 1.0).all()
        )
    else:
        checks["outputs_in_unit_interval"] = None  # not applicable for this objective

    checks["deterministic"] = bool(np.array_equal(p_random, p_random_repeat))
    checks["responds_to_input"] = bool(p_random.std() > 0.0)

    report["output_stats"] = {
        "random_mean": float(p_random.mean()),
        "random_std": float(p_random.std()),
        "random_min": float(p_random.min()),
        "random_max": float(p_random.max()),
        "zeros_mean": float(p_zeros.mean()),
        "ones_mean": float(p_ones.mean()),
    }

    hard_checks = [
        checks["json_parses"],
        checks["booster_loads"],
        checks["metadata_consistent"],
        checks["inference_runs"],
        checks["outputs_finite"],
        checks["deterministic"],
        checks["responds_to_input"],
    ]
    if checks["outputs_in_unit_interval"] is not None:
        hard_checks.append(checks["outputs_in_unit_interval"])

    report["status"] = PASS if all(hard_checks) else FAIL
    return report


def main() -> int:
    print("=" * 72)
    print("BASELINE XGBOOST MODEL VERIFICATION")
    print(f"source (single): {BASELINE_ROOT}")
    print(f"seed: {RANDOM_SEED}   xgboost: {xgb.__version__}")
    print("=" * 72)

    if not BASELINE_ROOT.is_dir():
        print(f"FAIL: baseline root does not exist: {BASELINE_ROOT}")
        return 1

    models: List[Dict[str, Any]] = []
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            models.append(verify_one(BASELINE_ROOT / symbol / timeframe / "model.json"))

    # Cross-model consistency: within each timeframe, the three symbol
    # feature-name lists must be identical. The schema is per-timeframe by
    # design (each timeframe stacks features from the timeframes above it),
    # so a cross-timeframe equality check would be wrong.
    name_lists: Dict[str, Dict[str, List[str]]] = {}
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            path = BASELINE_ROOT / symbol / timeframe / "model.json"
            if path.is_file():
                with open(path, encoding="utf-8") as f:
                    name_lists.setdefault(timeframe, {})[symbol] = list(
                        json.load(f)["learner"].get("feature_names") or []
                    )

    per_timeframe: Dict[str, Any] = {}
    for timeframe in TIMEFRAMES:
        symbol_lists = name_lists.get(timeframe, {})
        identical = len(symbol_lists) == 3 and len(
            {json.dumps(v) for v in symbol_lists.values()}
        ) == 1
        per_timeframe[timeframe] = {
            "n_features": len(next(iter(symbol_lists.values()))) if symbol_lists else 0,
            "identical_across_symbols": identical,
        }
    cross_model_consistent = all(
        per_timeframe[tf]["identical_across_symbols"] for tf in TIMEFRAMES
    ) and len(per_timeframe) == 3

    # Informational: the stacked hierarchy M15 ⊇ H1 ⊇ H4, if it holds.
    try:
        h4_set = set(name_lists["H4"]["EURUSD"])
        h1_set = set(name_lists["H1"]["EURUSD"])
        m15_set = set(name_lists["M15"]["EURUSD"])
        hierarchy = {
            "H1_superset_of_H4": h4_set.issubset(h1_set),
            "M15_superset_of_H1": h1_set.issubset(m15_set),
        }
    except KeyError:
        hierarchy = None

    label = str(BASELINE_ROOT)
    print(f"\n{'model':<24}{'objective':<18}{'feat':>5}{'trees':>7}  "
          f"{'mean':>7}{'std':>7}  status")
    print("-" * 92)
    for m in models:
        stats = m.get("output_stats", {})
        print(
            f"{m['model'].replace('models' + os.sep + 'baseline' + os.sep, ''):<24}"
            f"{m.get('objective', '?'):<18}"
            f"{m.get('num_feature', '?'):>5}"
            f"{m.get('n_trees', '?'):>7}  "
            f"{stats.get('random_mean', float('nan')):>7.4f}"
            f"{stats.get('random_std', float('nan')):>7.4f}  "
            f"{m.get('status', FAIL)}"
        )
        if m.get("status") == FAIL:
            print(f"    -> {m.get('error')}")

    print("-" * 92)
    for timeframe in TIMEFRAMES:
        info = per_timeframe[timeframe]
        print(
            f"schema {timeframe}: {info['n_features']:>3} features — "
            f"identical across symbols: {'YES (3/3)' if info['identical_across_symbols'] else 'NO'}"
        )
    if hierarchy:
        print(f"stacked hierarchy M15 ⊇ H1 ⊇ H4: {hierarchy}")

    all_pass = all(m.get("status") == PASS for m in models) and cross_model_consistent
    document = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "xgboost_version": xgb.__version__,
        "source": label,
        "seed": RANDOM_SEED,
        "models": models,
        "per_timeframe_feature_schema": per_timeframe,
        "feature_schema_consistent": cross_model_consistent,
        "stacked_hierarchy_check": hierarchy,
        "all_pass": all_pass,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
    print(f"report written: {REPORT_PATH}")

    print("\n" + "=" * 72)
    if all_pass:
        print("FINAL VERDICT: ALL 9 MODELS VERIFIED — LOAD + EXECUTION + OUTPUTS OK")
    else:
        print("FINAL VERDICT: VERIFICATION FAILED — see errors above")
    print("=" * 72)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())


