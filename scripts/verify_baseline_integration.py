from __future__ import annotations

"""scripts/verify_baseline_integration.py

Final verification of the baseline entry-model integration.

`models/baseline/` is the only source of trained entry models. This script
proves the five properties the integration must satisfy, plus one gold check:

  1. Existence & load — all nine artifacts load via the gate's own loader.
  2. Mapping — the (symbol, timeframe) -> artifact mapping is exact: the
     booster served for each combo was loaded from
     models/baseline/<SYMBOL>/<TIMEFRAME>/model.json and carries that
     timeframe's schema fingerprint (H4=66, H1=136, M15=203 features).
  3. xgboost_p_win — real predictions from live MT5 candles through the
     vendored training feature pipeline, for all nine combos, both sides.
  4. No ML_MODEL_MISSING — every combo returns available=True, status=OK.
  5. Source isolation — every Booster.load_model call observed during the
     whole exercise resolves under models/baseline; the gate module's source
     contains no reference to models/entry or any legacy/research path.
  6. Gold parity — the vendored feature pipeline reproduces the training
     repository's own dataset rows (parquet) to ~machine precision.

Writes reports/baseline_integration_verification.json. Exit 0 only when all
parts pass. Parts 1/2/5/6 are offline; part 3 needs MT5 running.
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import xgboost as xgb  # noqa: E402

REPORT_PATH = REPO_ROOT / "reports" / "baseline_integration_verification.json"
TRAINING_REPO = Path(r"F:\files (2)\kibo\h4\xgbooost")  # parity data only, never models

SYMBOLS = ("EURUSD", "GBPUSD", "XAUUSD")
TIMEFRAMES = ("M15", "H1", "H4")
EXPECTED_SCHEMA_COUNT = {"H4": 66, "H1": 136, "M15": 203}
PARITY_TOLERANCE = 1e-6

PASS = "PASS"
FAIL = "FAIL"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Instrumentation: record every model file xgboost actually opens.
# ---------------------------------------------------------------------------
_loaded_model_paths: List[str] = []
_original_load_model = xgb.Booster.load_model


def _spying_load_model(self, *args, **kwargs):
    where = str(args[0]) if args else str(kwargs)
    _loaded_model_paths.append(where)
    return _original_load_model(self, *args, **kwargs)


xgb.Booster.load_model = _spying_load_model


def _under_baseline(path: str) -> bool:
    try:
        Path(path).resolve().relative_to((REPO_ROOT / "models" / "baseline").resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Part 1 + 2: existence, load, mapping identity
# ---------------------------------------------------------------------------

def check_load_and_mapping() -> Dict[str, Any]:
    from analysis.baseline import gate

    per_model = []
    ok = True
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            entry = {"model": f"{symbol}/{timeframe}"}
            path = gate.model_path(symbol, timeframe)
            entry["path"] = str(path)
            entry["exists"] = path.is_file()
            entry["sha256"] = _sha256(path) if entry["exists"] else None
            try:
                booster = gate.load_model(symbol, timeframe)
                names = gate._model_names[(symbol, timeframe)]
                entry["loads"] = True
                entry["num_feature"] = int(booster.num_features())
                entry["schema_fingerprint_ok"] = (
                    entry["num_feature"] == EXPECTED_SCHEMA_COUNT[timeframe]
                    and len(names) == entry["num_feature"])
                entry["objective"] = "binary:logistic"
            except Exception as exc:  # noqa: BLE001
                entry["loads"] = False
                entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["ok"] = bool(entry["exists"] and entry.get("loads")
                               and entry.get("schema_fingerprint_ok"))
            ok = ok and entry["ok"]
            per_model.append(entry)
    return {"ok": ok, "models": per_model}


def check_mapping_via_spy() -> Dict[str, Any]:
    """Re-load every combo with the load spy active and confirm the file each
    booster came from is exactly its own models/baseline/<S>/<TF>/model.json."""
    from analysis.baseline import gate

    gate._models.clear()  # force real disk loads through the spy
    gate._model_names.clear()
    expected = {str(gate.model_path(s, t)) for s in SYMBOLS for t in TIMEFRAMES}
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            gate.load_model(symbol, timeframe)

    recorded = list(_loaded_model_paths)
    all_under_baseline = all(_under_baseline(p) for p in recorded)
    exact_mapping = set(recorded) == expected
    return {
        "ok": bool(all_under_baseline and exact_mapping),
        "load_calls_observed": len(recorded),
        "all_under_models_baseline": all_under_baseline,
        "exactly_the_nine_expected_files": exact_mapping,
        "recorded_paths": sorted(set(recorded)),
    }


# ---------------------------------------------------------------------------
# Part 3 + 4: real predictions on live candles, no ML_MODEL_MISSING
# ---------------------------------------------------------------------------

def check_predictions() -> Dict[str, Any]:
    from analysis.baseline import gate

    rows = []
    ok = True
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            buy = gate.predict_entry(symbol, timeframe, "BUY")
            sell = gate.predict_entry(symbol, timeframe, "SELL")
            row = {
                "model": f"{symbol}/{timeframe}",
                "buy_p_win": buy["p_win"], "buy_status": buy["status"],
                "sell_p_win": sell["p_win"], "sell_status": sell["status"],
                "direction_sensitive": None,
            }
            row["ok"] = bool(
                buy["available"] and sell["available"]
                and buy["status"] == "OK" and sell["status"] == "OK"
                and buy["p_win"] is not None and sell["p_win"] is not None
                and 0.0 <= buy["p_win"] <= 1.0 and 0.0 <= sell["p_win"] <= 1.0)
            if row["ok"]:
                row["direction_sensitive"] = abs(buy["p_win"] - sell["p_win"]) > 1e-9
            ok = ok and row["ok"]
            rows.append(row)
    return {"ok": ok, "predictions": rows}


def check_source_isolation() -> Dict[str, Any]:
    """No code path in the integration may reference any other model store."""
    gate_src = (REPO_ROOT / "analysis" / "baseline" / "gate.py").read_text(encoding="utf-8")
    init_src = (REPO_ROOT / "analysis" / "baseline" / "__init__.py").read_text(encoding="utf-8")
    banned_fragments = [
        "models/entry", "models\\entry", "entry_model.json",
        "entry_v2", "xgboost_v2_inference", "model_registry",
        "live_gate", "models/research",
    ]
    hits = [b for b in banned_fragments if b in gate_src or b in init_src]
    runtime = {
        "all_loads_under_models_baseline": all(_under_baseline(p) for p in _loaded_model_paths),
        "load_calls_outside_baseline": [p for p in _loaded_model_paths
                                        if not _under_baseline(p)],
    }
    return {"ok": not hits and runtime["all_loads_under_models_baseline"],
            "banned_reference_hits": hits, "runtime": runtime}


def check_gold_parity() -> Dict[str, Any]:
    """The vendored pipeline must reproduce the training repo's own dataset
    rows. Reads the training repo's parquet + raw CSVs as VERIFICATION DATA
    only -- models still come from models/baseline exclusively."""
    import pandas as pd

    from analysis.baseline import gate
    from src.config.loader import load_config
    from src.features.live import LiveFeaturePipeline

    cfg = load_config(gate.CONFIG_DIR)
    results = []
    ok = True
    symbol = "EURUSD"
    for timeframe, ctx in (("H4", []), ("H1", ["H4"]), ("M15", ["H1", "H4"])):
        pq_path = (TRAINING_REPO / "data" / "reports" / "training_dataset"
                   / symbol / timeframe / "validation.parquet")
        if not pq_path.is_file():
            results.append({"schema": timeframe, "ok": False, "error": "parquet missing"})
            ok = False
            continue
        pq = pd.read_parquet(pq_path)

        frames = {}
        for tf in [timeframe] + ctx:
            df = pd.read_csv(TRAINING_REPO / "data" / "raw" / "mt5" / symbol / (tf + ".csv"))
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["symbol"] = symbol
            df["timeframe"] = tf
            frames[tf] = df[["timestamp", "symbol", "timeframe",
                             "open", "high", "low", "close", "spread", "tick_volume"]]

        model_names = list(json.loads(
            (REPO_ROOT / "models" / "baseline" / symbol / timeframe / "model.json")
            .read_text(encoding="utf-8"))["learner"]["feature_names"])

        # Latest validation row whose entry candle sits inside every frame's
        # coverage, deep enough past all warm-ups.
        max_ts = min(frames[tf]["timestamp"].max() for tf in frames)
        cand = pq[pq["timestamp"] <= max_ts - pd.Timedelta(hours=72)]
        ref = cand.iloc[-1]

        sliced = LiveFeaturePipeline.slice_history(frames, pd.Timestamp(ref["timestamp"]))
        series, _specs = LiveFeaturePipeline(cfg, symbol, timeframe).compute(sliced)
        row = {str(k): float(v) for k, v in series.items()}
        row["entry_direction"] = float(ref["entry_direction"])

        worst_name, worst, missing = None, 0.0, []
        for n in model_names:
            if n not in row:
                missing.append(n)
                continue
            d = abs(row[n] - float(ref[n]))
            if d > worst:
                worst, worst_name = d, n
        part = {
            "schema": f"{symbol}/{timeframe}",
            "features_compared": len(model_names),
            "ref_close_time": str(ref["close_time"]),
            "max_abs_diff": worst,
            "worst_feature": worst_name,
            "missing_from_pipeline": missing,
            "tolerance": PARITY_TOLERANCE,
            "ok": bool(worst <= PARITY_TOLERANCE and not missing),
        }
        ok = ok and part["ok"]
        results.append(part)
    return {"ok": ok, "schemas": results}


def main() -> int:
    print("=" * 76)
    print("BASELINE INTEGRATION VERIFICATION")
    print("=" * 76)

    parts: Dict[str, Any] = {}
    parts["load_and_mapping"] = check_load_and_mapping()
    parts["mapping_identity"] = check_mapping_via_spy()
    parts["source_isolation"] = check_source_isolation()
    parts["gold_parity"] = check_gold_parity()
    parts["predictions"] = check_predictions()

    print(f"\n{'check':<22}{'result':<8}")
    print("-" * 34)
    all_ok = True
    for name, part in parts.items():
        print(f"{name:<22}{'PASS' if part.get('ok') else 'FAIL':<8}")
        all_ok = all_ok and bool(part.get("ok"))
    print("-" * 34)

    # Human-readable prediction table
    print(f"\n{'model':<16}{'p_win(BUY)':>11}{'p_win(SELL)':>12}   status")
    for r in parts["predictions"]["predictions"]:
        b = r["buy_p_win"]; s = r["sell_p_win"]
        print(f"{r['model']:<16}"
              f"{(f'{b:.4f}' if b is not None else 'n/a'):>11}"
              f"{(f'{s:.4f}' if s is not None else 'n/a'):>12}   "
              f"{r['buy_status']}/{r['sell_status']}")

    parity = parts["gold_parity"]
    print("\nGold parity (vendored pipeline vs training parquet):")
    for s in parity["schemas"]:
        if s.get("ok"):
            print(f"  {s['schema']:<14} max|diff|={s['max_abs_diff']:.3e} "
                  f"(worst: {s['worst_feature']})  [{s['features_compared']} features]")
        else:
            print(f"  {s['schema']:<14} FAIL — {s.get('error') or s.get('missing_from_pipeline') or s.get('worst_feature')}")

    document = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "xgboost_version": xgb.__version__,
        "all_pass": all_ok,
        **parts,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
    print(f"\nreport written: {REPORT_PATH}")

    print("\n" + "=" * 76)
    print("FINAL VERDICT:", "INTEGRATION VERIFIED — 9/9 MODELS MAPPED, SERVING, ISOLATED"
          if all_ok else "VERIFICATION FAILED")
    print("=" * 76)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

