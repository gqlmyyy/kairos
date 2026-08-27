"""Why won't the bot open a trade? One command, one answer.

The entry gate has several independent reasons to refuse, and a rejection
names only the first one hit. Worse, the reasons live in different places —
a file on disk, a booster's internal feature count, a config value — so
answering "why is nothing trading" meant reading three modules and a log.
This prints all of it at once and ends with the single next action.

Read-only and dependency-light by design: no MT5, no Windows, no broker
connection, no writes. It is meant to run anywhere the repository is
checked out, including in CI and on the machine that is NOT the trading
host.

Usage::

    python scripts/diagnose_entry_gate.py
    python scripts/diagnose_entry_gate.py --model models/entry/entry_model.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK = "ok  "
BAD = "FAIL"
WARN = "warn"


def line(status: str, label: str, detail: str = "") -> None:
    print(f"  [{status}] {label}" + (f"  {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def booster_feature_count(model_path: str):
    """(count, error). Never raises — a missing xgboost is a diagnostic
    result, not a crash."""
    try:
        import xgboost as xgb
    except ImportError:
        return None, "xgboost is not installed in this environment"
    try:
        booster = xgb.Booster()
        booster.load_model(model_path)
        return int(booster.num_features()), None
    except Exception as exc:  # noqa: BLE001 - report, never propagate
        return None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.path.join("models", "entry", "entry_model.json"))
    args = parser.parse_args()

    print("=" * 70)
    print("KAIROS — ENTRY GATE DIAGNOSIS")
    print("=" * 70)
    print("Read-only. No MT5 required. Nothing is written.")

    blockers: list = []
    notes: list = []

    # --- configuration ---------------------------------------------------
    section("1. CONFIGURATION")
    entry_model_version = ml_mode = absent_mult = None
    try:
        from config import (ENTRY_ML_ABSENT_SIZE_MULT, ENTRY_ML_MODE,
                            ENTRY_MODEL_VERSION)

        entry_model_version, ml_mode = ENTRY_MODEL_VERSION, ENTRY_ML_MODE
        absent_mult = ENTRY_ML_ABSENT_SIZE_MULT
        line(OK, "ENTRY_MODEL_VERSION =", entry_model_version)
        line(OK, "ENTRY_ML_MODE       =", ml_mode)
        line(OK, "ENTRY_ML_ABSENT_SIZE_MULT =", str(absent_mult))
    except Exception as exc:  # noqa: BLE001
        line(BAD, "config could not be imported:", str(exc))
        blockers.append("config is unreadable — fix that before anything else")

    # --- artifact --------------------------------------------------------
    section("2. MODEL ARTIFACT")
    model_exists = os.path.exists(args.model)
    line(OK if model_exists else BAD, f"{args.model}",
         "present" if model_exists else "MISSING")
    if not model_exists:
        blockers.append(f"no model artifact at {args.model}")

    sidecar = args.model + ".metadata.json"
    sidecar_exists = os.path.exists(sidecar)
    line(OK if sidecar_exists else BAD, f"{sidecar}",
         "present" if sidecar_exists else "MISSING")
    if model_exists and not sidecar_exists:
        blockers.append(
            "the model has no metadata sidecar, so entry_model_metadata refuses "
            "to serve it")

    # --- feature contract ------------------------------------------------
    section("3. FEATURE CONTRACT")
    live_names = ()
    try:
        from analysis.models.entry_feature_spec import FEATURE_NAMES

        live_names = tuple(FEATURE_NAMES)
        line(OK, "live spec features =", f"{len(live_names)}  {list(live_names)}")
    except Exception as exc:  # noqa: BLE001
        line(BAD, "cannot read the live feature spec:", str(exc))

    if model_exists:
        count, error = booster_feature_count(args.model)
        if error:
            line(WARN, "booster feature count =", f"unavailable ({error})")
            notes.append(f"booster feature count could not be read: {error}")
        else:
            matches = bool(live_names) and count == len(live_names)
            line(OK if matches else BAD, "booster features =", str(count))
            if not matches and live_names:
                line(BAD, "MISMATCH:",
                     f"artifact expects {count}, live path sends {len(live_names)}")
                blockers.append(
                    f"feature contract mismatch: artifact {count} vs live "
                    f"{len(live_names)} — the model cannot score this vector")

    # --- what the live loader actually does ------------------------------
    section("4. LIVE LOADER VERDICT")
    model_available = False
    try:
        from analysis.models.xgboost_v2_inference import load_v2_model

        model_available = load_v2_model() is not None
        line(OK if model_available else BAD, "load_v2_model() ->",
             "a model is serving" if model_available else "None (nothing is served)")
    except Exception as exc:  # noqa: BLE001
        line(BAD, "loader raised:", f"{type(exc).__name__}: {exc}")
        notes.append(f"loader raised: {exc}")

    # --- research path ----------------------------------------------------
    section("5. RESEARCH PATH (ENTRY_MODEL_VERSION=research)")
    try:
        from analysis.research import live_gate

        activations = live_gate.load_activations()
        line(OK, "pinned generation =", live_gate.VERSION)
        line(OK if activations else WARN, "activated models =",
             str(list(activations) or "none"))
        if entry_model_version == "research" and not activations:
            blockers.append(
                "ENTRY_MODEL_VERSION=research but no model is activated in "
                f"{live_gate.activation_path()}")
    except Exception as exc:  # noqa: BLE001
        line(WARN, "research path unavailable:", str(exc))

    # --- verdict -----------------------------------------------------------
    section("VERDICT")

    # Would a trade be blocked by ML right now, given the mode?
    ml_blocks = (not model_available) and ml_mode == "required"

    if ml_blocks:
        print("  NO TRADE WILL OPEN.")
        print()
        print("  Why: ENTRY_ML_MODE=required and no entry model is being served,")
        print("  so risk/trade_gate rejects every request with")
        print("      ml_unavailable:ML_MODEL_MISSING")
        print("  before it ever reaches sizing or the risk engine.")
        if blockers:
            print()
            print("  Root cause(s):")
            for item in blockers:
                print(f"    - {item}")
        print()
        print("  NEXT STEP — pick one:")
        print()
        print("   (a) Train a model that satisfies the contract (the real fix):")
        print("         python scripts/fetch_training_candles.py     # Windows + MT5")
        print("         python scripts/train_entry_model.py --dry-run")
        print("         python scripts/train_entry_model.py")
        print("       The trainer writes the metadata sidecar itself. Do NOT")
        print("       hand-write one: entry_model_metadata refuses a model that")
        print("       cannot prove its provenance, and that refusal is correct.")
        print()
        print("   (b) Trade without the ML filter, deliberately and at reduced size:")
        print("         ENTRY_ML_MODE=advisory")
        print(f"         ENTRY_ML_ABSENT_SIZE_MULT={absent_mult}   # optional")
        print("       Read ENTRY_MODEL_INVESTIGATION.md first: three years of real")
        print("       candles found NO exploitable edge (AUC 0.505-0.523, fold")
        print("       spreads of 17-21 sigma, negative Brier skill). The filter you")
        print("       would be switching off was not demonstrated to help.")
    elif not model_available and ml_mode in ("advisory", "off"):
        print(f"  TRADES CAN OPEN — but with NO ML FILTERING (mode={ml_mode}).")
        print(f"  Every entry is sized at {absent_mult} on signal + MTF alone.")
        if blockers:
            print()
            print("  The model is still unavailable because:")
            for item in blockers:
                print(f"    - {item}")
        print()
        print("  NEXT STEP: to restore filtering, train a model (see option (a) in")
        print("  KNOWN_ISSUES.md item 0), then set ENTRY_ML_MODE=required.")
    else:
        print("  The ML gate is NOT blocking: a model is being served.")
        print("  If trades still are not opening, the cause is downstream —")
        print("  check the DECISION_TRUTH line's reject_reason for the failing")
        print("  check (sizing, risk engine, or the risk governor).")

    if notes:
        print()
        print("  Notes:")
        for item in notes:
            print(f"    - {item}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
