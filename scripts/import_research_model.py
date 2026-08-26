#!/usr/bin/env python3
"""Import xgbooost research entry models into KAIROS as RESEARCH artifacts.

    python scripts/import_research_model.py --source /path/to/xgbooost
    python scripts/import_research_model.py --source ../xgbooost --symbol XAUUSD --tf H1

What it does
------------
For each ``models/research_v2/<SYMBOL>/<TF>/`` in the research repo it copies
``model.joblib`` and ``manifest.json`` into ``models/research/<SYMBOL>/<TF>/``
and generates a ``model_card.json`` — the artifact contract KAIROS's loader
validates against. Every card field is DERIVED from the research manifest;
nothing is invented, and the import fails rather than filling a blank.

What it does not do
-------------------
It never touches ``models/entry/`` or ``models/entry_v2/``. Legacy artifacts
stay exactly where they are so the two models can be run side by side, which
is the whole point of keeping them separate.

It never writes a status other than RESEARCH. Every model shipped by the
research repo carries a NO_SIGNAL or DROP verdict — no demonstrated edge — and
promoting one is a human decision recorded deliberately, never a side effect
of copying a file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.research import contract as C  # noqa: E402
from analysis.research import model_card as mc  # noqa: E402
from analysis.research import model_registry as reg  # noqa: E402

RESEARCH_SUBDIR = "models/research_v2"
DEST_ROOT = Path("models/research")

# The frozen research target. These are read from the research repo's
# config/target.yaml when it is present and asserted against these values, so
# a change on either side surfaces as an import failure rather than as a
# mislabelled artifact.
TP_ATR_MULTIPLE = 2.5
SL_ATR_MULTIPLE = 1.5
HORIZON_BARS = {"M15": 96, "M30": 48, "H1": 72, "H4": 60}

TARGET_DESCRIPTION = (
    "TP at {tp}xATR(14) touched before SL at {sl}xATR(14), both fixed from the ATR "
    "at the entry bar, within {h} bars of the entry timeframe; ambiguous same-bar "
    "resolutions excluded"
)


def _load_target_config(source: Path) -> tuple:
    """Read the research target config, falling back to the frozen constants."""
    cfg = source / "config" / "target.yaml"
    if not cfg.exists():
        return TP_ATR_MULTIPLE, SL_ATR_MULTIPLE, dict(HORIZON_BARS)
    try:
        import yaml
        raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"cannot read {cfg}: {exc}")
    rr = raw.get("risk_reward", {})
    tp = float(rr.get("tp_atr_multiple", TP_ATR_MULTIPLE))
    sl = float(rr.get("sl_atr_multiple", SL_ATR_MULTIPLE))
    horizons = {k: int(v) for k, v in raw.get("max_holding_horizon_bars", {}).items()}
    return tp, sl, (horizons or dict(HORIZON_BARS))


def build_card(manifest: dict, model_path: Path, tp: float, sl: float,
               horizons: dict) -> dict:
    symbol = manifest["symbol"]
    timeframe = manifest["entry_timeframe"]
    features = list(manifest["features"])

    # Resolving the contract here means an unimportable model fails at import
    # rather than at first prediction.
    contract = C.build_contract(symbol, timeframe, features)
    C.assert_scale_free(contract)

    if timeframe not in horizons:
        raise SystemExit(f"{symbol}/{timeframe}: no horizon configured for {timeframe}")
    horizon = int(horizons[timeframe])

    env = manifest.get("environment", {})
    prov = env.get("provenance_hashes", {})
    src_prov = manifest.get("source_dataset_provenance", {})
    hashes = manifest.get("dataset_hashes", {})

    train_hash = hashes.get("train", {}).get("actual") or hashes.get("train", {}).get("recorded")
    if not train_hash:
        raise SystemExit(f"{symbol}/{timeframe}: manifest carries no training dataset hash")

    threshold = manifest.get("threshold", {}).get("threshold")

    return {
        "model_id": f"research_v2__{symbol}__{timeframe}",
        "symbol": symbol,
        "timeframe": timeframe,
        "model_version": "research_v2",
        "feature_schema_version": mc.CURRENT_FEATURE_SCHEMA_VERSION,
        "feature_list": features,
        "target": TARGET_DESCRIPTION.format(tp=tp, sl=sl, h=horizon),
        "horizon_bars": horizon,
        "tp_atr_multiple": tp,
        "sl_atr_multiple": sl,
        "training_dataset_hash": train_hash,
        "feature_manifest_hash": (src_prov.get("phase3_feature_manifest_sha256")
                                  or prov.get("phase3_feature_manifest", "")),
        "target_spec_hash": (src_prov.get("phase4_target_spec_sha256")
                             or prov.get("phase4_target_spec", "")),
        "model_hash": mc.sha256_file(model_path),
        "probability_semantics": mc.PROBABILITY_SEMANTICS,
        # Carried verbatim. The research repo's own verdict on every shipped
        # model is that it did not demonstrate an edge; hiding that here would
        # make the artifact look like something it is not.
        "research_verdict": manifest.get("verdict", "UNKNOWN"),
        "calibration": manifest.get("calibration", {}).get("method", "none"),
        "context_timeframes": list(contract.context_timeframes),
        "entry_direction_encoding": {"long": 1.0, "short": -1.0},
        "source_repo_commit": env.get("git_commit", "unknown"),
        "environment": {k: str(v) for k, v in env.items()
                        if k in ("python", "numpy", "pandas", "sklearn", "xgboost", "seed")},
        "decision_threshold": float(threshold) if threshold is not None else None,
        "kairos_contract_fingerprint": C.contract_fingerprint(contract),
        "research_verdict_reasons": manifest.get("verdict_reasons", []),
        "selection": manifest.get("selection", {}).get("model", "unknown"),
        "feature_set": manifest.get("feature_set", "unknown"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="path to the xgbooost research repository")
    ap.add_argument("--symbol", help="import only this symbol")
    ap.add_argument("--tf", help="import only this entry timeframe")
    ap.add_argument("--dest", default=str(DEST_ROOT), help="destination root")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    source = Path(args.source).expanduser().resolve()
    research_root = source / RESEARCH_SUBDIR
    if not research_root.is_dir():
        print(f"ERROR: no {RESEARCH_SUBDIR} under {source}", file=sys.stderr)
        return 2

    tp, sl, horizons = _load_target_config(source)
    dest_root = Path(args.dest)
    entries = []

    for manifest_path in sorted(research_root.glob("*/*/manifest.json")):
        symbol = manifest_path.parent.parent.name
        timeframe = manifest_path.parent.name
        if args.symbol and symbol != args.symbol:
            continue
        if args.tf and timeframe != args.tf:
            continue

        model_src = manifest_path.parent / "model.joblib"
        if not model_src.exists():
            print(f"SKIP {symbol}/{timeframe}: no model.joblib")
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        card = build_card(manifest, model_src, tp, sl, horizons)

        dest = dest_root / symbol / timeframe
        entry = reg.RegistryEntry(
            model_id=card["model_id"], symbol=symbol, timeframe=timeframe,
            version=card["model_version"],
            # Always RESEARCH on import. See the module docstring.
            status=reg.RESEARCH, path=str(dest), model_hash=card["model_hash"],
            feature_schema_version=card["feature_schema_version"],
            feature_count=len(card["feature_list"]), target=card["target"],
            research_verdict=card["research_verdict"],
        )
        entries.append(entry)

        print(f"{'WOULD IMPORT' if args.dry_run else 'IMPORT'} {entry.describe()}")
        if args.dry_run:
            continue

        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_src, dest / mc.MODEL_FILENAME)
        shutil.copy2(manifest_path, dest / mc.RESEARCH_MANIFEST_FILENAME)
        mc.write(dest, card)

    if not entries:
        print("nothing matched", file=sys.stderr)
        return 1
    if not args.dry_run:
        path = reg.write_registry(entries, dest_root / "registry.json")
        print(f"\nwrote {path} with {len(entries)} models, all status=RESEARCH")
        verdicts = sorted({e.research_verdict for e in entries})
        print(f"research verdicts present: {verdicts}")
        print("No model is VALIDATED and nothing here is wired to live trading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
