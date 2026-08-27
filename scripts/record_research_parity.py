#!/usr/bin/env python3
"""Run golden parity for imported research models and record the result.

    python scripts/record_research_parity.py
    python scripts/record_research_parity.py --prefix golden_v3 --symbol XAUUSD

Why this is a separate step
---------------------------
The production gate asks whether KAIROS reproduces the research repository's
feature vectors and probabilities. That question can only be answered by
running KAIROS's own engine against the reference fixtures and measuring the
difference -- so the answer is written here, by the side being tested, and
never by the exporter that produced the reference.

The recorded numbers are the MAXIMUM absolute deltas over every golden sample,
not an average: a mean hides the one row where an implementation diverges, and
the one row is the whole point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.research import candles as cd  # noqa: E402
from analysis.research import engine as E  # noqa: E402
from analysis.research import inference as inf  # noqa: E402
from analysis.research.model_loader import ModelNotCompatible, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "fixtures" / "research" / "golden"
CANDLES = ROOT / "tests" / "fixtures" / "research" / "candles"
EVIDENCE = "research_evidence.json"

#: The tolerance a parity claim is made at. Both repositories run identical
#: library versions, so agreement is exact in practice; the tolerance exists so
#: a future libm or BLAS change moves the last bits without falsely failing.
TOLERANCE = 1e-12


def version_of(golden_path: Path) -> str:
    """Which research generation this fixture references.

    Both are registered for every symbol/timeframe, so a load has to name one;
    the registry refuses to choose.
    """
    return "research_v3" if golden_path.name.startswith("golden_v3_") else "research_v2"


def check(golden_path: Path, source: cd.CandleSource) -> dict:
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    symbol, tf = golden["symbol"], golden["entry_timeframe"]
    model = load_model(symbol, tf, version=version_of(golden_path))

    stack = cd.load_stack(source, golden["fixture_symbol"],
                          [tf, *golden["context_timeframes"]])
    frame = E.build_feature_frame(golden["fixture_symbol"], tf, stack,
                                  golden["context_timeframes"]).set_index("timestamp")

    max_feat = 0.0
    worst_feature = None
    max_prob = 0.0
    n = 0
    statuses = set()
    for sample in golden["samples"]:
        ts = pd.Timestamp(sample["timestamp"])
        row = frame.loc[ts].to_dict()
        for name, expected in sample["features"].items():
            if name == "entry_direction":
                continue
            d = abs(float(row[name]) - float(expected))
            if d > max_feat:
                max_feat, worst_feature = d, name
        pred = inf.predict_row(model, row, entry_direction=sample["direction"],
                               timestamp=ts)
        statuses.add(pred.status)
        if pred.status != "OK":
            continue
        max_prob = max(max_prob, abs(pred.p_win - float(sample["p_win"])))
        n += 1

    passed = (n == len(golden["samples"]) and max_feat <= TOLERANCE
              and max_prob <= TOLERANCE)
    return {
        "passed": bool(passed),
        "n_samples": n,
        "n_expected": len(golden["samples"]),
        "max_feature_delta": float(max_feat),
        "worst_feature": worst_feature,
        "max_probability_delta": float(max_prob),
        "tolerance": TOLERANCE,
        "statuses": sorted(statuses),
        "golden_fixture": golden_path.name,
        "model_id": model.card.model_id,
        "model_hash": model.card.model_hash,
        "fixture_ohlc": golden.get("fixture_ohlc"),
        "note": ("Maximum absolute delta over every golden sample. The fixture's "
                 "spread column is synthetic and some stacks are synthetic OHLC: "
                 "this measures implementation agreement, not model quality."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="golden",
                    help="golden fixture prefix: 'golden' for research_v2, "
                         "'golden_v3' for research_v3")
    ap.add_argument("--symbol"), ap.add_argument("--tf")
    ap.add_argument("--dry-run", action="store_true",
                    help="report parity without writing the evidence file")
    args = ap.parse_args()

    source = cd.JsonCandleSource(CANDLES)
    files = sorted(GOLDEN.glob(f"{args.prefix}_*.json"))
    if args.prefix == "golden":
        # `golden_*` would also sweep in `golden_v3_*`; the two generations are
        # recorded separately so a v3 parity result never lands on a v2 model.
        files = [f for f in files if not f.name.startswith("golden_v3_")]
    if not files:
        print(f"no golden fixtures matching {args.prefix}_*.json under {GOLDEN}",
              file=sys.stderr)
        return 2

    failures = 0
    for path in files:
        golden = json.loads(path.read_text(encoding="utf-8"))
        if args.symbol and golden["symbol"] != args.symbol:
            continue
        if args.tf and golden["entry_timeframe"] != args.tf:
            continue
        try:
            result = check(path, source)
        except (ModelNotCompatible, cd.CandleSourceError) as exc:
            print(f"{path.name}: SKIP -- {exc}")
            continue

        flag = "PASS" if result["passed"] else "FAIL"
        print(f"{flag} {golden['symbol']}/{golden['entry_timeframe']}: "
              f"max|dfeature|={result['max_feature_delta']:.3e} "
              f"max|dp_win|={result['max_probability_delta']:.3e} "
              f"({result['n_samples']}/{result['n_expected']} samples)")
        failures += 0 if result["passed"] else 1

        if args.dry_run:
            continue
        model_dir = (ROOT / "models" / "research" / version_of(path)
                     / golden["symbol"] / golden["entry_timeframe"])
        evidence_path = model_dir / EVIDENCE
        if not evidence_path.exists():
            # research_v2 artifacts carry no evidence file; parity is still
            # measured and printed, it simply has nowhere to be recorded.
            continue
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["kairos_parity"] = result
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True),
                                 encoding="utf-8")
        print(f"     recorded in {evidence_path.relative_to(ROOT)}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
