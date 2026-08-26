#!/usr/bin/env python3
"""Replay a research entry model over stored candles. Offline, Linux, no MT5.

    # against KAIROS's stored history (which carries no spread column)
    python scripts/research_replay.py --symbol XAUUSD --tf H1 --limit 200

    # against a source that does carry spread
    python scripts/research_replay.py --symbol XAUUSD --tf H1 \
        --source-kind json --source-root tests/fixtures/research/candles

Exit codes: 0 the replay ran (whether or not rows were servable), 2 the model
or the candles could not be resolved at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.research import candles as cd  # noqa: E402
from analysis.research import replay as rp  # noqa: E402
from analysis.research.model_loader import ModelNotCompatible  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--tf", required=True, help="entry timeframe, e.g. H1")
    ap.add_argument("--source-kind", default="kairos", choices=("kairos", "csv", "json"))
    ap.add_argument("--source-root", default="data/historical")
    ap.add_argument("--start"), ap.add_argument("--end")
    ap.add_argument("--limit", type=int, help="score the FIRST n rows of the range")
    ap.add_argument("--tail", type=int,
                    help="score the LAST n rows (usually what you want: the first "
                         "rows of a short history are still warming up)")
    ap.add_argument("--show", type=int, default=10, help="rows of output to print")
    args = ap.parse_args()

    if args.source_kind == "kairos":
        source = cd.KairosHistoricalSource(args.source_root)
    elif args.source_kind == "csv":
        source = cd.CsvCandleSource(args.source_root)
    else:
        source = cd.JsonCandleSource(args.source_root)

    try:
        limit, tail = args.limit, args.tail
        if limit is None and tail is None:
            tail = 200
        result = rp.replay(args.symbol, args.tf, source, start=args.start,
                           end=args.end, limit=limit, tail=tail)
    except (ModelNotCompatible, cd.CandleSourceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(result.summary())
    served = result.predictions[result.predictions["status"] == "OK"]
    if len(served):
        print(f"\np_win over {len(served)} served rows: "
              f"min={served['p_win'].min():.4f} median={served['p_win'].median():.4f} "
              f"max={served['p_win'].max():.4f}")
        print(served.head(args.show).to_string(index=False))
    else:
        print("\nNo row was servable. First refusals:")
        print(result.predictions.head(args.show)[
            ["timestamp", "direction", "status", "reason"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
