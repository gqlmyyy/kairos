from __future__ import annotations

"""Temporary diagnostic script for QuantDinger M15 timeframe support.

Does NOT modify pipeline code and does NOT touch Exit.

Runs from project root so imports resolve correctly.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional


# Ensure project root is on sys.path when run via `python scripts/...`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.market.client import get_candles  # noqa: E402


SYMBOLS: List[str] = ["EURUSD", "GBPUSD", "XAUUSD"]
TIMEFRAME_VARIANTS: List[str] = [
    "M15",
    "15M",
    "15",
    "15Min",
    "MIN15",
    "PERIOD_M15",
    "15m",
    "m15",
]

MIN_COUNT_THRESHOLD = 100
COUNT = 200


def _first_rows(candles: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
    return [x for x in candles[:n] if isinstance(x, dict)]


def main() -> None:
    any_success = False

    for sym in SYMBOLS:
        print(f"\n=== Diagnosing M15 for {sym} ===")
        success_variants: List[str] = []

        for tf in TIMEFRAME_VARIANTS:
            try:
                candles = get_candles(sym, tf, COUNT)
                cnt = len(candles) if candles else 0
                time.sleep(0.4)


                if cnt > 0:
                    any_success = True
                    success_variants.append(tf)
                    print(f"[OK]  tf='{tf}' -> count={cnt}")
                    for row in _first_rows(candles, 3):
                        print("   first:", row)
                else:
                    print(f"[NO ] tf='{tf}' -> count=0")
            except Exception as e:
                print(f"[ERR] tf='{tf}' -> {type(e).__name__}: {e}")

        if success_variants:
            best: Optional[str] = None
            best_cnt = -1
            for tf in success_variants:
                cnt = len(get_candles(sym, tf, COUNT))
                if cnt > best_cnt:
                    best_cnt = cnt
                    best = tf

            print(f"\n✅ Successful variants for {sym}: {success_variants}")
            print(f"Best='{best}' count={best_cnt} (>= {MIN_COUNT_THRESHOLD}? {best_cnt >= MIN_COUNT_THRESHOLD})")
        else:
            print(f"\n❌ M15 غير مدعوم من QuantDinger لهذه الأزواج (no successful timeframe variant found for {sym}).")

    if not any_success:
        print("\nM15 غير مدعوم من QuantDinger لهذه الأزواج")


if __name__ == "__main__":
    main()

