from __future__ import annotations

import logging
import sys
from typing import Any, Dict

from execution.post_entry.red_flags.red_flag_detector import RedFlagDetector


class MockPositionState:
    def __init__(self) -> None:
        # Ensure ProfitDecay can trigger (mfe high, current profit lower)
        self.mfe = 100.0
        self.mae = -10.0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    detector = RedFlagDetector()

    position_state = MockPositionState()

    # Dry-run snapshot with raw keys used by the updated flag checkers.
    # Goal: trigger at least 2 flags (TradeHealth + BadRegime + optional ProfitDecay).
    snapshot: Dict[str, Any] = {
        "trade": {
            "order_id": 123,
            "symbol": "EURUSD",
            "direction": "buy",
            "profit": 10.0,
            "spread": 15,
            "time_open": None,
            "volume": 1.0,
        },
        "expected_row": {
            "p_win": 0.1,  # health_score=10 < 40 => TradeHealth triggers
        },
        "market_regime": "ranging",  # => BadRegime triggers
    }

    red_flags, model_score, meta = detector.detect(snapshot, position_state=position_state)

    print("\n=== DETECTOR OUTPUT ===")
    print("red_flags:", red_flags)
    print("model_score:", model_score)
    print("meta keys:", list(meta.keys()))
    print("_red_flag_report:", meta.get("report"))
    print("=== END ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

