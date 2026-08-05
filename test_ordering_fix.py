from __future__ import annotations

import logging
import sys
from typing import Any, Dict

from execution.post_entry.red_flags.red_flag_detector import RedFlagDetector


class MockPositionState:
    def __init__(self) -> None:
        # IMPORTANT: start from empty tracking to prove ordering fix
        self.mfe = 0.0
        self.mae = 0.0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    detector = RedFlagDetector()

    # This must be called FIRST (same ordering as post_entry_manager.py)
    position_state = MockPositionState()

    # Adapter is inside red flag detector module usage chain; we import directly
    from execution.post_entry.xgboost_exit_model_adapter import XGBoostExitModelAdapter

    adapter = XGBoostExitModelAdapter()

    # Scenario:
    # - call adapter.update_mfe_mae(position_state, current_profit=10)
    # - expect adapter to set position_state.mfe=90 via internal peak tracking? ->
    #   In this adapter logic, mfe becomes max(prev_mfe, current_profit).
    #   Therefore to get mfe=90 with current_profit=10, we must simulate a peak update.
    #   We do this by calling adapter twice: first to set peak to 90, then to update current profit to 10.

    # Step 1: set a peak profit of 90
    adapter.update_mfe_mae(position_state=position_state, current_profit=90.0)

    # Step 2: now update current profit to 10 (mfe must stay 90, mae becomes 10 or smaller depending)
    current_profit = 10.0
    adapter.update_mfe_mae(position_state=position_state, current_profit=current_profit)

    snapshot: Dict[str, Any] = {
        "trade": {
            "order_id": 123,
            "symbol": "EURUSD",
            "direction": "buy",
            "profit": current_profit,
            "spread": 15,
            "time_open": None,
            "volume": 1.0,
        },
        "expected_row": {
            "p_win": 0.1,
        },
        "market_regime": "ranging",
    }

    # Now run detector after mfe/mae were updated in correct order
    red_flags, model_score, meta = detector.detect(snapshot, position_state=position_state)

    report = meta.get("report")
    profit_decay = report.flags.get("ProfitDecay") if report else None

    print("\n=== ORDERING FIX TEST OUTPUT ===")
    print("position_state.mfe:", getattr(position_state, "mfe", None))
    print("position_state.mae:", getattr(position_state, "mae", None))
    print("current_profit:", current_profit)
    print("ProfitDecay.triggered:", getattr(profit_decay, "triggered", None))
    print("red_flags:", red_flags)
    print("meta keys:", list(meta.keys()))
    print("=== END ===")

    # Assertion-like behavior (exit 1 on failure)
    if report is None or profit_decay is None or profit_decay.triggered is not True:
        logging.error("ProfitDecay did NOT trigger; ordering fix may be broken")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

