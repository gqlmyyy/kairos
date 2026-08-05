from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

import os
import sys

# Ensure project root is on sys.path when running as `python scripts/...`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from execution.post_entry.xgboost_exit_model_adapter import XGBoostExitModelAdapter


@dataclass
class DummyPositionState:
    mfe: float = 0.0
    mae: float = 0.0
    last_spread: float = 15.0


def run_case(adapter: XGBoostExitModelAdapter, current_profit: float) -> Dict[str, Any]:
    snapshot = {
        "trade": {
            "symbol": "EURUSD",
            "time_open": time.time() - 60,  # recent
            "spread": 15.0,
            "profit": current_profit,
            "order_id": f"debug_profit_{current_profit}",
            "volume": 0.6,
            "direction": "buy",
        },
        "expected_row": {
            "symbol": "EURUSD",
            "expected_session": "london",
            "expected_trend_h1": 50.0,
            "expected_trend_h4": 50.0,
            "expected_news_impact_score": 0.0,
            "p_win": 0.5,
            # These should NOT be used for mfe/mae when position_state is provided,
            # but we keep them for completeness.
            "mfe": 0.0,
            "mae": 0.0,
        },
        "market_regime": "TRENDING",
    }

    pos_state = DummyPositionState(mfe=0.0, mae=0.0, last_spread=15.0)
    out = adapter.predict(snapshot=snapshot, position_state=pos_state)
    return {
        "current_profit": current_profit,
        "exit_probability": out.get("exit_probability"),
        "continue_probability": out.get("continue_probability"),
        "features_incomplete": out.get("features_incomplete"),
        "position_state_after": {"mfe": pos_state.mfe, "mae": pos_state.mae},
    }


def main() -> None:
    adapter = XGBoostExitModelAdapter()

    cases = [50.0, -30.0]

    print("=== debug_exit_model_sensitivity_profit_injection ===")
    for cp in cases:
        res = run_case(adapter, cp)
        print(f"\n--- case current_profit={cp} ---")
        print(res)

    print("\nDone.")


if __name__ == "__main__":
    main()
