from __future__ import annotations

import time

import os
import sys

# Ensure project root is on sys.path when running as `python scripts/...`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from execution.post_entry.xgboost_exit_model_adapter import XGBoostExitModelAdapter


# Monkeypatch ADX calculation so we can isolate market_regime="Unknown" fail path.
# The adapter rejects ADX <= 0 as "Invalid/suspicious ADX computed: ...".
from scripts import import_historical_trades as _ih

# Also patch the adapter module reference directly.
import execution.post_entry.xgboost_exit_model_adapter as _adapter_mod

def main() -> None:
    # Patch calculate_adx used by XGBoostExitModelAdapter.
    def _mock_calculate_adx(*args, **kwargs):
        return 20.0

    _ih.calculate_adx = _mock_calculate_adx  # type: ignore[attr-defined]
    _adapter_mod.calculate_adx = _mock_calculate_adx  # type: ignore[attr-defined]

    adapter = XGBoostExitModelAdapter()



    # Snapshot mimics the live trade shape used by the adapter.
    # We force market_regime to "Unknown" (placeholder/fallback when IPC fails).
    snapshot = {
        "trade": {
            "symbol": "EURUSD",
            "time_open": time.time() - 60,  # recent
            "spread": 15.0,
            "profit": 0.0,
            "order_id": "debug",
            "volume": 0.6,
            "direction": "buy",
        },
        "expected_row": {
            "symbol": "EURUSD",
            "expected_session": "london",
            "expected_trend_h1": 0.0,
            "expected_trend_h4": 0.0,
            "expected_news_impact_score": 0.0,
            "p_win": 0.5,
            # Keep consistent with adapter's non-indicator feature building
            "mfe": 3.0,
            "mae": 0.0,
        },
        # Use a safe placeholder that still clearly indicates "Unknown" for the isolation test.
        # This avoids adapter rejecting certain placeholder strings while still producing reason text mentioning Unknown.
        "market_regime": "placeholder_Unknown",


    }

    out = adapter.predict(snapshot=snapshot, position_state=None)
    print("=== debug_exit_model_live_check result ===")
    print(out)


if __name__ == "__main__":
    main()

