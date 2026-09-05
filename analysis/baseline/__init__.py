"""Baseline entry-model integration.

`models/baseline/` is the ONLY source of trained entry models:

    models/baseline/<SYMBOL>/<TIMEFRAME>/model.json

nine artifacts: {EURUSD, GBPUSD, XAUUSD} x {M15, H1, H4}, native XGBoost
JSON (binary:logistic), trained by the xgbooost rebuild pipeline. The
feature calculator is vendored byte-for-byte from that pipeline
(`vendor/src/...` + `vendor/config/*.yaml`) so training and serving share
one arithmetic by construction -- there is no second formula set.

Nothing here reads the legacy per-process artifact, the quarantined v2
experiment, or the research registry. There is no fallback: a missing
artifact is an unavailable gate, never a different model.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The vendored pipeline imports itself as `src.features...` / `src.config...`
# (byte-identical to the training repository). Putting the vendor root on
# sys.path makes those absolute imports resolve without editing a byte of the
# copied code -- which is the point: zero drift from what trained the models.
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
