from __future__ import annotations

from typing import Any, Dict, Optional

from .flag_result import FlagResult


class ATRSpikeFlag:
    """ATR Spike red-flag.

    Feature disabled (baseline_atr unavailable).

    Reason (required):
      - baseline/average ATR keys are not available in the snapshot sources we verified.
      - Therefore we explicitly disable this feature.
    """

    name = "ATRTrailing"

    @classmethod
    def check(
        cls,
        snapshot: Dict[str, Any],
        position_state: Optional[Any] = None,
    ) -> FlagResult:
        return FlagResult(
            name="ATRSpike",
            triggered=False,
            score=0.0,
            severity=0,
            reason="baseline_atr unavailable — feature disabled pending data source",
            metadata={
                "feature_disabled": True,
                "baseline_atr": None,
                "source": "snapshot.expected_row/market_snapshot_builder (not available)",
            },
        )

