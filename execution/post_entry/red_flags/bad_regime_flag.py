from __future__ import annotations

from typing import Any, Dict, Optional

from .flag_result import FlagResult


class BadRegimeFlag:
    """Bad Market Regime red-flag.

    Source of truth (raw):
      - snapshot['market_regime'] (from market_snapshot_builder.py candle fields)

    Trigger when regime is in {'ranging', 'volatile'}.
    """

    name = "BadMarketRegime"

    @classmethod
    def check(
        cls,
        snapshot: Dict[str, Any],
        position_state: Optional[Any] = None,
    ) -> FlagResult:
        market_regime = snapshot.get("market_regime")
        if market_regime is None:
            trade = snapshot.get("trade") or {}
            market_regime = trade.get("market_regime")

        regime_str = str(market_regime or "UNKNOWN").lower()

        bad = {"ranging", "volatile"}
        triggered = regime_str in bad

        score = 1.0 if triggered else 0.0
        severity = 2 if triggered else 0

        return FlagResult(
            name="BadMarketRegime",
            triggered=bool(triggered),
            score=float(score),
            severity=int(severity),
            reason=f"bad_regime:{regime_str}" if triggered else "regime_ok",
            metadata={
                "market_regime": regime_str,
                "source": "snapshot.market_regime",
                "bad_set": list(bad),
            },
        )

