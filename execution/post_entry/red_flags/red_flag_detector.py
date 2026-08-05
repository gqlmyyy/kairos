from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

from .flag_result import FlagResult, RedFlagReport
from .profit_decay_flag import ProfitDecayFlag
from .trade_health_flag import TradeHealthFlag
from .atr_spike_flag import ATRSpikeFlag
from .bad_regime_flag import BadRegimeFlag


logger = get_logger("red_flag_detector")


class RedFlagDetector:
    """Red Flag Aggregator (no execution).

    This detector aggregates 4 independent red-flag checkers.

    Integration contract with PostEntryManager:
      - PostEntryManager injects snapshot['expected_row'] from db_row.
      - PostEntryManager provides position_state to detect() for MFE/MAE tracking.

    Return signature kept compatible with DecisionFusionEngine:
      (red_flags: List[str], model_score: float, meta: Dict[str, Any])
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _severity_from_flag_count(flag_count: int) -> int:
        if flag_count <= 0:
            return 0
        if flag_count == 1:
            return 1
        if flag_count == 2:
            return 2
        return 3

    def detect(
        self,
        snapshot: Dict[str, Any],
        position_state: Optional[Any] = None,
    ) -> Tuple[List[str], float, Dict[str, Any]]:
        profit_decay = ProfitDecayFlag.check(snapshot=snapshot, position_state=position_state)
        trade_health = TradeHealthFlag.check(snapshot=snapshot, position_state=position_state)
        atr_spike = ATRSpikeFlag.check(snapshot=snapshot, position_state=position_state)
        bad_regime = BadRegimeFlag.check(snapshot=snapshot, position_state=position_state)

        # --- Suppress single BadRegime early closes (first 3 minutes) ---
        # Requirement:
        # - prevent closing based solely on BadRegime / single red flag within first 3 minutes.
        early_window_minutes = 3.0
        trade = snapshot.get("trade") or {}
        time_open_raw = trade.get("time_open")
        elapsed_min: Optional[float] = None
        if time_open_raw is not None:
            try:
                open_val = float(time_open_raw)
                # If it's epoch seconds, compute age; otherwise treat it as "seconds since open" duration.
                if open_val > 1e8:
                    elapsed_min = (time.time() - open_val) / 60.0
                else:
                    elapsed_min = open_val / 60.0
            except Exception:
                elapsed_min = None

        # If only BadMarketRegime would trigger and we're still in early window -> neutralize it.
        if elapsed_min is not None and elapsed_min < early_window_minutes:
            only_bad_triggers = bool(bad_regime.triggered) and (not profit_decay.triggered) and (not trade_health.triggered) and (not atr_spike.triggered)
            if only_bad_triggers:
                bad_regime = type(bad_regime)(
                    name=bad_regime.name,
                    triggered=False,
                    score=0.0,
                    severity=0,
                    reason="bad_regime_suppressed_early_window",
                    metadata=bad_regime.metadata,
                )

        flags: Dict[str, FlagResult] = {
            profit_decay.name: profit_decay,
            trade_health.name: trade_health,
            atr_spike.name: atr_spike,
            bad_regime.name: bad_regime,
        }

        triggered_flags = {k: v for k, v in flags.items() if v.triggered}
        triggered_names = list(triggered_flags.keys())
        flag_count = len(triggered_flags)

        total_score = float(sum(float(fr.score or 0.0) for fr in triggered_flags.values()))
        severity = self._severity_from_flag_count(flag_count)
        should_consult_exit_model = flag_count >= 2

        report = RedFlagReport(
            flags=flags,
            triggered_flags=triggered_flags,
            flag_count=flag_count,
            total_score=total_score,
            severity=severity,
            should_consult_exit_model=should_consult_exit_model,
            meta={
                "enabled_flags": list(flags.keys()),
                "flag_count": flag_count,
            },
        )

        logger.info("========== RED FLAG REPORT ==========")
        logger.info("ProfitDecay : %s", "YES" if profit_decay.triggered else "NO")
        logger.info("TradeHealth : %s", "YES" if trade_health.triggered else "NO")
        logger.info("ATRSpike : %s", "YES" if atr_spike.triggered else "NO")
        logger.info("BadRegime : %s", "YES" if bad_regime.triggered else "NO")
        logger.info("FlagCount : %s", flag_count)
        logger.info("Score : %.3f", total_score)
        logger.info("Severity : %s", severity)
        # ConsultExitModel log intentionally hidden (report-only change)
        logger.info("======================================")

        meta = {
            "report": report,
            "triggered_flags": triggered_names,
            "flag_count": flag_count,
            "total_score": total_score,
            "severity": severity,
            "should_consult_exit_model": should_consult_exit_model,
            "triggered_details": {k: v.metadata for k, v in triggered_flags.items()},
        }

        return triggered_names, total_score, meta

