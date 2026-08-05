from __future__ import annotations

import sys
import logging
from typing import Any, Dict, List

from execution.post_entry.rules.rule_engine import RuleEngine
from execution.post_entry.red_flags.red_flag_detector import RedFlagDetector
from execution.post_entry.xgboost_exit_model_adapter import XGBoostExitModelAdapter
from execution.post_entry.decision_fusion import DecisionFusionEngine


class MockPositionState:
    def __init__(self) -> None:
        self.mfe = 0.0
        self.mae = 0.0
        self.last_spread = 15.0


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )


def _build_snapshot(
    profit: float,
    p_win: float,
    market_regime: str,
) -> Dict[str, Any]:
    return {
        "trade": {
            "order_id": 123,
            "symbol": "EURUSD",
            "direction": "buy",
            "profit": profit,
            "spread": 15.0,
            "time_open": None,
            "volume": 1.0,
        },
        "expected_row": {
            "p_win": p_win,
        },
        "market_regime": market_regime,
    }


def _inject_snapshot_bridges(
    snapshot: Dict[str, Any],
    rule_results: List[Any],
    red_flag_meta: Dict[str, Any],
) -> None:
    # EXACTLY like post_entry_manager bridges
    snapshot["_rule_results_by_name"] = {rr.rule_name: rr for rr in rule_results}

    snapshot["_red_flag_report"] = {
        "should_consult_exit_model": red_flag_meta.get("should_consult_exit_model"),
        "flag_count": red_flag_meta.get("flag_count"),
        "severity": red_flag_meta.get("severity"),
        "triggered_flags": red_flag_meta.get("triggered_flags"),
    }


def _expected_decision_from_spec(exit_probability: float) -> str:
    # Spec in Arabic prompt:
    # Probability > 90%  -> Close
    # 70–90% -> Tighten SL
    # <70% -> Continue
    # But DecisionFusion implementation does not implement these thresholds.
    if exit_probability > 0.90:
        return "ClosePosition"
    if 0.70 <= exit_probability <= 0.90:
        return "MoveSL"
    return "Continue"


def main() -> int:
    _setup_logging()

    rule_engine = RuleEngine()
    red_flag_detector = RedFlagDetector()
    adapter = XGBoostExitModelAdapter()
    fusion = DecisionFusionEngine()

    state = MockPositionState()

    # Chosen values:
    # - profit low (10) so after update_mfe_mae(90) ProfitDecay triggers (decay ~ 88.8%)
    # - p_win low so TradeHealth triggers
    # - market_regime ranging so BadMarketRegime triggers
    profit = 10.0
    p_win = 0.1
    market_regime = "ranging"

    snapshot = _build_snapshot(profit=profit, p_win=p_win, market_regime=market_regime)

    # ====== (a) End-to-end chain with REAL model predict ======
    print("===== CASE A: REAL MODEL predict_exit_probability =====")

    rule_results = rule_engine.evaluate(snapshot, state.state if hasattr(state, "state") else "NEW")

    # ORDER as in post_entry_manager.py
    _inject_snapshot_bridges(snapshot, rule_results, red_flag_meta={})

    # adapter.update_mfe_mae before detector.detect
    adapter.update_mfe_mae(
        position_state=state,
        current_profit=90.0,  # simulate peak first so mfe becomes 90
    )
    adapter.update_mfe_mae(
        position_state=state,
        current_profit=profit,  # then current profit becomes 10
    )

    red_flags, _rf_score, red_flag_score_meta = red_flag_detector.detect(snapshot, position_state=state.state if hasattr(state, "state") else None)

    _inject_snapshot_bridges(snapshot, rule_results, red_flag_meta=red_flag_score_meta)

    xgb = adapter.predict(snapshot, position_state=state)
    exit_prob_real = xgb.get("exit_probability")
    decision_real = fusion.fuse(rule_results, red_flags, xgb, snapshot)

    print(f"update state: mfe={state.mfe}, mae={state.mae}, profit={profit}")
    print(f"real exit_probability: {exit_prob_real}")
    print(f"real decision.decision: {decision_real.decision}")
    print(f"red_flag_report: {snapshot.get('_red_flag_report')}")

    # ====== (b) Threshold simulation ======
    print("\n===== CASE B: Threshold simulation (force exit_probability) =====")

    test_probs = [0.95, 0.80, 0.50]
    rows = []

    # Keep same computed rule_results + red_flags (from real red flag detector)
    # DecisionFusion ignores thresholds, but we still run to verify influence.
    for forced in test_probs:
        # Create forced xgb dict; preserve confidence/continue fields for completeness
        xgb_forced = {
            "continue_probability": 1.0 - forced,
            "exit_probability": forced,
            "confidence": forced,
        }

        # Ensure snapshot bridge injection is present (same meta as computed)
        # (No code change in DecisionFusion; injection is for thresholds-like logic, if used elsewhere.)
        decision_forced = fusion.fuse(rule_results, red_flags, xgb_forced, snapshot)

        expected = _expected_decision_from_spec(forced)
        actual = decision_forced.decision
        should_consult = snapshot["_red_flag_report"]["should_consult_exit_model"]

        rows.append((forced, expected, actual, expected == actual, should_consult))

        print(
            f"forced exit_probability={forced:.2f} | "
            f"expected(spec)={expected} | actual={actual} | "
            f"should_consult_exit_model={should_consult}"
        )

    print("\n===== SUMMARY TABLE =====")
    print("exit_probability | expected_decision(spec) | actual_decision | matches? | should_consult_exit_model")
    for (forced, expected, actual, ok, should_consult) in rows:
        print(f"{forced:.2f} | {expected} | {actual} | {ok} | {should_consult}")

    # Assertions / explicit conclusions:
    any_should_consult_all = all(r[4] for r in rows)
    print(f"\nshould_consult_exit_model reached in all forced scenarios? {any_should_consult_all}")

    # Coverage: DecisionFusion called in all 4 cases (1 real + 3 forced)
    print("\n[END-TO-END] DecisionFusion.fuse called for:")
    print("- CASE A (real model): YES")
    print("- CASE B forced 0.95: YES")
    print("- CASE B forced 0.80: YES")
    print("- CASE B forced 0.50: YES")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
