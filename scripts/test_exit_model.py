#!/usr/bin/env python3
"""
Test script for Exit Model - simulates a high-risk trade scenario
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger

logger = get_logger("test_exit_model")


def main():
    print("=" * 60)
    print("Exit Model Test - High Risk Trade Scenario")
    print("=" * 60)
    print()

    # Simulate high-risk position data
    symbol = "EURUSD"
    pos = {
        "symbol": symbol,
        "order_id": 999999,
        "type": "buy",
        "volume": 0.1,
        "spread": 2,
        "time_open_hours": 12.0,  # Trade open for 12 hours
        "atr": 0.0015,
    }

    # Expected row with dangerous indicators
    expected_row = {
        "actual_rsi": 65,
        "actual_atr": 0.0045,  # 3x ATR spike (0.0015 * 3 = 0.0045)
        "mfe": 0.001,
        "mae": 0.003,
        "actual_market_regime": "Ranging",  # Bad regime
        "profit_decay_pct": 80,  # 80% profit decay
    }

    regime = "ranging"  # Bad regime
    peak_profit = 150.0  # Was up $150
    current_profit = 30.0  # Now only up $30 (80% erosion)

    print(f"Symbol: {symbol}")
    print(f"Direction: {pos.get('type')}")
    print(f"Entry ATR: {pos.get('atr')}")
    print(f"Current ATR: {expected_row.get('actual_atr')}")
    print(f"ATR Spike: {expected_row.get('actual_atr') / pos.get('atr'):.1f}x")
    print(f"Market Regime: {regime}")
    print(f"Trade Health: 35 (low)")
    print(f"Peak Profit: ${peak_profit}")
    print(f"Current Profit: ${current_profit}")
    print(f"Profit Decay: {((peak_profit - current_profit) / peak_profit * 100):.0f}%")
    print()

    # Calculate trade health score
    trade_health = 35  # Low health

    # Calculate profit decay
    profit_decay = (peak_profit - current_profit) / peak_profit * 100

    # Check ATR spike
    atr = pos.get("atr", 0.001)
    current_atr = expected_row.get("actual_atr", atr)
    atr_spike = current_atr / atr if atr else 1.0

    print(f"--- Red Flags Calculated ---")
    print(f"  - profit_decay: {profit_decay:.0f}% (threshold: 70%) -> {'YES' if profit_decay > 70 else 'NO'}")
    print(f"  - trade_health: {trade_health} (threshold: 50) -> {'YES' if trade_health < 50 else 'NO'}")
    print(f"  - atr_spike: {atr_spike:.1f}x (threshold: 2.5) -> {'YES' if atr_spike > 2.5 else 'NO'}")
    print(f"  - bad_regime: {regime} (bad: ['ranging','volatile']) -> {'YES' if regime in ['ranging','volatile'] else 'NO'}")
    print()

    # Import and call _check_red_flag
    try:
        from execution.reconciliation import _check_red_flag

        # Get trade health score
        trade_health = 35.0  # Low health
        atr = pos.get("atr", 0.0015)  # Entry ATR

        has_red_flags, flag_count, exit_prob, reasons = _check_red_flag(
            pos=pos,
            expected_row=expected_row,
            regime=regime,
            peak_profit=peak_profit,
            current_profit=current_profit,
            trade_health=trade_health,
            atr=atr,
        )

        print(f"--- Red Flag Check Result ---")
        print(f"  Has Red Flags: {has_red_flags}")
        print(f"  Flag Count: {flag_count}")
        print(f"  Reasons: {reasons}")
        print()

    except Exception as e:
        print(f"ERROR calling _check_red_flag: {e}")
        import traceback
        traceback.print_exc()
        return

    # If red flag raised, use model to predict exit probability
    if has_red_flags:
        print(f"--- XGBoost Exit Model Prediction ---")

        try:
            from analysis.models.xgboost_exit_model import predict_exit_probability

            features = {
                "symbol": symbol,
                "direction": pos.get("type", ""),
                "atr": atr,
                "rsi": float(expected_row.get("actual_rsi", 50) or 50),
                "mfe": float(expected_row.get("mfe", 0) or 0),
                "mae": float(expected_row.get("mae", 0) or 0),
                "trade_health": trade_health,
                "profit_decay_pct": profit_decay,
                "time_open_hours": pos.get("time_open_hours", 0),
                "spread": float(pos.get("spread", 0) or 0),
                "news_impact": 0.0,
                "market_regime": regime,
                "volume": float(pos.get("volume", 0) or 0),
            }

            exit_prob = predict_exit_probability(features)
            print(f"  Exit Probability: {exit_prob:.1%}")
            print()

            # Make decision based on probability
            if exit_prob > 0.90:
                decision = "CLOSE IMMEDIATELY"
                action = "close"
            elif exit_prob > 0.70:
                decision = "TIGHTEN STOP LOSS"
                action = "tighten"
            else:
                decision = "IGNORE (monitor)"
                action = "ignore"

            print(f"--- Final Decision ---")
            print(f"  Action: {decision}")
            print()
            print(f"  Result: Trade would be {action.upper()}ed")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No red flags raised - no action needed")


if __name__ == "__main__":
    main()