from __future__ import annotations

"""scripts/dryrun_baseline_gate.py

Runtime dry-run of the baseline XGBoost integration. No order is ever sent.

Part A — the prediction matrix the integration contract asks for: every
(symbol, timeframe) combo, both directions, served from live MT5 candles:

    symbol=EURUSD tf=M15 model=models/baseline/EURUSD/M15/model.json p_win=... available=True status=OK

Part B — the decision chain for the live decision timeframe (H1, TF_DECISION)
through the REAL trade gate with REAL sizing inputs (equity from MT5, ATR from
the feature row, SL/TP from tm_config multipliers, position size from
calculate_position_size, multiplier from get_size_multiplier), demonstrating
ALLOW and REJECT outcomes with their true reasons.

Part C — proof of source isolation: every Booster.load_model call observed in
this process resolved under models/baseline.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import xgboost as xgb  # noqa: E402

# Instrument every model load BEFORE the gate module is imported.
_loaded: list = []
_original = xgb.Booster.load_model


def _spy(self, *a, **k):
    _loaded.append(str(a[0]) if a else str(k))
    return _original(self, *a, **k)


xgb.Booster.load_model = _spy

from analysis.baseline import gate  # noqa: E402
from config import SYMBOLS, TF_DECISION  # noqa: E402
from data.market.mt5_client import get_equity  # noqa: E402
from risk.position_sizing import (  # noqa: E402
    MAX_LOT_PER_SYMBOL,
    calculate_position_size,
)
from risk.trade_gate import GateDecision, TradeRequest, validate_trade_request  # noqa: E402
from analysis.models.xgboost_v2_inference import get_size_multiplier  # noqa: E402
from trade_management.tm_config import (  # noqa: E402
    ATR_SL_BASE_MULTIPLIER,
    ATR_TP_BASE_MULTIPLIER,
)

TIMEFRAMES = ("M15", "H1", "H4")


def part_a_matrix() -> bool:
    print("\n--- PART A: prediction matrix (live candles, both sides) ---")
    ok = True
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            for direction in ("BUY", "SELL"):
                r = gate.predict_entry(symbol, tf, direction)
                model = r.get("model") or str(gate.model_path(symbol, tf))
                print(f"symbol={symbol} tf={tf} model={model} direction={direction} "
                      f"p_win={r['p_win'] if r['p_win'] is not None else float('nan'):.4f} "
                      f"available={r['available']} status={r['status']}")
                ok = ok and r["available"] and r["status"] == "OK" \
                    and r["p_win"] is not None and 0.0 <= r["p_win"] <= 1.0
    return ok


def part_b_decision_chain() -> bool:
    print(f"\n--- PART B: decision chain on the live decision timeframe "
          f"(tf={TF_DECISION}) ---")
    equity = get_equity()
    print(f"equity={equity}")
    ok = True
    for symbol in SYMBOLS:
        row = gate.compute_feature_row(symbol, TF_DECISION)
        atr = row["atr"]
        sl_distance = atr * ATR_SL_BASE_MULTIPLIER
        tp_distance = atr * ATR_TP_BASE_MULTIPLIER
        position_size = calculate_position_size(
            equity, sl_distance, symbol, consecutive_losses=0, score=70.0)

        for direction in ("BUY", "SELL"):
            r = gate.predict_entry(symbol, TF_DECISION, direction)
            p_win = r["p_win"]
            multiplier = get_size_multiplier(p_win) if p_win is not None else 0.0
            request = TradeRequest(
                symbol=symbol,
                direction=direction,
                final_score=70.0,
                ai_confidence=0.8,
                confidence=0.8,
                equity=equity,
                position_size=position_size,
                sl_distance=sl_distance,
                tp_distance=tp_distance,
                signal_is_valid=True,
                ml_available=r["available"],
                ml_p_win=p_win,
                ml_threshold=0.60,
                ml_status=r["status"],
                size_multiplier=multiplier,
                open_position_count=0,
                risk_passed=True,
                risk_reason="",
            )
            result = validate_trade_request(request)
            if result.allowed:
                base = position_size
                final = round(min(base * multiplier,
                                  MAX_LOT_PER_SYMBOL.get(symbol, 0.10)), 2)
                outcome = (f"ALLOW final_size={final} lots "
                           f"(base={base} x mult={multiplier}, "
                           f"cap={MAX_LOT_PER_SYMBOL.get(symbol, 0.10)})")
            else:
                outcome = f"REJECT reason={result.reason}"
            print(f"symbol={symbol} tf={TF_DECISION} direction={direction} "
                  f"p_win={p_win:.4f} multiplier={multiplier} "
                  f"gate={result.decision.value} -> {outcome}")
            ok = ok and result.decision in (GateDecision.ALLOW, GateDecision.REJECT)
    return ok


def part_c_isolation() -> bool:
    print("\n--- PART C: source isolation (every model load this process made) ---")
    baseline_root = (REPO_ROOT / "models" / "baseline").resolve()
    outside = [p for p in _loaded
               if not Path(p).resolve().is_relative_to(baseline_root)]
    for p in sorted(set(_loaded)):
        print(f"loaded: {p}")
    print(f"loads outside models/baseline: {len(outside)}")
    return not outside


def main() -> int:
    print("=" * 76)
    print("BASELINE INTEGRATION DRY-RUN (no orders are sent)")
    print("=" * 76)
    a = part_a_matrix()
    b = part_b_decision_chain()
    c = part_c_isolation()
    print("\n" + "=" * 76)
    verdict = a and b and c
    print("DRY-RUN VERDICT:", "PASS" if verdict else "FAIL")
    print("=" * 76)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
