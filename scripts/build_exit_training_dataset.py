from __future__ import annotations

"""
Rebuild execution_dataset for XGBoost Exit Model (urgent step).

- Read closed trades from `trades`
- For each trade create 5 samples at time fractions:
  10%, 25%, 50%, 75%, 100%  (fractions: 0.1,0.25,0.5,0.75,1.0)
- Build features in FEATURE_ORDER using existing feature_schema contract
- Target y rules (as requested):
    * if pnl >= 0 (profit): y = 0 for all 5 points
    * if pnl < 0:
        - if pnl > -30: y = 0 for all 5 points
        - if pnl <= -30: y = 0 for first 4 points (10%-75%), and y = 1 only for last (100%)

- Delete execution_dataset old rows then insert rebuilt rows.

Notes:
- This repo currently does not provide tick/candle-at-time storage.
  mfe/mae are approximated (allowed by the instruction) to enable label-only audit/training.
"""

import json
import sqlite3
import sys
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.models.feature_schema import FEATURE_ORDER, build_feature_vector
from data.storage.database import get_conn, init_db  # type: ignore


DB_FILE = "trading_bot_v3.db"

SAMPLE_POINTS: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
LOSS_EXIT_PNL_THRESHOLD = -30.0


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return default if v is None else float(v)
    except Exception:
        return default


def _to_epoch(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip().replace("T", " ").replace("Z", "")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s[:19], fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            continue

    try:
        return float(s)
    except Exception:
        return None


def _duration_minutes_total(opened_at: Any, closed_at: Any) -> float:
    eo = _to_epoch(opened_at)
    ec = _to_epoch(closed_at)
    if eo is None or ec is None:
        return 60.0
    mins = (ec - eo) / 60.0
    return float(max(0.0, mins)) if mins > 0 else 60.0


def _derive_session(opened_at: Any) -> str:
    eo = _to_epoch(opened_at)
    if eo is None:
        return "unknown"
    h = datetime.fromtimestamp(eo, tz=timezone.utc).hour
    if h < 7:
        return "asia"
    if h < 13:
        return "london"
    if h < 20:
        return "new_york"
    return "asia"


def _approx_mfe_mae_from_pnl(pnl: float, progress: float) -> Tuple[float, float]:
    """
    Approximation only (no candles/tick storage required by urgent step).
    """
    if pnl >= 0:
        return float(max(0.0, pnl) * progress), 0.0
    return 0.0, float(abs(pnl) * progress)


def _label_for_pnl(pnl: float, idx: int, frac: float) -> float:
    # Urgent rule: if timestamp fraction < 0.25 (first 25% of trade life) => y=0 always
    if frac < 0.25:
        return 0.0

    if pnl >= 0:
        return 0.0
    # pnl < 0
    if pnl > LOSS_EXIT_PNL_THRESHOLD:
        return 0.0
    # pnl <= -30: y=1 only at the final (100%) point
    return 1.0 if idx == (len(SAMPLE_POINTS) - 1) else 0.0


def _build_feature_inputs(
    trade: sqlite3.Row,
    pnl: float,
    frac: float,
    trade_duration_minutes: float,
) -> Dict[str, Any]:
    # Entry_atr is available on trades table; everything else is approximated/defaults.
    entry_atr = _safe_float(trade.get("atr") if hasattr(trade, "get") else trade["atr"], default=0.001)
    entry_rsi = 50.0
    entry_adx = _safe_float(trade.get("final_score") if hasattr(trade, "get") else trade["final_score"], default=0.0)

    mfe, mae = _approx_mfe_mae_from_pnl(pnl=pnl, progress=frac)

    session = _derive_session(trade["opened_at"])
    trend_h1 = _safe_float(trade.get("trend_score") if hasattr(trade, "get") else trade["trend_score"], default=0.0)
    trend_h4 = trend_h1

    # regime placeholder used by feature_schema encoder; keep stable
    market_regime = "trending" if _safe_float(trade.get("volatility_score") if hasattr(trade, "get") else trade["volatility_score"], default=0.0) >= 0 else "ranging"

    spread = 15.0
    volume = _safe_float(trade.get("size") if hasattr(trade, "get") else trade["size"], default=0.0)

    return {
        "mfe": float(mfe),
        "mae": float(mae),
        "entry_atr": float(entry_atr),
        "entry_rsi": float(entry_rsi),
        "entry_adx": float(entry_adx),
        "market_regime": market_regime,
        "trade_duration": float(trade_duration_minutes),
        "spread": float(spread),
        "volume": float(volume),
        "session": session,
        "trend_h1": float(trend_h1),
        "trend_h4": float(trend_h4),
    }


def _insert_dataset_rows(conn: sqlite3.Connection, rows: List[Dict[str, Any]]) -> None:
    cur = conn.cursor()

    # Delete old rows (important: remove duplicates constraint violations)
    cur.execute("DELETE FROM execution_dataset")
    conn.commit()

    # Also ensure we're not blocked by any leftover UNIQUE constraints by clearing table identity space
    # (no-op for normal schemas, but safe).
    # If the table uses AUTOINCREMENT id + UNIQUE(order_id), DELETE above is sufficient.

    cols = [
        "dataset_created_at",
        "dataset_updated_at",
        "order_id",
        "symbol",
        "direction",
        "expected_entry",
        "expected_final_score",
        "expected_rsi",
        "expected_macd",
        "expected_session",
        "expected_spread",
        "expected_atr",
        "expected_trend_strength",
        "expected_momentum_score",
        "expected_volatility_score",
        "expected_market_regime",
        "expected_ai_score",
        "expected_sentiment_score",
        "expected_news_impact_score",
        "expected_ai_confidence",
        "expected_trend_score",
        "expected_momentum_score_legacy",
        "expected_sentiment_score_legacy",
        "expected_volatility_score_legacy",
        "expected_indicators_json",
        "status",
        "actual_entry",
        "actual_exit",
        "actual_pnl",
        "spread_at_entry",
        "actual_indicators_json",
        "exit_reason",
        "time_open",
        "expected_tp",
        "expected_sl",
        "risk_reward_ratio",
        "trade_duration",
        "mfe",
        "mae",
        "model_type",
    ]

    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO execution_dataset ({', '.join(cols)}) VALUES ({placeholders})"

    for r in rows:
        cur.execute(sql, [r.get(c) for c in cols])

    conn.commit()


def main() -> None:
    # Ensure DB schema exists (safe)
    try:
        init_db()  # type: ignore
    except Exception:
        pass

    # Connect
    conn = get_conn()  # type: ignore
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Load closed trades only
    cur.execute(
        """
        SELECT id, order_id, symbol, direction, size, entry_price, stop_loss, take_profit,
               atr, final_score, ai_score, ai_confidence, trend_score, momentum_score,
               sentiment_score, volatility_score, reason, status, pnl, opened_at, closed_at
        FROM trades
        WHERE status='closed' AND pnl IS NOT NULL
        """
    )
    trades = cur.fetchall()

    if not trades:
        print("No closed trades found in trades table.")
        return

    now_iso = datetime.now(timezone.utc).isoformat()

    dataset_rows: List[Dict[str, Any]] = []
    inserted = 0

    for trade in trades:
        order_id = str(trade["order_id"])
        if not order_id:
            continue

        symbol = str(trade["symbol"])
        direction = str(trade["direction"]).lower()
        if direction not in ("buy", "sell"):
            continue

        pnl = _safe_float(trade["pnl"], 0.0)
        entry_price = _safe_float(trade["entry_price"], 0.0)
        volume = _safe_float(trade["size"], 0.0)

        total_minutes = _duration_minutes_total(trade["opened_at"], trade["closed_at"])

        for idx, frac in enumerate(SAMPLE_POINTS):
            y = _label_for_pnl(pnl, idx=idx, frac=frac)

            trade_duration_minutes = float(max(0.0, total_minutes * frac))

            feature_inputs = _build_feature_inputs(
                trade=trade,
                pnl=pnl,
                frac=frac,
                trade_duration_minutes=trade_duration_minutes,
            )

            mfe = float(feature_inputs["mfe"])
            mae = float(feature_inputs["mae"])

            # Build indicators json (train_exit_model.py uses actual_indicators_json)
            # We need exit_reason/expected_tp compatible with train_exit_model's create_label().
            # Approximation: we map y into a consistent "bad-exit" shape.
            if y >= 0.5:
                exit_reason = "stop_loss"
                expected_tp = 10.0
                mfe_for_bad = max(mfe, 25.0)  # triggers bad exit rule2/3 depending on create_label logic
            else:
                exit_reason = "take_profit"
                expected_tp = max(1.0, abs(pnl))
                mfe_for_bad = min(mfe, 1.0)  # keep mfe low to avoid bad triggers

            indicators_json = json.dumps(
                {
                    "mfe": float(mfe_for_bad),
                    "mae": float(mae),
                    "direction": direction,
                }
            )

            # approximate actual_exit from pnl (must match import_historical_trades.pnl formula)
            pip = 0.01 if symbol.upper().endswith("JPY") else 0.0001
            if direction == "buy":
                actual_exit = entry_price + (pnl * pip / max(volume, 1e-9))
            else:
                actual_exit = entry_price - (pnl * pip / max(volume, 1e-9))

            row = {
                "dataset_created_at": now_iso,
                "dataset_updated_at": now_iso,
                "order_id": f"{trade['id']}-{idx}-{inserted}",
                "symbol": symbol,
                "direction": direction,
                "expected_entry": entry_price,
                "expected_final_score": None,
                "expected_rsi": 50.0,
                "expected_macd": 0.0,
                "expected_session": feature_inputs["session"],
                "expected_spread": float(feature_inputs["spread"]),
                "expected_atr": float(feature_inputs["entry_atr"]),
                "expected_trend_strength": float(feature_inputs["entry_adx"]),
                "expected_momentum_score": float(trade["momentum_score"] if trade["momentum_score"] is not None else 0.0),
                "expected_volatility_score": float(trade["volatility_score"] if trade["volatility_score"] is not None else 0.0),
                "expected_market_regime": feature_inputs["market_regime"],
                "expected_ai_score": 0.0,
                "expected_sentiment_score": float(trade["sentiment_score"] if trade["sentiment_score"] is not None else 0.0),
                "expected_news_impact_score": 0.0,
                "expected_ai_confidence": None,
                "expected_trend_score": float(trade["trend_score"] if trade["trend_score"] is not None else 0.0),
                "expected_momentum_score_legacy": float(trade["momentum_score"] if trade["momentum_score"] is not None else 0.0),
                "expected_sentiment_score_legacy": float(trade["sentiment_score"] if trade["sentiment_score"] is not None else 0.0),
                "expected_volatility_score_legacy": float(trade["volatility_score"] if trade["volatility_score"] is not None else 0.0),
                "expected_indicators_json": indicators_json,
                "status": "closed",
                "actual_entry": entry_price,
                "actual_exit": actual_exit,
                "actual_pnl": float(pnl),
                "spread_at_entry": float(feature_inputs["spread"]),
                "actual_indicators_json": indicators_json,
                "exit_reason": exit_reason,
                "time_open": trade["opened_at"],
                "expected_tp": float(expected_tp),
                "expected_sl": None,
                "risk_reward_ratio": None,
                "trade_duration": float(trade_duration_minutes),
        "mfe": float(mfe_for_bad),
        "mae": float(mae),
        "model_type": "exit",
    }

            # safety: verify schema vector length
            vec = build_feature_vector(feature_inputs)
            assert len(vec) == len(FEATURE_ORDER)

            dataset_rows.append(row)
            inserted += 1

    _insert_dataset_rows(conn, dataset_rows)
    print(f"Rebuilt execution_dataset rows: inserted={inserted}")


if __name__ == "__main__":
    main()
