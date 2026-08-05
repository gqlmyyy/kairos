from __future__ import annotations

import math

from data.storage.database import get_conn, get_execution_dataset, upsert_execution_expected
from data.market.client import get_macd, get_rsi, get_atr, get_price, get_market_regime


def _is_noneish(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def main():
    conn = get_conn()
    c = conn.cursor()

    rows = c.execute(
        """SELECT order_id, symbol, direction, status, expected_entry, expected_rsi, expected_macd, expected_atr, expected_session,
                   expected_trend_strength, expected_momentum_score, expected_volatility_score, expected_market_regime,
                   expected_ai_score, expected_sentiment_score, expected_news_impact_score,
                   expected_ai_confidence, expected_trend_score, expected_final_score, expected_trend_score,
                   expected_spread, expected_final_score
            FROM execution_dataset
            WHERE status IN ('open','closed') AND expected_macd IS NULL"""
    ).fetchall()

    total = len(rows)
    print("rows_to_repair", total)
    repaired = 0
    skipped = 0

    for r in rows:
        order_id = r["order_id"]
        symbol = r["symbol"]
        direction = r["direction"]
        try:
            # Re-extract from QuantDinger endpoints (real data)
            macd = get_macd(symbol)
            rsi = get_rsi(symbol)
            atr = get_atr(symbol)
            entry_price = r["expected_entry"]

            regime = get_market_regime(symbol)
            spread = r["expected_spread"]

            # If any of these comes back as None (shouldn't, client returns 0.0)
            if _is_noneish(macd) or _is_noneish(rsi) or _is_noneish(atr):
                skipped += 1
                print("skip", order_id, "macd/rsi/atr is noneish")
                continue

            # Build minimal upsert payload: keep all existing expected_* except macd/rsi/atr/regime/session/spread
            # NOTE: upsert_execution_expected requires many args; we pass existing columns from row.
            upsert_execution_expected(
                order_id=order_id,
                symbol=symbol,
                direction=direction,
                expected_entry=entry_price,
                expected_final_score=r["expected_final_score"],

                expected_ai_score=r["expected_ai_score"],
                expected_ai_confidence=r["expected_ai_confidence"],
                expected_trend_score=r["expected_trend_score"],
                expected_momentum_score=r["expected_momentum_score"],
                expected_sentiment_score=r["expected_sentiment_score"],
                expected_volatility_score=r["expected_volatility_score"],

                expected_rsi=float(rsi) / 100.0 if rsi is not None and rsi != 0 else r["expected_rsi"],
                expected_macd=float(macd),
                expected_session=r["expected_session"],
                expected_spread=spread,
                expected_atr=float(atr) if atr is not None else r["expected_atr"],
                expected_trend_strength=r["expected_trend_strength"],
                expected_market_regime=regime,
                expected_news_impact_score=r["expected_news_impact_score"],

                expected_indicators_json=None,
                strategy="V3-repair"
            )

            repaired += 1
            print("repaired", order_id, "symbol", symbol, "macd", macd, "rsi", rsi, "atr", atr, "regime", regime)

        except Exception as e:
            skipped += 1
            print("repair_failed", order_id, "err", e)

    conn.close()
    print("done", {"total": total, "repaired": repaired, "skipped": skipped})


if __name__ == "__main__":
    main()

