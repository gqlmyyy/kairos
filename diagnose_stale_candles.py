"""Read-only diagnostic: proves whether QuantDinger's candle feed is advancing.

Safe to run anytime — makes two GET requests to the kline endpoint a few
minutes apart and compares the returned bar timestamps. Does not touch any
trading logic, does not open/close positions.

Usage:  python diagnose_stale_candles.py
"""
import time
from datetime import datetime, timezone

from execution.quantdinger_client import get_headers
from config import QUANTDINGER_URL
import requests

SYMBOLS = ["EURUSD", "XAUUSD", "GBPUSD"]
WAIT_SECONDS = 120  # gap between the two probes


def fetch_last_bar(symbol: str):
    r = requests.get(
        f"{QUANTDINGER_URL}/api/indicator/kline",
        params={"symbol": symbol, "market": "Forex", "timeframe": "1H", "limit": 5},
        headers=get_headers(),
        timeout=10,
    )
    data = r.json()
    bars = data.get("data", [])
    if not bars:
        return None, data
    last = bars[-1]
    # try common time field names
    ts = last.get("time") or last.get("t") or last.get("timestamp")
    return (ts, last.get("close") or last.get("c")), data.get("code")


def main():
    print(f"=== فحص أول: {datetime.now(timezone.utc).isoformat()} ===")
    first = {}
    for s in SYMBOLS:
        (ts, close), code = fetch_last_bar(s)
        first[s] = (ts, close)
        print(f"  {s:8s} code={code} last_bar_time={ts} close={close}")

    print(f"\nانتظار {WAIT_SECONDS}s ثم إعادة الفحص...")
    time.sleep(WAIT_SECONDS)

    print(f"\n=== فحص ثانٍ: {datetime.now(timezone.utc).isoformat()} ===")
    for s in SYMBOLS:
        (ts, close), code = fetch_last_bar(s)
        prev_ts, prev_close = first[s]
        changed = (ts != prev_ts) or (close != prev_close)
        flag = "✓ تحدّثت" if changed else "✗ لم تتغيّر إطلاقاً"
        print(f"  {s:8s} code={code} last_bar_time={ts} close={close}  {flag}")

    print(
        "\nإذا ظهرت '✗ لم تتغيّر إطلاقاً' لكل الرموز رغم فارق دقيقتين، "
        "فالمشكلة مؤكدة في خادم QuantDinger نفسه (أو اتصاله بالبروكر) "
        "وليست في كود البوت."
    )


if __name__ == "__main__":
    main()
