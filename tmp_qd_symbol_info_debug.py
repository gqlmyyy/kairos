import requests
from config import QUANTDINGER_URL, QUANTDINGER_USERNAME, QUANTDINGER_PASSWORD
from execution.quantdinger_client import get_headers

if __name__ == "__main__":
    symbol = "EURUSD"
    # Try symbol_info endpoint
    r = requests.get(
        f"{QUANTDINGER_URL}/api/mt5/symbol_info",
        headers=get_headers(),
        params={"symbol": symbol},
        timeout=10,
    )
    print("status_code:", getattr(r, "status_code", None))
    try:
        print("json:", r.json())
    except Exception:
        print("text:", r.text[:1000])

