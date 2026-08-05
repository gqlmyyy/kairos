from execution.quantdinger_client import open_trade

# Debug helper: calls open_trade but intercepts logs only.
# This does NOT guarantee broker connectivity.

if __name__ == "__main__":
    res = open_trade(
        symbol="EURUSD",
        direction="BUY",
        size=0.01,
        sl=1.095,
        tp=1.105,
        reason="test_filling_mode",
    )
    print("result:", res)

