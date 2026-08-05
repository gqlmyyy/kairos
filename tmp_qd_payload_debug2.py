from execution.quantdinger_client import open_trade

if __name__ == "__main__":
    res = open_trade(
        symbol="EURUSD",
        direction="BUY",
        size=0.01,
        sl=1.095,
        tp=1.105,
        reason="test_filling_mode_more_candidates",
    )
    print("result:", res)

