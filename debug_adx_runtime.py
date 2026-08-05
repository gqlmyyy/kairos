import traceback
import numpy as np

from data.market.client import get_candles
from scripts.import_historical_trades import calculate_adx


def _build_bars_for_calculate_adx(candles):
    """
    Build numpy ndarray with columns compatible with calculate_adx:
      bars[:,0]=time (any numeric)
      bars[:,1]=open
      bars[:,2]=high
      bars[:,3]=low
      bars[:,4]=close
    """
    arr = []
    if candles is None:
        return np.array([], dtype=float)

    for c in candles:
        if isinstance(c, dict):
            t = c.get("time") or c.get("timestamp") or c.get("t") or c.get("date") or 0
            o = c.get("open", None)
            if o is None:
                o = c.get("o", 0)
            h = c.get("high", None)
            if h is None:
                h = c.get("h", 0)
            l = c.get("low", None)
            if l is None:
                l = c.get("l", 0)
            cl = c.get("close", None)
            if cl is None:
                cl = c.get("c", 0)
            arr.append([t, o, h, l, cl])
        else:
            # namedtuple / object
            t = getattr(c, "time", None) or getattr(c, "timestamp", None) or getattr(c, "t", None) or 0
            o = getattr(c, "open", None) or getattr(c, "o", None) or 0
            h = getattr(c, "high", None) or getattr(c, "h", None) or 0
            l = getattr(c, "low", None) or getattr(c, "l", None) or 0
            cl = getattr(c, "close", None) or getattr(c, "c", None) or 0
            arr.append([t, o, h, l, cl])

    return np.array(arr, dtype=float)


def main():
    symbol = "EURUSD"
    timeframe = "H4"
    count = 100
    period = 14

    try:
        candles = get_candles(symbol, timeframe=timeframe, count=count)

        print("type(candles) =", type(candles))

        bars = _build_bars_for_calculate_adx(candles)
        print("type(bars) =", type(bars))
        print("bars.shape =", getattr(bars, "shape", None))
        print("bars.dtype =", getattr(bars, "dtype", None))

        print("bars[:5] =")
        print(bars[:5])
        print("bars[-5:] =")
        print(bars[-5:])

        # Column mapping (explicit)
        print("Column mapping:")
        print("Column0(Time)  = bars[:,0]")
        print("Column1(Open)  = bars[:,1]")
        print("Column2(High)  = bars[:,2]")
        print("Column3(Low)   = bars[:,3]")
        print("Column4(Close) = bars[:,4]")
        print("Volume = not used by calculate_adx (expects OHLC only)")

        # Validate that calculate_adx reads expected columns
        highs = bars[:, 2]
        lows = bars[:, 3]
        closes = bars[:, 4]

        # Quick sanity checks: print min/max for each
        print("High sanity: min/max =", float(np.nanmin(highs)), float(np.nanmax(highs)))
        print("Low sanity:  min/max =", float(np.nanmin(lows)), float(np.nanmax(lows)))
        print("Close sanity: min/max =", float(np.nanmin(closes)), float(np.nanmax(closes)))

        # Run calculate_adx on the same bars
        adx = calculate_adx(bars, period=period, symbol=symbol)
        print("ADX =", adx)

    except Exception as e:
        print("Exception occurred:", repr(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()
