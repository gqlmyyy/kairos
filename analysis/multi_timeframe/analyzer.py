# Trading Bot V3 - analysis/multi_timeframe/analyzer.py
# Multi-timeframe analysis (H4 + H1 + M15)

from utils.logger import get_logger
from config import TF_TREND, TF_DECISION, TF_TIMING
from data.market.hybrid_client import get_candles, get_indicators_hybrid as get_indicators

from analysis.technical.indicators import get_trend_score, get_momentum_score
from core.models import MultiTimeframeData

logger = get_logger("mtf")

def get_multi_timeframe_analysis(symbol: str) -> MultiTimeframeData:
    """Analyze H4 (trend), H1 (decision), M15 (timing)"""

    # H4 - Trend direction
    h4_score, h4_dir = get_trend_score(symbol)

    # H1 - Momentum / decision
    h1_score, h1_dir = get_momentum_score(symbol)

    # M15 - Timing (short-term momentum)
    m15_score, m15_dir = get_momentum_score(symbol, timeframe=TF_TIMING)


    # Check alignment

    directions = [h4_dir, h1_dir, m15_dir]
    non_neutral = [d for d in directions if d != "neutral"]

    if len(non_neutral) == 3 and all(d == non_neutral[0] for d in non_neutral):
        aligned = True
        strength = "strong"
    elif len(non_neutral) >= 2 and all(d == non_neutral[0] for d in non_neutral):
        aligned = True
        strength = "moderate"
    elif h4_dir != "neutral" and h1_dir == h4_dir:
        aligned = True
        strength = "moderate"
    elif h1_dir != "neutral" and m15_dir == h1_dir:
        aligned = True
        strength = "weak"
    else:
        aligned = False
        strength = "weak"

    mtf = MultiTimeframeData(
        h4_direction=h4_dir, h4_score=h4_score,
        h1_direction=h1_dir, h1_score=h1_score,
        m15_direction=m15_dir, m15_score=m15_score,
        aligned=aligned,
        strength=strength
    )

    # Verification Proof logs (no assumptions): print per-timeframe bars + last values.
    def _last_close(candles):
        if not candles:
            return 0.0
        last = candles[-1]
        try:
            return float(last.get("close", last.get("c", 0)) or 0)
        except Exception:
            return 0.0

    def _last_ts(candles):
        if not candles:
            return ""
        last = candles[-1]
        # common keys: time / timestamp
        for k in ("time", "timestamp", "t"):
            if k in last:
                return str(last.get(k))
        return ""

    tf_bars = {
        "H4": get_candles(symbol, timeframe=TF_TREND, count=100),
        "H1": get_candles(symbol, timeframe=TF_DECISION, count=100),
        "M15": get_candles(symbol, timeframe=TF_TIMING, count=100),
    }

    # Indicators for each timeframe (RSI/ATR/MACD)
    ind_h4 = get_indicators(symbol, timeframe=TF_TREND) or {}
    ind_h1 = get_indicators(symbol, timeframe=TF_DECISION) or {}
    ind_m15 = get_indicators(symbol, timeframe=TF_TIMING) or {}

    logger.info(
        f"MTF_VERIFY {symbol}: "
        f"TIMEFRAME=H4 candles={len(tf_bars['H4'])} last_ts={_last_ts(tf_bars['H4'])} last_close={_last_close(tf_bars['H4'])} "
        f"RSI={ind_h4.get('rsi', 'NA')} ATR={ind_h4.get('atr', 'NA')} MACD={ind_h4.get('macd', 'NA')} | "
        f"TIMEFRAME=H1 candles={len(tf_bars['H1'])} last_ts={_last_ts(tf_bars['H1'])} last_close={_last_close(tf_bars['H1'])} "
        f"RSI={ind_h1.get('rsi', 'NA')} ATR={ind_h1.get('atr', 'NA')} MACD={ind_h1.get('macd', 'NA')} | "
        f"TIMEFRAME=M15 candles={len(tf_bars['M15'])} last_ts={_last_ts(tf_bars['M15'])} last_close={_last_close(tf_bars['M15'])} "
        f"RSI={ind_m15.get('rsi', 'NA')} ATR={ind_m15.get('atr', 'NA')} MACD={ind_m15.get('macd', 'NA')}"
    )

    logger.info(f"MTF {symbol}: H4={h4_dir} H1={h1_dir} M15={m15_dir} aligned={aligned} ({strength})")

    return mtf

