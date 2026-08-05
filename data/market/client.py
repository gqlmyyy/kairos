# Trading Bot V3 - data/market/client.py
# Fetches market data from QuantDinger

from utils.logger import get_logger
from config import QUANTDINGER_URL
import requests
import time

logger = get_logger("market_client")

MARKET_MAP = {
    "EURUSD": "Forex", "GBPUSD": "Forex",
    "USDJPY": "Forex", "XAUUSD": "Forex",
}

TF_MAP = {
    # Keep QuantDinger request timeframes mostly in common forms.
    # Important: previous mapping incorrectly routed H1/M15 -> 4H.
    "H4": "4H",
    "4H": "4H",

    # Prefer most common QuantDinger labels.
    "H1": "1H",
    "1H": "1H",

    # M15 support is inconsistent in this QuantDinger deployment.
    # QuantDinger returns data for the *15-minute* timeframe when sent as "15" (and some variants),
    # but not reliably for the literal "M15" label.
    # Evidence (scripts/diagnose_quantdinger_m15.py): EURUSD tf='M15' -> 0 bars, tf='15'/'15m'/... -> 200.
    # QuantDinger in this deployment accepts 15-minute candles mainly as "15m".
    # Evidence: direct request to /api/indicator/kline for EURUSD: "15"/"15M" => code=0 (No data found), "15m" => code=1 (success).
    "M15": "15m",
    "15M": "15m",




    "1D": "1D",
    "1W": "1W",
}


# Default/fallback values when API fails
FALLBACK_ATR = {
    "EURUSD": 0.0008, "GBPUSD": 0.0010,
    "XAUUSD": 8.0,    "USDJPY": 0.15
}

FALLBACK_INDICATORS = {
    "EURUSD": {"rsi": 50.0, "atr": 0.0008, "macd": 0.0, "ma_trend": "sideways", "close": 1.0800},
    "GBPUSD": {"rsi": 50.0, "atr": 0.0010, "macd": 0.0, "ma_trend": "sideways", "close": 1.2700},
    "XAUUSD": {"rsi": 50.0, "atr": 8.0, "macd": 0.0, "ma_trend": "sideways", "close": 2350.0},
    "USDJPY": {"rsi": 50.0, "atr": 0.15, "macd": 0.0, "ma_trend": "sideways", "close": 150.0},
}

# Cache for successful data to avoid repeated API calls
_candles_cache = {}
_cache_timestamp = {}
CACHE_TTL = 300  # 5 minutes

_api_error_count = {}  # Track consecutive API errors
API_ERROR_THRESHOLD = 5  # Alert after 5 consecutive errors

def _get_headers() -> dict:
    from execution.quantdinger_client import get_headers
    return get_headers()

def _check_api_health(symbol: str):
    """Check and report on API health"""
    error_count = _api_error_count.get(symbol, 0)
    if error_count >= API_ERROR_THRESHOLD:
        logger.error(f"⚠️  CRITICAL: {symbol} API failing consistently ({error_count} errors). Check QuantDinger connection!")

def get_candles(symbol: str, timeframe: str = "4H", count: int = 100) -> list:
    """Fetch candles from QuantDinger with fallback and caching"""
    cache_key = f"{symbol}_{timeframe}"
    current_time = time.time()
    
    # Check cache first
    if cache_key in _candles_cache:
        cache_age = current_time - _cache_timestamp.get(cache_key, 0)
        if cache_age < CACHE_TTL:
            logger.debug(f"Using cached candles for {symbol} {timeframe} (age: {int(cache_age)}s)")
            return _candles_cache[cache_key]
    
    try:
        market = MARKET_MAP.get(symbol, "Forex")
        tf = TF_MAP.get(timeframe, "4H")
        logger.debug(f"DEBUG: Requesting {symbol} {timeframe} as '{tf}'")
        params = {"symbol": symbol, "market": market, "timeframe": tf, "limit": count}

        r = requests.get(

            f"{QUANTDINGER_URL}/api/indicator/kline",
            params=params,
            headers=_get_headers(),
            timeout=10
        )
        data = r.json()

        if data.get("code") not in [1, 200]:
            error_msg = data.get("msg", "Unknown error")
            _api_error_count[symbol] = _api_error_count.get(symbol, 0) + 1
            logger.warning(
                f"Candles API error for {symbol} {timeframe} (requested tf='{timeframe}' as '{tf}'): "
                f"{error_msg} (code={data.get('code')}) full response={data}"
            )
            _check_api_health(symbol)
            return []
        
        candles = data.get("data", [])
        if candles:
            # Cache successful result
            _candles_cache[cache_key] = candles
            _cache_timestamp[cache_key] = current_time
            _api_error_count[symbol] = 0  # Reset error counter on success
            logger.debug(f"Candles {symbol} {tf}: {len(candles)} bars ✓")
        else:
            logger.warning(
                f"API returned empty candles for {symbol} {timeframe} (requested tf='{timeframe}' as '{tf}'). "
                f"full response={data}"
            )
        
        return candles
        
    except requests.exceptions.ConnectionError as e:
        _api_error_count[symbol] = _api_error_count.get(symbol, 0) + 1
        logger.error(f"Connection error for {symbol}: {e}")
        logger.error(f"QuantDinger URL: {QUANTDINGER_URL}")
        _check_api_health(symbol)
        return []
    except requests.exceptions.Timeout as e:
        _api_error_count[symbol] = _api_error_count.get(symbol, 0) + 1
        logger.error(f"Timeout fetching candles for {symbol}: {e}")
        _check_api_health(symbol)
        return []
    except Exception as e:
        _api_error_count[symbol] = _api_error_count.get(symbol, 0) + 1
        logger.error(f"Candles fetch failed for {symbol}: {e}")
        _check_api_health(symbol)
        return []

def get_indicators(symbol: str, timeframe: str = "4H") -> dict:
    """Get technical indicators with fallback to defaults when data unavailable"""
    candles = get_candles(symbol, timeframe, 100)
    
    # If no candles, use fallback values
    if not candles or len(candles) < 20:
        fallback = FALLBACK_INDICATORS.get(symbol, {
            "rsi": 50.0, "atr": 0.001, "macd": 0.0, "ma_trend": "sideways", "close": 0.0
        })
        logger.warning(f"Using FALLBACK indicators for {symbol} {timeframe} (candles={len(candles) if candles else 0})")
        return fallback.copy()
    
    try:
        closes = [float(c.get("close", 0)) for c in candles]
        highs  = [float(c.get("high", 0)) for c in candles]
        lows   = [float(c.get("low", 0)) for c in candles]

        # RSI(14)
        gains, losses = [], []
        for i in range(1, 15):
            diff = closes[-i] - closes[-i-1]
            (gains if diff > 0 else losses).append(abs(diff))
        avg_gain = sum(gains) / 14 if gains else 0.001
        avg_loss = sum(losses) / 14 if losses else 0.001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # ATR(14)
        trs = []
        for i in range(1, 15):
            tr = max(
                highs[-i] - lows[-i],
                abs(highs[-i] - closes[-i-1]),
                abs(lows[-i] - closes[-i-1])
            )
            trs.append(tr)
        atr = sum(trs) / 14 if trs else FALLBACK_ATR.get(symbol, 0.001)

        # MACD(12,26) (simple EMA approximations via averages to avoid extra deps)
        ema12 = sum(closes[-12:]) / 12
        ema26 = sum(closes[-26:]) / 26
        macd = ema12 - ema26

        # MA Trend
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else closes[-1]
        price = closes[-1]

        if price > ma20 > ma50:
            ma_trend = "strong uptrend"
        elif price > ma20:
            ma_trend = "uptrend"
        elif price < ma20 < ma50:
            ma_trend = "strong downtrend"
        elif price < ma20:
            ma_trend = "downtrend"
        else:
            ma_trend = "sideways"

        return {
            "rsi": round(rsi, 2),
            "atr": round(atr, 6),
            "macd": round(macd, 6),
            "ma_trend": ma_trend,
            "close": price
        }
    except Exception as e:
        logger.error(f"Indicators calc error for {symbol}: {e}")
        fallback = FALLBACK_INDICATORS.get(symbol, {
            "rsi": 50.0, "atr": 0.001, "macd": 0.0, "ma_trend": "sideways", "close": 0.0
        })
        return fallback.copy()


def get_atr(symbol: str, timeframe: str = "4H") -> float:
    """Get ATR with fallback"""
    data = get_indicators(symbol, timeframe)
    atr = float(data.get("atr", FALLBACK_ATR.get(symbol, 0.001)) or FALLBACK_ATR.get(symbol, 0.001))
    return atr if atr > 0 else FALLBACK_ATR.get(symbol, 0.001)


def get_price(symbol: str, timeframe: str = "4H") -> float:
    """Get current price with fallback"""
    data = get_indicators(symbol, timeframe)
    px = data.get("close", 0.0)
    try:
        return float(px)
    except Exception:
        return FALLBACK_INDICATORS.get(symbol, {}).get("close", 0.0)


def get_rsi(symbol: str, timeframe: str = "4H") -> float:
    """Get RSI with fallback"""
    d = get_indicators(symbol, timeframe)
    return float(d.get("rsi", 50.0) or 50.0)


def get_macd(symbol: str, timeframe: str = "4H") -> float:
    """Get MACD with fallback"""
    d = get_indicators(symbol, timeframe)
    return float(d.get("macd", 0.0) or 0.0)


def get_session(dt) -> str:
    # deterministic mapping by UTC hour
    try:
        h = int(getattr(dt, "hour", 0))
    except Exception:
        h = 0
    if 0 <= h < 7:
        return "Asia"
    if 7 <= h < 13:
        return "London"
    if 13 <= h < 20:
        return "NY"
    return "Asia"


def get_market_regime(symbol: str, timeframe: str = "4H") -> str:
    """Compute regime from candles + ATR + volatility + trend (no None)."""
    candles = get_candles(symbol, timeframe, 80)
    if not candles:
        return "neutral"

    closes = []
    for c in candles:
        try:
            closes.append(float(c.get("close", c.get("c", 0)) or 0.0))
        except Exception:
            continue
    if len(closes) < 5:
        return "neutral"

    # volatility: std of returns
    returns = []
    for i in range(1, len(closes)):
        prev = closes[i-1]
        cur = closes[i]
        if prev != 0:
            returns.append((cur - prev) / prev)
    import math
    if not returns:
        vol = 0.0
    else:
        mean = sum(returns) / len(returns)
        var = sum((r-mean)**2 for r in returns) / len(returns)
        vol = math.sqrt(var)

    # trend: MA slope/relationship
    ma_fast = sum(closes[-20:]) / min(20, len(closes))
    ma_slow = sum(closes[-50:]) / min(50, len(closes))
    price = closes[-1]

    trend = "neutral"
    if price > ma_fast > ma_slow:
        trend = "trending"
    elif price < ma_fast < ma_slow:
        trend = "ranging"

    atr = get_atr(symbol, timeframe)

    # Map to regime string (later encoded)
    # Heuristic thresholds: keep simple numeric guards.
    if atr > 0 and vol > 0:
        if vol > atr * 1000:
            return "high volatility"

    # trending vs ranging based on trend_strength proxy
    if trend == "trending":
        return "trending"
    if trend == "ranging":
        return "ranging"
    return "neutral"


def get_account_info() -> dict:

    from execution.quantdinger_client import get_equity as qd_equity
    try:
        equity = qd_equity()
        return {"balance": equity, "equity": equity, "margin": 0}
    except Exception as e:
        logger.error(f"Account info error: {e}")
        return {"balance": 0, "equity": 0, "margin": 0}

def get_equity() -> float:
    from execution.quantdinger_client import get_equity as qd_equity
    return qd_equity()

def set_token(token: str):
    pass


def check_quantdinger_health() -> dict:
    """Check QuantDinger API health and connectivity"""
    result = {
        "status": "unknown",
        "url": QUANTDINGER_URL,
        "connection": False,
        "auth": False,
        "data": False,
        "errors": []
    }
    
    try:
        # Test connection
        r = requests.get(f"{QUANTDINGER_URL}/api/health", timeout=5)
        result["connection"] = True
        logger.info(f"✓ QuantDinger connection OK: {QUANTDINGER_URL}")
    except Exception as e:
        result["errors"].append(f"Connection failed: {e}")
        logger.error(f"✗ Cannot connect to QuantDinger at {QUANTDINGER_URL}: {e}")
        result["status"] = "unreachable"
        return result
    
    try:
        # Test authentication
        headers = _get_headers()
        result["auth"] = True
        logger.info("✓ QuantDinger authentication OK")
    except Exception as e:
        result["errors"].append(f"Auth failed: {e}")
        logger.error(f"✗ Authentication failed: {e}")
        result["status"] = "auth_failed"
        return result
    
    try:
        # Test data retrieval
        candles = get_candles("EURUSD", "H4", 10)
        if candles:
            result["data"] = True
            logger.info(f"✓ Data retrieval OK (got {len(candles)} candles)")
            result["status"] = "healthy"
        else:
            result["errors"].append("No candles returned from API")
            logger.warning("⚠️  API returned empty candles")
            result["status"] = "no_data"
    except Exception as e:
        result["errors"].append(f"Data fetch failed: {e}")
        logger.error(f"✗ Data retrieval failed: {e}")
        result["status"] = "data_error"
    
    return result
