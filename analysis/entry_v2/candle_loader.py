from __future__ import annotations

"""analysis/entry_v2/candle_loader.py

Candle loader for Entry v2 dataset building.

Responsibilities (strict):
- Load candles for EURUSD/GBPUSD/XAUUSD from QuantDinger/MT5 client helpers.
- Support timeframes: H4, H1, M15.
- Produce deduplicated, chronologically sorted OHLCV series.
- Do NOT compute indicators, labels, or features.
- Validate required candle fields are present.

Important: time synchronization between timeframes happens in dataset_builder.py.
This module only loads per-(symbol,timeframe) candles.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils.logger import get_logger

try:
    # Prefer existing QuantDinger candle source
    from data.market.client import get_candles  # type: ignore
except Exception:  # pragma: no cover
    get_candles = None  # type: ignore

logger = get_logger("entry_v2.candle_loader")


SUPPORTED_SYMBOLS = {"EURUSD", "GBPUSD", "XAUUSD"}
SUPPORTED_TIMEFRAMES = {"H4", "H1", "M15"}


@dataclass(frozen=True)
class Candle:
    t: float  # unix seconds
    open: float
    high: float
    low: float
    close: float
    volume: float


def _parse_time_to_unix_seconds(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            # Try ISO8601
            try:
                s2 = s.replace("Z", "+00:00")
                dt = datetime.fromisoformat(s2)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except Exception:
                pass
            # Try epoch seconds as string
            return float(s)
    except Exception:
        return None
    return None


def _validate_and_normalize_candle(raw: Dict[str, Any]) -> Optional[Candle]:
    if not isinstance(raw, dict):
        return None

    t_raw = raw.get("time") or raw.get("timestamp") or raw.get("t") or raw.get("date")

    t = _parse_time_to_unix_seconds(t_raw)
    if t is None:
        return None

    def gf(keys: Iterable[str]) -> Optional[float]:
        for k in keys:
            if k in raw and raw.get(k) is not None:
                try:
                    return float(raw.get(k))
                except Exception:
                    return None
        return None

    o = gf(["open", "o"])
    h = gf(["high", "h"])
    l = gf(["low", "l"])
    c = gf(["close", "c"])
    v = gf(["volume", "vol"])

    if o is None or h is None or l is None or c is None:
        return None
    if v is None:
        v = 0.0

    # Basic OHLC sanity
    if h < max(o, c) or l > min(o, c):
        # Allow rare rounding issues but drop clearly invalid bars
        return None

    return Candle(t=t, open=o, high=h, low=l, close=c, volume=v)


def _deduplicate_by_time(candles: List[Candle]) -> List[Candle]:
    # Keep the last occurrence for identical timestamps
    candles_sorted = sorted(candles, key=lambda x: x.t)
    out: List[Candle] = []
    last_t: Optional[float] = None
    last_idx: Optional[int] = None

    for cd in candles_sorted:
        if last_t is None or cd.t != last_t:
            out.append(cd)
            last_t = cd.t
            last_idx = len(out) - 1
        else:
            # overwrite last
            assert last_idx is not None
            out[last_idx] = cd

    return out


def load_candles(symbol: str, timeframe: str, *, count: int) -> List[Candle]:
    """Load candles from QuantDinger.

    Returns chronologically sorted candles with duplicates removed.

    Note (tolerance):
    - For production robustness, failures to load a single (symbol,timeframe)
      should be tolerated by the caller (load_required_history).
    - Therefore this function may still raise, but load_required_history will
      catch and skip failing pairs.

    Raises:
      - ValueError for unsupported symbol/timeframe
      - RuntimeError if get_candles is unavailable or returns invalid data
    """

    sym = str(symbol).strip().upper()
    tf = str(timeframe).strip().upper()

    if sym not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Unsupported symbol: {symbol}")
    if tf not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    if count <= 0:
        raise ValueError("count must be positive")
    if get_candles is None:
        raise RuntimeError("get_candles() is unavailable")

    logger.info(f"[entry_v2] loading candles symbol={sym} timeframe={tf} count={count}")
    raw = get_candles(sym, timeframe=tf, count=count)

    if raw is None:
        raise RuntimeError(f"get_candles returned None for {sym} {tf}")

    candles: List[Candle] = []

    if isinstance(raw, list):
        for item in raw:
            parsed: Optional[Candle]
            if isinstance(item, dict):
                parsed = _validate_and_normalize_candle(item)
            else:
                # allow object-like candles
                try:
                    parsed = _validate_and_normalize_candle(getattr(item, "__dict__", {}) or {})
                except Exception:
                    parsed = None
            if parsed is not None:
                candles.append(parsed)

    if not candles:
        raise RuntimeError(f"No valid candles loaded for {sym} {tf}")

    deduped = _deduplicate_by_time(candles)
    deduped = sorted(deduped, key=lambda x: x.t)

    return deduped



def load_required_history(
    symbols: List[str],
    timeframes: List[str],
    *,
    months_min: int,
) -> Dict[Tuple[str, str], List[Candle]]:
    """Load at least `months_min` of history.

    Implementation detail: we convert months_min to a conservative count per timeframe
    using approximate days (30.44). This loader is only for loading raw candles.

    dataset_builder.py performs additional validation (coverage, missing-candle checks).
    """

    days = months_min * 30.44
    # approximate bars per day
    per_day = {
        "H4": 6.0,
        "H1": 24.0,
        "M15": 96.0,
    }

    out: Dict[Tuple[str, str], List[Candle]] = {}

    loaded_summary: Dict[str, int] = {}
    skipped_summary: List[str] = []

    for sym in symbols:
        for tf in timeframes:
            tf_u = tf.strip().upper()
            if tf_u not in per_day:
                raise ValueError(f"Unsupported timeframe for count calc: {tf}")
            count = int(days * per_day[tf_u])
            # add warm buffer; indicators need warm-up anyway
            count = max(count, 500)
            key = (sym.strip().upper(), tf_u)

            try:
                candles = load_candles(sym, tf_u, count=count)
                out[key] = candles
                loaded_summary[f"{key[0]}_{key[1]}"] = len(candles)
            except Exception as e:
                logger.warning(f"⚠️ تخطي {key[0]} {key[1]}: لا توجد بيانات ({e})")
                skipped_summary.append(f"{key[0]}_{key[1]}: {e}")

    total_candles = sum(len(v) for v in out.values())
    if total_candles <= 0:
        raise RuntimeError(
            "Critical: no candles loaded for any symbol/timeframe (total=0). "
            "Check QuantDinger/MT5 connectivity and symbol availability."
        )

    # Log final summary
    logger.info("[entry_v2] candle_loader load_required_history summary:")
    for k, n in sorted(loaded_summary.items()):
        logger.info("  ✅ loaded %s bars=%d", k, n)
    for item in skipped_summary[:50]:
        logger.info("  ⛔ skipped %s", item)
    if len(skipped_summary) > 50:
        logger.info("  ... and %d more skipped", len(skipped_summary) - 50)

    return out


