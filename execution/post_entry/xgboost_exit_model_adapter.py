from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

from utils.logger import get_logger
from data.storage.database import update_execution_mfe_mae
from data.market.hybrid_client import get_indicators_hybrid
from data.market.client import get_candles
from scripts.import_historical_trades import calculate_adx

from datetime import datetime, timezone


from analysis.models.xgboost_exit_model import predict_exit_probability
from analysis.models.feature_schema import FEATURE_ORDER, build_feature_vector

logger = get_logger("xgboost_exit_adapter")


@dataclass
class _CacheEntry:
    indicators: Dict[str, Any]
    last_success_ts: float


class XGBoostExitModelAdapter:
    """Returns probabilities only; does not modify MT5 nor telegram."""

    # Must match training TFs and expected_* columns.
    EXIT_MODEL_TIMEFRAME = "H4"

    # Retry: attempt #1 + one extra after timeout
    QD_TIMEOUT_SEC = 6.0
    QD_RETRY_DELAY_SEC = 0.2
    QD_MAX_ATTEMPTS = 2

    # Throttle/cache: 5 seconds per (symbol,timeframe)
    INDICATORS_CACHE_TTL_SEC = 5.0

    # CRITICAL escalation thresholds
    CRITICAL_N_CONSECUTIVE_FEATURE_INCOMPLETE = 6
    CRITICAL_TOTAL_STALE_SEC = 30.0

    def __init__(self) -> None:
        # (symbol,timeframe) -> last successful indicators
        self._indicators_cache: Dict[Tuple[str, str], _CacheEntry] = {}
        self._indicators_cache_lock = threading.Lock()

        # per-symbol failure counters for CRITICAL
        self._consecutive_incomplete_by_symbol: Dict[str, int] = {}
        self._last_success_ts_by_symbol: Dict[str, float] = {}

        # ADX cache per symbol
        self._adx_cache: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._adx_cache_lock = threading.Lock()
        self._adx_cache_ttl_sec = 5.0

        # Last known valid ATR for H4 (fallback when QuantDinger returns empty for H4)
        self._last_valid_atr_by_symbol: Dict[str, float] = {}

    # -----------------------------
    # Keep existing MFE/MAE persistence
    # -----------------------------
    def update_mfe_mae(
        self,
        position_state: Optional[Any],
        current_profit: float,
        order_id: Optional[Any] = None,
    ) -> tuple[float, float]:
        if position_state is None:
            return 0.0, 0.0

        state_mfe = getattr(position_state, "mfe", 0.0)
        state_mae = getattr(position_state, "mae", 0.0)

        mfe = state_mfe
        mae = state_mae

        if current_profit > state_mfe:
            mfe = current_profit
            position_state.mfe = mfe
        else:
            mfe = state_mfe

        if current_profit < state_mae:
            mae = current_profit
            position_state.mae = mae
        else:
            mae = state_mae

        if order_id is not None:
            try:
                update_execution_mfe_mae(order_id=str(order_id), mfe=float(mfe), mae=float(mae))
            except Exception:
                pass

        return float(mfe), float(mae)

    # -----------------------------
    # QuantDinger-only indicator fetch
    # -----------------------------
    def get_indicators_from_quantdinger(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        sym = str(symbol).strip()
        tf = str(timeframe).strip()
        if not sym:
            raise ValueError("symbol is empty")
        if not tf:
            raise ValueError("timeframe is empty")

        cache_key = (sym, tf)
        now_ts = time.time()

        # throttle/cache
        with self._indicators_cache_lock:
            entry = self._indicators_cache.get(cache_key)
            if entry is not None and (now_ts - entry.last_success_ts) < self.INDICATORS_CACHE_TTL_SEC:
                return entry.indicators

        last_err: Optional[Exception] = None

        for attempt in range(1, self.QD_MAX_ATTEMPTS + 1):
            try:
                result_container: Dict[str, Any] = {}
                exc_container: Dict[str, Exception] = {}

                def _worker() -> None:
                    try:
                        result_container["indicators"] = get_indicators_hybrid(sym, timeframe=tf)
                    except Exception as e:
                        exc_container["exc"] = e

                th = threading.Thread(target=_worker, daemon=True)
                th.start()
                th.join(timeout=self.QD_TIMEOUT_SEC)

                if th.is_alive():
                    raise TimeoutError(f"QuantDinger get_indicators_hybrid timeout after {self.QD_TIMEOUT_SEC}s")

                if "exc" in exc_container:
                    raise exc_container["exc"]

                indicators = result_container.get("indicators") or {}
                if not indicators:
                    raise RuntimeError(f"QuantDinger returned empty indicators for {sym} {tf}")

                # explicitly guard placeholder behavior
                if indicators.get("rsi") == 50.0:
                    raise RuntimeError(f"QuantDinger returned RSI placeholder=50.0 for {sym} {tf}")

                with self._indicators_cache_lock:
                    self._indicators_cache[cache_key] = _CacheEntry(
                        indicators=indicators,
                        last_success_ts=time.time(),
                    )

                return indicators

            except Exception as e:
                last_err = e
                if attempt < self.QD_MAX_ATTEMPTS:
                    time.sleep(self.QD_RETRY_DELAY_SEC)
                    continue

        raise RuntimeError(f"QuantDinger indicators failed for {sym} {tf}: {last_err}")

    # -----------------------------
    # ADX from candles (QuantDinger candles only)
    # -----------------------------
    def _compute_adx_from_candles_h4(self, symbol: str, period: int = 14) -> float:
        sym = str(symbol).strip()
        if not sym:
            raise ValueError("symbol is empty")

        cache_key = (sym, "ADX_H4")
        now_ts = time.time()

        with self._adx_cache_lock:
            cached = self._adx_cache.get(cache_key)
            if cached is not None:
                cached_ts, cached_val = cached
                if (now_ts - cached_ts) < self._adx_cache_ttl_sec:
                    print(
                        f"[ADX_RT_TRACE] cache_hit symbol={sym} timeframe=H4 count=50 -> returned={cached_val}"
                    )
                    return cached_val

        # Cache miss: fetch directly from QuantDinger
        print(
            f"[ADX_RT_TRACE] cache_miss symbol={sym} timeframe=H4 count=50 period={period} (fetching candles)"
        )
        candles = get_candles(sym, timeframe="H4", count=50)
        print(
            f"[ADX_RT_TRACE] candles_fetched len(candles)={len(candles) if candles is not None else None} type={type(candles)}"
        )

        # Convert QuantDinger candles list -> numpy.ndarray bars with the exact mapping:
        # Column0=Time, Column1=Open, Column2=High, Column3=Low, Column4=Close
        import numpy as np  # local import to avoid affecting module globals

        arr = []
        if candles is None:
            raise RuntimeError("QuantDinger returned candles=None for ADX")
        if isinstance(candles, list) and len(candles) > 0:
            sample0 = candles[0]
            if isinstance(sample0, dict):
                for c in candles:
                    t = c.get("time") or c.get("timestamp") or c.get("t") or c.get("date")
                    o = c.get("open") or c.get("o")
                    h = c.get("high") or c.get("h")
                    l = c.get("low") or c.get("l")
                    cl = c.get("close") or c.get("c")
                    arr.append([t, o, h, l, cl])
            else:
                for c in candles:
                    t = getattr(c, "time", None) or getattr(c, "timestamp", None) or getattr(c, "t", None) or getattr(c, "date", None)
                    o = getattr(c, "open", None) or getattr(c, "o", None)
                    h = getattr(c, "high", None) or getattr(c, "h", None)
                    l = getattr(c, "low", None) or getattr(c, "l", None)
                    cl = getattr(c, "close", None) or getattr(c, "c", None)
                    arr.append([t, o, h, l, cl])

        bars = np.array(arr, dtype=float)
        print(f"[ADX_RT_TRACE] bars.shape={bars.shape} bars.dtype={bars.dtype}")

        # Call calculate_adx with numpy.ndarray bars.
        # Keep signature usage compatible: calculate_adx(bars, period=period)
        adx_val = calculate_adx(bars, period=period)
        print(f"[ADX_RT_TRACE] raw_adx={adx_val} (returned by calculate_adx)")

        adx_f = float(adx_val) if adx_val is not None else 0.0
        print(f"[ADX_RT_TRACE] adx_f={adx_f} (after float conversion/none->0)")

        # Trace subsequent usage by returning path only (no logic changes)
        with self._adx_cache_lock:
            self._adx_cache[cache_key] = (time.time(), adx_f)

        print(f"[ADX_RT_TRACE] return adx_f={adx_f}")
        return adx_f

    # -----------------------------
    # CRITICAL escalation
    # -----------------------------
    def _maybe_critical_log(self, symbol: str, timeframe: str, reason: str, features_incomplete: bool) -> None:
        if not features_incomplete:
            return

        sym = str(symbol).strip()
        consecutive = self._consecutive_incomplete_by_symbol.get(sym, 0)
        last_ok = self._last_success_ts_by_symbol.get(sym, 0.0)
        stale_for = time.time() - last_ok if last_ok else float("inf")

        if consecutive >= self.CRITICAL_N_CONSECUTIVE_FEATURE_INCOMPLETE or stale_for >= self.CRITICAL_TOTAL_STALE_SEC:
            logger.critical(
                f"[EXIT_MODEL][CRITICAL] features_incomplete persists: symbol={sym} timeframe={timeframe} "
                f"consecutive={consecutive} stale_for_sec={stale_for:.1f} reason={reason}"
            )

    # -----------------------------
    # time_open -> trade_duration minutes
    # -----------------------------
    def _time_open_to_minutes(self, time_open_raw: Any) -> Optional[float]:
        """Return minutes since entry until now.

        live trade uses entry time only; exit_time is not available.
        """
        if time_open_raw is None:
            return None

        now_ts = time.time()

        try:
            if hasattr(time_open_raw, "timestamp"):
                ts_open = float(time_open_raw.timestamp())
                return max(0.0, (now_ts - ts_open) / 60.0)

            if isinstance(time_open_raw, (int, float)):
                ts_open = float(time_open_raw)
                return max(0.0, (now_ts - ts_open) / 60.0)

            if isinstance(time_open_raw, str):
                s = time_open_raw.strip()
                s2 = s.replace("Z", "").replace("+00:00", "").replace("+00", "")
                dt = None
                if "T" in s2:
                    dt = datetime.fromisoformat(s2)
                else:
                    dt = datetime.strptime(s2[:19], "%Y-%m-%d %H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
                ts_open = dt.timestamp()
                return max(0.0, (now_ts - float(ts_open)) / 60.0)

        except Exception:
            return None

        return None

    # -----------------------------
    # predict
    # -----------------------------
    def predict(
        self,
        snapshot: Dict[str, Any],
        position_state: Optional[Any] = None,
    ) -> Dict[str, Optional[float]]:

        trade = snapshot.get("trade", {}) or {}
        expected_row = snapshot.get("expected_row") or {}

        symbol = str(trade.get("symbol") or expected_row.get("symbol") or "").strip()
        if not symbol:
            return {
                "continue_probability": None,
                "exit_probability": None,
                "confidence": None,
                "features_incomplete": True,
            }

        timeframe = self.EXIT_MODEL_TIMEFRAME

        features_incomplete = False
        incomplete_reason = ""

        # -------- QuantDinger-only indicators (ATR H4 with smart fallback) --------
        atr: Optional[float] = None
        rsi: Optional[float] = None
        adx: Optional[float] = None
        market_regime: Any = None
        time_open_minutes: Optional[float] = None

        # 1) ATR (H4) - do NOT block prediction if QuantDinger H4 is empty
        try:
            atr_ind = self.get_indicators_from_quantdinger(symbol, timeframe="H4")
            atr_raw = (atr_ind or {}).get("atr")
            if atr_raw is None:
                raise RuntimeError(f"ATR missing from QuantDinger for {symbol} H4")
            atr = float(atr_raw)
        except Exception as e:
            # retry once more explicitly (one extra attempt)
            logger.warning(f"[EXIT_MODEL][WARN] ATR(H4) fetch failed for {symbol}: {e} -> retrying once...")
            try:
                atr_ind = self.get_indicators_from_quantdinger(symbol, timeframe="H4")
                atr_raw = (atr_ind or {}).get("atr")
                if atr_raw is None:
                    raise RuntimeError(f"ATR missing from QuantDinger for {symbol} H4")
                atr = float(atr_raw)
            except Exception as e2:
                # fallback to cached last valid ATR
                cached = self._last_valid_atr_by_symbol.get(symbol)
                if cached is not None and cached > 0:
                    atr = float(cached)
                    logger.warning(f"[EXIT_MODEL][WARN] Using cached ATR(H4) for {symbol}: atr={atr} (QuantDinger failed: {e2})")
                else:
                    # final default fallback
                    atr = 0.0010
                    logger.warning(f"[EXIT_MODEL][WARN] Using DEFAULT ATR(H4) for {symbol}: atr={atr} (no cache; QuantDinger failed: {e2})")

        # Store valid ATR for next time
        if atr is not None and atr > 0:
            self._last_valid_atr_by_symbol[symbol] = float(atr)

        # 2) RSI (H1) - smart fallback (so live check can proceed)
        try:
            rsi_ind = self.get_indicators_from_quantdinger(symbol, timeframe="H1")
            rsi_raw = (rsi_ind or {}).get("rsi")
            if rsi_raw is None:
                raise RuntimeError(f"RSI missing from QuantDinger for {symbol} H1")
            rsi = float(rsi_raw)
        except Exception as e:
            logger.warning(f"[EXIT_MODEL][WARN] RSI(H1) fetch failed for {symbol}: {e} -> retrying once...")
            try:
                rsi_ind = self.get_indicators_from_quantdinger(symbol, timeframe="H1")
                rsi_raw = (rsi_ind or {}).get("rsi")
                if rsi_raw is None:
                    raise RuntimeError(f"RSI missing from QuantDinger for {symbol} H1")
                rsi = float(rsi_raw)
            except Exception as e2:
                # no cached RSI structure exists; use neutral default with explicit warning
                rsi = 50.0
                logger.warning(f"[EXIT_MODEL][WARN] Using DEFAULT RSI(H1)=50.0 for {symbol} (QuantDinger failed: {e2})")
        if not features_incomplete:
            # 3) ADX (H4 candles) - strict
            try:
                adx = self._compute_adx_from_candles_h4(symbol, period=14)
            except Exception as e:
                features_incomplete = True
                incomplete_reason = str(e)

        if not features_incomplete:
            # 4) market_regime - strict
            try:
                market_regime = snapshot.get("market_regime")
                if market_regime is None or (isinstance(market_regime, str) and not market_regime.strip()):
                    raise RuntimeError(f"market_regime missing/empty for {symbol}")
            except Exception as e:
                features_incomplete = True
                incomplete_reason = str(e)

        if not features_incomplete:
            # 5) time_open
            try:
                time_open_raw = trade.get("time_open")
                time_open_minutes = self._time_open_to_minutes(time_open_raw)
                if time_open_minutes is None:
                    raise RuntimeError(f"time_open missing/unparseable for {symbol}")
            except Exception as e:
                features_incomplete = True
                incomplete_reason = str(e)

        # Validation (after fallbacks)
        if not features_incomplete:
            try:
                if atr is None or not (atr > 0.0):
                    raise RuntimeError(f"Invalid atr={atr} for {symbol}")
                if rsi is None or not (0.0 <= rsi <= 100.0):
                    raise RuntimeError(f"Invalid rsi={rsi} for {symbol}")
                if adx is None or not (adx >= 0.0):
                    raise RuntimeError(f"Invalid adx={adx} for {symbol}")
                INVALID_REGIME_VALUES = {"unknown", "undefined", "n/a", "none"}
                if isinstance(market_regime, str) and market_regime.strip().lower() in INVALID_REGIME_VALUES:
                    raise RuntimeError(f"Invalid/suspicious market_regime placeholder for {symbol}: {market_regime}")

                if market_regime is None or (isinstance(market_regime, str) and not market_regime.strip()):
                    raise RuntimeError(f"Invalid market_regime for {symbol}")
            except Exception as e:
                features_incomplete = True
                incomplete_reason = str(e)

        if features_incomplete:
            prev = self._consecutive_incomplete_by_symbol.get(symbol, 0)
            self._consecutive_incomplete_by_symbol[symbol] = prev + 1
            self._maybe_critical_log(symbol=symbol, timeframe=timeframe, reason=incomplete_reason, features_incomplete=True)

            return {
                "continue_probability": None,
                "exit_probability": None,
                "confidence": None,
                "features_incomplete": True,
            }

        # success -> reset counters
        self._consecutive_incomplete_by_symbol[symbol] = 0
        self._last_success_ts_by_symbol[symbol] = time.time()

        # Build non-indicator features
        # 1) spread (prefer MT5 symbol_info spread; best-effort)
        spread: float = 0.0
        spread_source: str = "none"

        # Prefer MT5 spread if available in local terminal
        try:
            import MetaTrader5 as mt5  # type: ignore
            from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

            if mt5 is not None:
                try:
                    if not mt5.terminal_info():
                        mt5.initialize()
                except Exception:
                    pass

                # Try to ensure symbol info
                try:
                    if not mt5.initialize(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
                        # some builds require initialize() then login; best-effort
                        pass
                except Exception:
                    pass

                info = mt5.symbol_info(symbol)
                if info is not None and getattr(info, "spread", None) is not None:
                    spread = float(getattr(info, "spread"))
                    spread_source = "mt5.symbol_info.spread"
        except Exception as e:
            # Keep going with other sources
            logger.warning(f"[EXIT_MODEL][WARN] mt5 spread fetch failed symbol={symbol}: {e}")

        # Fallback to snapshot/position/expected_row
        if spread <= 0:
            spread_raw = trade.get("spread")
            if spread_raw is not None:
                spread = float(spread_raw)
                spread_source = "snapshot.trade.spread"

        if spread <= 0 and position_state is not None:
            spread = float(getattr(position_state, "last_spread", 0.0) or 0.0)
            spread_source = "position_state.last_spread"

        if spread <= 0:
            try:
                spread = float(expected_row.get("expected_spread") or expected_row.get("actual_spread") or 0.0)
                spread_source = "expected_row.spread"
            except Exception:
                spread = 0.0

        if spread <= 0:
            logger.warning(f"[EXIT_MODEL][WARN] spread not available for symbol={symbol} -> using 15.0 (spread_source={spread_source})")
            spread = 15.0

        # 2) volume
        volume_raw = trade.get("volume")
        volume = float(volume_raw) if volume_raw is not None else 0.0
        if volume <= 0 and position_state is not None:
            volume = float(getattr(position_state, "volume", 0.0) or 0.0)
        if volume <= 0:
            try:
                volume = float(expected_row.get("expected_volume") or expected_row.get("actual_volume") or 0.0)
            except Exception:
                volume = 0.0
        if volume <= 0:
            logger.warning(f"[EXIT_MODEL][WARN] volume not available for symbol={symbol} -> using 0.0")

        # 3) mfe/mae from open-trade state
        current_profit = float(trade.get("profit") or 0.0)
        mfe: float = 0.0
        mae: float = 0.0

        if position_state is not None:
            order_id = trade.get("order_id")
            mfe, mae = self.update_mfe_mae(position_state=position_state, current_profit=current_profit, order_id=order_id)

        # If mae still invalid/0, compute from current_profit
        # Requested: mae = min(0.0, current_profit)
        # Requested: mfe from current profit as well
        if mae == 0.0:
            mae = float(min(0.0, current_profit))
            if position_state is not None:
                try:
                    position_state.mae = mae
                except Exception:
                    pass
            logger.warning(f"[EXIT_MODEL][WARN] mae was 0.0 -> computed mae=min(0.0, current_profit) mae={mae} current_profit={current_profit}")

        if mfe == 0.0:
            mfe = float(max(0.0, current_profit))
            if position_state is not None:
                try:
                    position_state.mfe = mfe
                except Exception:
                    pass
            logger.warning(f"[EXIT_MODEL][WARN] mfe was 0.0 -> computed mfe=max(0.0, current_profit) mfe={mfe} current_profit={current_profit}")


        # 4) profit_decay_pct and trade_health
        profit_decay_pct = 0.0
        if mfe > 0:
            if current_profit > 0:
                profit_decay_pct = max(0.0, (mfe - current_profit) / mfe * 100.0)
            else:
                profit_decay_pct = 100.0 + abs(current_profit)
        elif current_profit <= 0:
            profit_decay_pct = 0.0
        profit_decay_pct = min(profit_decay_pct, 200.0)

        legacy_trade_health = float(expected_row.get("p_win") or 0.5) * 100.0

        # 5) session/trend_h1/trend_h4
        session = expected_row.get("expected_session") or expected_row.get("actual_session")

        trend_h1 = expected_row.get("expected_trend_h1") or expected_row.get("trend_h1")
        trend_h4 = expected_row.get("expected_trend_h4") or expected_row.get("trend_h4")

        # If missing/zero, try QuantDinger-derived trend values (best-effort)
        missing_trend = (trend_h1 is None or trend_h4 is None)
        if missing_trend:
            try:
                t1_ind = self.get_indicators_from_quantdinger(symbol, timeframe="H1")
                t4_ind = self.get_indicators_from_quantdinger(symbol, timeframe="H4")
                if trend_h1 is None and isinstance(t1_ind, dict):
                    trend_h1 = t1_ind.get("trend_h1")
                if trend_h4 is None and isinstance(t4_ind, dict):
                    trend_h4 = t4_ind.get("trend_h4")
            except Exception as e:
                logger.warning(f"[EXIT_MODEL][WARN] failed fetching trend from QuantDinger for symbol={symbol}: {e}")

        # Never silently send 0.0 for trend_h1/trend_h4
        if trend_h1 is None:
            logger.warning(f"[EXIT_MODEL][WARN] trend_h1 missing for symbol={symbol} -> using 50.0 (neutral placeholder)")
            trend_h1 = 50.0
        if trend_h4 is None:
            logger.warning(f"[EXIT_MODEL][WARN] trend_h4 missing for symbol={symbol} -> using 50.0 (neutral placeholder)")
            trend_h4 = 50.0

        news_impact = expected_row.get("expected_news_impact_score") or 0.0

        # Warnings for mfe/mae/volume staying at 0
        if mfe == 0.0:
            logger.warning(f"[EXIT_MODEL][WARN] mfe is 0 for symbol={symbol} (possible state not updating?)")
        if mae == 0.0:
            logger.warning(f"[EXIT_MODEL][WARN] mae is 0 for symbol={symbol} (possible state not updating?)")

        # 6) market_regime: must not allow placeholder_Unknown to reach model
        if isinstance(market_regime, str) and market_regime.strip() == "placeholder_Unknown":
            logger.warning(f"[EXIT_MODEL][WARN] market_regime placeholder_Unknown for symbol={symbol} -> using safe default 'ranging'")
            market_regime = "ranging"

        features_for_model: Dict[str, Any] = {
            "mfe": float(mfe),
            "mae": float(mae),
            "entry_atr": float(atr),
            "entry_rsi": float(rsi),
            "entry_adx": float(adx),
            "market_regime": market_regime,
            "trade_duration": float(time_open_minutes),
            "spread": float(spread),
            "volume": float(volume),
            "session": session,
            "trend_h1": float(trend_h1),
            "trend_h4": float(trend_h4),
            "profit_decay_pct": float(profit_decay_pct),
            "trade_health": float(legacy_trade_health),
            "news_impact": float(news_impact),
        }

        # Log exact 12 features in FEATURE_ORDER just before prediction
        vec_list = build_feature_vector(features_for_model)
        logger.info(f"[EXIT_MODEL][DEBUG_FEATURE_VECTOR] symbol={symbol} FEATURE_ORDER_VALUES=")
        for i, name in enumerate(FEATURE_ORDER):
            val = vec_list[i] if i < len(vec_list) else None
            logger.info(f"[EXIT_MODEL][DEBUG_FEATURE] {name}={val}")


        # model call
        try:
            exit_prob = predict_exit_probability(features_for_model)
        except Exception:
            # Fail-safe: do not decide exit on invalid model
            return {
                "continue_probability": None,
                "exit_probability": None,
                "confidence": None,
                "features_incomplete": True,
            }

        exit_prob = float(max(0.0, min(1.0, float(exit_prob))))
        expected_row["exit_probability"] = exit_prob
        p_win = float(max(0.0, min(1.0, 1.0 - exit_prob)))
        expected_row["p_win"] = p_win

        # (Removed) Early-window guard (minimum_trade_duration_minutes=3.0)

        cont_prob = 1.0 - exit_prob
        conf = float(exit_prob)

        logger.info(
            f"[EXIT_MODEL] symbol={symbol} prediction_exit_probability={exit_prob:.4f} decision_signal={exit_prob:.1%}"
        )

        return {
            "continue_probability": cont_prob,
            "exit_probability": exit_prob,
            "confidence": conf,
            "features_incomplete": False,
        }

