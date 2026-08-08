
# Trading Bot V3 - execution/reconciliation.py
# Reconcile DB vs QuantDinger vs MT5 every 60 seconds + Profit Target + Smart News Monitor

import time
import threading
from typing import Dict

import math


from utils.logger import get_logger
from data.storage.database import (
    get_open_trades,
    close_trade_db_by_order_id,
    get_execution_dataset,
    upsert_execution_actual,
)
from risk.risk_governor import get_risk_governor




# MT5 direct (smart SL/TP modifications must go directly to MT5)
try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    mt5 = None


def _recover_mt5_session(context: str) -> bool:
    """Re-establish the shared MT5 session after an IPC failure.

    Recovery must go through data.market.mt5_session, which owns the connection
    and holds the process-wide lock while rebuilding it. Calling
    mt5.shutdown()/mt5.initialize() from here — as this module used to — kills
    the connection while the main cycle, the post-entry manager and the watchdog
    are mid-call, producing exactly the IPC errors it was trying to repair.
    """
    logger.warning("[RECONCILE] %s — recovering session via mt5_session", context)
    try:
        from data.market.mt5_session import ensure_session

        recovered = ensure_session(force_relogin=True)
        if not recovered:
            logger.error("[RECONCILE] session recovery failed (%s)", context)
        return recovered
    except Exception as exc:
        logger.error("[RECONCILE] session recovery raised (%s): %s", context, exc)
        return False


def safe_positions_get(max_retries: int = 3, delay: float = 2.0):
    """Get MT5 positions with smart retries + reinitialize on IPC loss.

    Requirements:
    - 3 attempts with delay=2
    - If last_error indicates (-10004, 'No IPC connection'), do shutdown+initialize
    - Log clear error when all attempts fail
    """
    if mt5 is None:
        logger.error("safe_positions_get: MT5 library not available (mt5=None)")
        return None

    for attempt in range(max_retries):
        try:
            positions = mt5.positions_get()
            if positions is not None:
                return positions
            # if None returned, try reconnect logic
            try:
                last_err = mt5.last_error()
            except Exception:
                last_err = None
            logger.warning(f"[RECONCILE] positions_get returned None attempt={attempt+1}/{max_retries} last_error={last_err}")
        except Exception as e:
            # attempt reconnect if IPC error
            try:
                last_err = mt5.last_error()
            except Exception:
                last_err = None
            logger.warning(f"[RECONCILE] positions_get exception attempt={attempt+1}/{max_retries} err={e} last_error={last_err}")

            code = None
            try:
                if last_err is not None and isinstance(last_err, (tuple, list)) and len(last_err) >= 1:
                    code = int(last_err[0])
            except Exception:
                code = None

            if code == -10004:
                # Recover through the session owner. This used to call
                # mt5.shutdown() directly, tearing down the connection the
                # other three threads were mid-call on — which manufactured
                # more IPC failures than it fixed.
                _recover_mt5_session("IPC lost (-10004) during positions_get")

        # retry delay + smart reinit for IPC even if no exception
        if attempt < max_retries - 1:
            try:
                last_err = None
                try:
                    last_err = mt5.last_error()
                except Exception:
                    last_err = None

                code = None
                if last_err is not None:
                    try:
                        code = int(last_err[0])
                    except Exception:
                        code = None

                if code == -10004:
                    _recover_mt5_session("IPC still failing (-10004) before retry")
            except Exception:
                pass

            time.sleep(delay)

    logger.error(f"[RECONCILE] Failed to get MT5 positions after {max_retries} attempts (IPC={True if mt5 else False})")
    return None


# DeepSeek integration for smart profit protection
import json


from analysis.models.performance_monitor import get_global_monitor
from telegram.notifier import notify_trade_closed
from core.heartbeat import seconds_since_last_beat


logger = get_logger("reconciliation")


# ==============================
# إعدادات حد الربح الذكي (Smart Profit Protection)
# ==============================
# هذا النظام جديد: يعتمد على LLM triggers (breakeven/trail) مع fallback دفاعي.
# الفكرة: يتم تخزين خطة التتبع داخل comment الصفقة بصيغة JSON صغيرة.
# إذا لم تتوفر AI triggers (أو فشلت الاستدعاءات)، نستخدم قيم افتراضية مشتقة من ATR والهدف.

# fallback defaults (used only if AI fails or plan missing)
SMART_BREAKEVEN_PCT_OF_TARGET = 0.01   # 1% من الهدف
SMART_TRAIL_START_PCT_OF_TARGET = 0.50 # 50% من الهدف
SMART_TRAIL_DIST_PCT_OF_TARGET = 0.01  # 1% من الهدف

# AI loop: apply breakeven/trailing gradually, and ensure idempotency per stage
SMART_LOOP_INTERVAL = 30  # seconds

# Dedup stages markers
BREAKEVEN_DONE_KEY = "be_done"
TRAILING_STARTED_KEY = "trail_started"
TRAILING_LOCKED_SL_KEY = "trail_sl_locked"

# TTL for plan in-memory to avoid repeated parsing failures
_PLAN_PARSE_CACHE_TTL_SEC = 600

# Strategy constants
# - breakeven_trigger: min(target*1%, 2*ATR)
# - trail_start: target*50%
# - trail_distance: target*1%

# For compatibility with previous version that used simple profit target.
# Keep existing values so reconciliation doesn't regress if plan missing.
PROFIT_TARGET_USD = 50.0
TRAILING_TRIGGER = 0.50
TRAILING_LOCK_PERCENT = 0.30


# ==============================
# إعدادات المراقبة الذكية (Smart News Monitor)
# ==============================
NEWS_MONITOR_INTERVAL = 900     # كل 15 دقيقة
MIN_CONFIDENCE_TO_EXIT = 0.75   # ثقة AI للخروج
_last_news_check = 0


# Dedup: notifications/DB close must be idempotent by order_id only (no pnl-based logic)
_DEDUP_TTL_SEC = 15 * 60

# Ignore recently opened trades (to avoid false orphans during reconciliation timing gap)
# Trade opened in last 90 seconds is considered "not yet synced" - skip orphan check
_ORPHAN_IGNORE_WINDOW_SEC = 90

_recent_closed_order_ids = {}  # order_id -> closed_at_epoch

# Dedup for SL modification notifications to avoid spam
_DEDUP_SL_TTL_SEC = 10 * 60
_recent_sl_updates = {}  # order_id -> last_update_epoch

# Unified dedup for Locking-in-Profits notifications: order_id -> set(new_sl)
_lock_notifications: Dict[str, set] = {}

# Best-effort cache for profit locked amount per (order_id, new_sl)
# Keyed as f"{order_id}:{new_sl}"
_lock_profit_amount_cache: Dict[str, float] = {}

# ==============================
# Orphan confirmation tracker
# ==============================
# When a DB trade is not found on MT5/QD during reconciliation restart,
# we wait for multiple reconciliation cycles before closing the DB row.
_ORPHAN_CONFIRM_CYCLES = 3

# order_id -> {"count": int, "last_seen": epoch}
orphan_tracker: Dict[str, Dict[str, float]] = {}

# capture bot start time (best-effort for orphan ignore window)
_BOT_START_EPOCH = time.time()



def _is_recently_closed(order_id: str) -> bool:
    now = time.time()
    if not order_id:
        return False
    # purge old
    for oid, ts in list(_recent_closed_order_ids.items()):
        if now - ts > _DEDUP_TTL_SEC:
            _recent_closed_order_ids.pop(oid, None)
    return str(order_id) in _recent_closed_order_ids

def _mark_closed(order_id: str):
    if order_id:
        _recent_closed_order_ids[str(order_id)] = time.time()


def _was_opened_recently(open_time) -> bool:
    """Check if trade was opened within the last _ORPHAN_IGNORE_WINDOW_SEC seconds.

    Args:
        open_time: Trade open time - can be epoch timestamp (int/float) or datetime object

    Returns:
        True if trade was opened within the last 90 seconds, False otherwise
    """
    if open_time is None:
        return False

    try:
        now = time.time()
        open_ts = None

        # Handle different time formats
        if isinstance(open_time, (int, float)):
            open_ts = float(open_time)
        else:
            # Try to convert from datetime or string
            try:
                from datetime import datetime
                if isinstance(open_time, datetime):
                    open_ts = open_time.timestamp()
                elif isinstance(open_time, str):
                    # Try parsing common formats
                    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]:
                        try:
                            dt = datetime.strptime(open_time, fmt)
                            open_ts = dt.timestamp()
                            break
                        except Exception:
                            continue
            except Exception:
                pass

        if open_ts is None:
            return False

        # Check if within 90 second window
        age_seconds = now - open_ts
        return 0 <= age_seconds <= _ORPHAN_IGNORE_WINDOW_SEC
    except Exception:
        return False


def _record_live_performance(order_id: str, actual_pnl: float):
    """Record online performance at trade close.

    Uses expected_ai_confidence as proxy for p_win unless you later add
    a dedicated expected_p_win column.
    """
    try:
        expected_row = get_execution_dataset(order_id) or {}
        expected_p_win = expected_row.get("expected_ai_confidence", None)
        get_global_monitor().record(
            p_win=float(expected_p_win) if expected_p_win is not None else 0.0,
            actual_pnl=actual_pnl,
        )
    except Exception:
        pass


def _feed_risk_governor(order_id: str, pnl: float):
    """Feed the Risk Governor with a closed trade's realized PnL.

    This is called whenever a trade is detected as closed (manually at broker,
    by SL/TP at broker level, or by the bot itself). The Risk Governor uses
    this to update cumulative_loss_r and potentially halt new entries.

    Uses order_id as a unique key (dedup) so that the same trade is never
    recorded twice - regardless of which path closed it (post_entry_manager,
    reconciliation, or broker-side).

    IMPORTANT: This ONLY affects NEW entries. It NEVER closes or manages
    open positions.
    """
    try:
        governor = get_risk_governor()
        # ============================================================
        # UNITS FIX: Compute risk_amount_usd in DOLLARS (not price distance)
        # ============================================================
        # OLD (BUGGY): risk_amount_usd = abs(entry - sl)
        #   -> This is a PRICE DISTANCE (e.g. 0.0030 for EURUSD)
        #   -> risk_governor computes r_multiple = pnl_usd / risk_amount_usd
        #   -> e.g. -50 / 0.0030 = -16666 R (absurd!)
        #   -> Causes cumulative_loss_r to explode and trigger halt immediately
        #
        # NEW (FIXED): Convert price distance to dollars using:
        #   sl_pips = sl_distance / pip
        #   risk_amount_usd = sl_pips * pip_value_per_lot * volume
        # ============================================================
        risk_amount_usd = None
        try:
            expected_row = get_execution_dataset(order_id) or {}
            expected_entry = expected_row.get("expected_entry", None)
            expected_sl = expected_row.get("expected_sl", None)

            # Get trade size (volume) from open trades table
            trade_size = None
            trade_symbol = None
            try:
                open_trades = get_open_trades() or []
                for t in open_trades:
                    if str(t.get("order_id", "")) == str(order_id):
                        trade_size = float(t.get("size", 0) or 0)
                        trade_symbol = t.get("symbol", "")
                        break
            except Exception:
                pass

            if expected_entry and expected_sl and trade_size and trade_size > 0:
                sl_distance = abs(float(expected_entry) - float(expected_sl))
                # Use shared R-multiple module for consistent calculation
                from risk.r_multiple import calculate_risk_amount_usd
                risk_amount_usd = calculate_risk_amount_usd(sl_distance, trade_symbol, trade_size)
                if risk_amount_usd is not None:
                    logger.info(
                        f"[RISK_GOVERNOR] risk_amount_usd={risk_amount_usd:.2f} "
                        f"(sl_dist={sl_distance:.5f} size={trade_size} "
                        f"symbol={trade_symbol})"
                    )
        except Exception:
            risk_amount_usd = None

        governor.record_trade_close(
            pnl_usd=float(pnl or 0.0),
            risk_amount_usd=risk_amount_usd,
            order_id=order_id,
        )
    except Exception as e:
        logger.error(f"[RISK_GOVERNOR] _feed_risk_governor error order_id={order_id}: {e}")


def _normalize_type_to_direction(raw_type) -> str:
    """
    Normalize QuantDinger pos['type'] (can be 'buy'/'sell' OR 0/1 OR similar)
    Return 'buy' or 'sell'.
    """
    s = str(raw_type).strip().lower()
    if s in ["buy", "0", "long", "1"]:  # legacy mapping: you can adjust if your backend uses 0/1 differently
        # In your current logic: direction in ['buy','0'] treated as BUY side.
        return "buy"
    if s in ["sell", "1", "short"]:
        return "sell"
    # fallback: best-effort
    return "buy" if "buy" in s else ("sell" if "sell" in s else "buy")


def _calc_slippage_for_direction(expected_entry: float, actual_entry: float, direction_norm: str) -> float:
    """
    BUY  : slippage = actual - expected
    SELL : slippage = expected - actual
    """
    try:
        if expected_entry is None or actual_entry is None:
            return None
        if float(expected_entry) == 0 or float(actual_entry) == 0:
            return None
        exp = float(expected_entry)
        act = float(actual_entry)
        if str(direction_norm).lower() == "sell":
            return exp - act
        return act - exp
    except Exception:
        return None


def _extract_plan_from_comment(comment: str) -> dict:
    """Parse plan JSON from comment.

    Expected comment format examples:
    - "Bot V3|plan={...}|be_done=0"
    - "Bot V3|plan={...}"

    Returns dict; never raises.
    """
    try:
        if not comment:
            return {}
        c = str(comment)
        # find 'plan=' substring
        if "plan=" not in c:
            return {}
        part = c.split("plan=", 1)[1]
        # plan ends at next delimiter '|'
        if "|" in part:
            part = part.split("|", 1)[0]
        part = part.strip()
        if not part:
            return {}
        # ensure it's JSON
        if part[0] != "{":
            return {}
        return json.loads(part)
    except Exception:
        return {}


def _update_plan_stage(plan: dict, stage_key: str, value):
    try:
        if not isinstance(plan, dict):
            return plan
        plan[stage_key] = value
    except Exception:
        pass
    return plan


def _plan_to_comment(plan: dict) -> str:
    """Serialize plan into comment JSON small."""
    try:
        if plan is None:
            plan = {}
        compact = json.dumps(plan, ensure_ascii=True, separators=(",", ":"))
        # Keep within MT5 comment constraints (<=31) is for direct mt5 comment.
        # Here QuantDinger's stored comment may allow longer.
        # We'll still keep it compact.
        return f"Bot V3|plan={compact}"
    except Exception:
        return "Bot V3"


def _calc_profit_usd(pos: dict) -> float:
    try:
        return float(pos.get("profit", pos.get("pnl", 0)) or 0.0)
    except Exception:
        return 0.0


# ==============================
# Red Flag System - Expert Decision Maker
# ==============================
RED_FLAG_PROFIT_DECAY_THRESHOLD = 70.0
RED_FLAG_TRADE_HEALTH_THRESHOLD = 50.0
RED_FLAG_ATR_MULTIPLIER_THRESHOLD = 2.5
RED_FLAG_REGIME_BAD = ["ranging", "volatile"]
RED_FLAG_MIN_COUNT = 2


def _check_red_flag(
    pos: dict,
    expected_row: dict,
    regime: str,
    peak_profit: float,
    current_profit: float,
    trade_health: float,
    atr: float,
) -> tuple:
    """Check for red flags and return (has_red_flags, flag_count, exit_probability, reasons).

    Args:
        pos: Position dict from MT5
        expected_row: Expected row from execution_dataset
        regime: Market regime string
        peak_profit: Peak profit reached in this trade
        current_profit: Current profit
        trade_health: Trade health score (0-100)
        atr: ATR value

    Returns:
        (has_red_flags, flag_count, exit_probability, reasons_list)
    """
    flags = []
    reasons = []

    try:
        # 1. Profit Decay Check
        if peak_profit > 0 and current_profit < peak_profit:
            decay_pct = 100.0 * (1.0 - current_profit / peak_profit)
            if decay_pct > RED_FLAG_PROFIT_DECAY_THRESHOLD:
                flags.append("profit_decay")
                reasons.append(f"profit_decay:{decay_pct:.0f}%")

        # 2. Trade Health Check
        health_score = float(trade_health or 0)
        if health_score < RED_FLAG_TRADE_HEALTH_THRESHOLD:
            flags.append("trade_health")
            reasons.append(f"trade_health:{health_score:.0f}")

        # 3. ATR Check (if ATR is abnormally high relative to entry)
        if atr and atr > 0:
            expected_atr = float(expected_row.get("expected_atr", 0) or 0)
            if expected_atr > 0 and atr > expected_atr * RED_FLAG_ATR_MULTIPLIER_THRESHOLD:
                flags.append("atr_spike")
                reasons.append(f"atr_spike:{atr:.5f}")

        # 4. Market Regime Check
        regime_lower = str(regime or "").lower()
        if regime_lower in RED_FLAG_REGIME_BAD:
            flags.append("bad_regime")
            reasons.append(f"regime:{regime}")

        flag_count = len(flags)
        has_red_flags = flag_count >= RED_FLAG_MIN_COUNT

        # Calculate exit probability if we have red flags
        exit_prob = 0.0
        if has_red_flags:
            # Import Exit Model
            try:
                from analysis.models.xgboost_exit_model import predict_exit_probability

                features = {
                    "symbol": str(pos.get("symbol", "")),
                    "direction": str(pos.get("type", "")),
                    "atr": atr,
                    "rsi": float(expected_row.get("actual_rsi", 50) or 50),
                    "mfe": float(expected_row.get("mfe", 0) or 0),
                    "mae": float(expected_row.get("mae", 0) or 0),
                    "trade_health": health_score,
                    "profit_decay_pct": decay_pct if flags and "profit_decay" in flags else 0,
                    "time_open_hours": float(pos.get("time_open_hours", 0) or 0),
                    "spread": float(pos.get("spread", 0) or 0),
                    "news_impact": float(expected_row.get("news_impact_score", 0) or 0),
                    "market_regime": regime,
                    "volume": float(pos.get("volume", 0) or 0),
                }
                exit_prob = predict_exit_probability(features)
            except Exception:
                # Fallback: simple probability based on flag count
                exit_prob = 0.7 + (flag_count * 0.1)

        logger.info(f"[RED_FLAG] {pos.get('symbol')} flags={flags} count={flag_count} exit_prob={exit_prob:.1%}")
        return has_red_flags, flag_count, exit_prob, reasons

    except Exception as e:
        logger.error(f"_check_red_flag error: {e}")
        return False, 0, 0.0, []


def _parse_bv3_plan(plan_comment: str) -> tuple:
    """Parse plan format: BV3|B20,T50,D15 (percent values of target profit).

    Returns (ok, B, T, D) where B/T/D are floats.
    Defensive: any parse error returns (False,0,0,0)
    """
    try:
        if not plan_comment:
            return False, 0.0, 0.0, 0.0
        c = str(plan_comment)
        if "BV3|" not in c:
            return False, 0.0, 0.0, 0.0
        # after BV3| can be like B20,T50,D15 or plan=... (ignore)
        after = c.split("BV3|", 1)[1]
        parts = after.split("|", 1)[0]
        chunks = [x.strip() for x in parts.split(",") if x.strip()]
        m = {}
        for ch in chunks:
            # expecting Bxx / Txx / Dxx
            key = ch[0].upper() if ch else ""
            val = float(ch[1:]) if len(ch) > 1 else None
            if key in ["B", "T", "D"] and val is not None:
                m[key] = val
        if not all(k in m for k in ["B", "T", "D"]):
            return False, 0.0, 0.0, 0.0
        return True, float(m["B"]), float(m["T"]), float(m["D"])
    except Exception:
        return False, 0.0, 0.0, 0.0


def _pip_value_usd_per_pip(symbol: str, volume: float) -> float:
    """Compute pip value in USD per 1 pip * 1 volume.

    Uses: info.tick_value * (0.0001 / info.tick_size) for most pairs,
    and 0.01 for JPY pairs.
    """
    if mt5 is None:
        return 0.0
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            return 0.0
        tick_value = float(getattr(info, "tick_value", 0) or 0)
        tick_size = float(getattr(info, "tick_size", 0) or 0)
        if tick_value <= 0 or tick_size <= 0 or volume is None:
            return 0.0

        # JPY pairs heuristic
        pip = 0.01 if symbol.upper().endswith("JPY") else 0.0001
        pip_in_ticks = pip / tick_size
        return tick_value * pip_in_ticks * float(volume)
    except Exception:
        return 0.0


def _distance_price_from_usd(symbol: str, volume: float, distance_usd: float) -> float:
    pv = _pip_value_usd_per_pip(symbol, volume)
    if pv <= 0:
        return 0.0
    # price distance corresponding to USD distance = distance_usd / (volume*pip_value_per_pip)
    # Here pv already includes volume.
    # USD per 1 pip -> convert back to pip count then to price distance.
    pip = 0.01 if symbol.upper().endswith("JPY") else 0.0001
    pip_count = distance_usd / pv
    return pip_count * pip



def _calc_target_from_dataset(order_id: str, default_profit_target: float) -> float:
    """Derive target USD from expected_entry/TP/SL if present.
    We fallback to configured default_profit_target.
    """
    try:
        row = get_execution_dataset(order_id) or {}
        # Try TP distance if stored; else use default.
        # execution_dataset schema currently has expected_final_score etc but not TP.
        # So fallback.
        return float(row.get("expected_entry", 0) or 0)  # placeholder, but fallback below will override
    except Exception:
        return default_profit_target


def _get_fallback_triggers(order_id: str, atr: float, volatility: float, target_usd: float) -> dict:
    """Return defensive triggers based on target and ATR."""
    # breakeven_trigger = min(target*1%, 2*ATR)
    be_by_target = target_usd * SMART_BREAKEVEN_PCT_OF_TARGET
    be_by_atr = (2.0 * atr) if atr is not None else be_by_target
    breakeven_trigger = min(be_by_target, be_by_atr)

    trail_start = target_usd * SMART_TRAIL_START_PCT_OF_TARGET
    trail_distance = target_usd * SMART_TRAIL_DIST_PCT_OF_TARGET

    # Ensure non-negative minimal values
    if breakeven_trigger < 0:
        breakeven_trigger = 0.0
    if trail_start < 0:
        trail_start = 0.0
    if trail_distance < 0:
        trail_distance = 0.0

    return {
        "breakeven_trigger": round(breakeven_trigger, 6),
        "trail_start": round(trail_start, 6),
        "trail_distance": round(trail_distance, 6),
    }


def _call_llm_for_triggers(symbol: str, entry: float, sl: float, tp: float,
                            atr: float, volatility: float, p_win: float,
                            mtf_dir: str, target_usd: float, timeout_sec: int = 20) -> dict:
    """Call DeepSeek/LLM to get triggers.
    Defensive: any error raises to caller.
    """
    import requests
    from config import DEEPSEEK_API_KEY

    prompt = (
        "You are a professional risk manager.\n"
        f"Return ONLY valid JSON (no markdown).\n"
        f"Symbol: {symbol}\n"
        f"Entry: {entry}\nSL: {sl}\nTP: {tp}\n"
        f"Distance to SL in $ (approx): abs(entry-sl)\n"
        f"ATR (current): {atr}\nVolatility: {volatility}\n"
        f"p_win from XGBoost: {p_win}\n"
        f"Multi-timeframe general direction: {mtf_dir}\n"
        f"Total target (expected move) in $: {target_usd}\n\n"
        "Decide and output three values:\n"
        "- breakeven_trigger: profit in $ when to move SL to entry\n"
        "- trail_start: profit in $ when to start trailing\n"
        "- trail_distance: distance in $ to keep between current price and SL while trailing\n\n"
        "Example: {"
        "\"breakeven_trigger\": 2.0, \"trail_start\": 5.0, \"trail_distance\": 1.5}"
        "\n"
    )

    resp = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 120,
            "temperature": 0.2,
        },
        timeout=timeout_sec,
    )

    content = resp.json()["choices"][0]["message"]["content"]
    content = content.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(content)
    # validate keys
    for k in ["breakeven_trigger", "trail_start", "trail_distance"]:
        if k not in data:
            raise ValueError(f"Missing {k} in LLM response")
    return data


def _apply_sltp_modification(order_id: str, symbol: str, direction: str,
                             new_sl: float, new_tp: float = None) -> bool:
    """Apply SL/TP modification DIRECTLY to MT5 using mt5.order_send(action=TRADE_ACTION_SLTP).

    Defensive:
    - If mt5 module is unavailable or request fails, return False and keep reconciliation running.
    - Does NOT go through QuantDinger.
    """
    if mt5 is None:
        logger.error("MT5 library not available; cannot modify SL")
        return False

    if not symbol or new_sl is None:
        return False

    try:
        # Session ownership belongs to mt5_session; do not initialize here.
        from data.market.mt5_session import ensure_session

        if not ensure_session():
            logger.error("[SMART_PROFIT] MT5 session unavailable — skipping SL/TP modify")
            return False

        # best-effort select symbol
        try:
            mt5.symbol_select(symbol, True)
        except Exception:
            pass

        # Current price for deviation and fallback
        tick = None
        try:
            tick = mt5.symbol_info_tick(symbol)
        except Exception:
            tick = None

        # Determine order type for SLTP request
        order_type = mt5.ORDER_TYPE_BUY if str(direction).lower() == "buy" else mt5.ORDER_TYPE_SELL

        price = 0.0
        if tick is not None:
            price = float(tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid or 0.0)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "sl": float(new_sl) if new_sl is not None else 0.0,
            "tp": float(new_tp) if new_tp is not None else 0.0,
            "magic": 0,
            "order": int(order_id) if str(order_id).isdigit() else 0,
            "type": order_type,
            "price": price,
            "deviation": 20,
        }

        logger.info(f"[SMART_PROFIT] Sending SLTP to MT5 directly: {request}")

        result = mt5.order_send(request)
        if result is None:
            logger.error(f"[SMART_PROFIT] MT5 order_send returned None last_error={mt5.last_error()}")
            return False

        if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
            logger.error(f"[SMART_PROFIT] MT5 SLTP failed retcode={getattr(result,'retcode',None)} result={result}")
            return False

        # 🔒 Locking in Profits notification (dedup)
        # dedup by (order_id + new_sl)
        try:
            oid = str(order_id)
            ns = float(new_sl)
            if ns and ns > 0:
                sent_set = _lock_notifications.setdefault(oid, set())
                dedup_key = ns
                if dedup_key not in sent_set:
                    # best-effort: profit amount prepared by caller in reconciliation step
                    cache_key = f"{oid}:{ns}"
                    profit_locked = float(_lock_profit_amount_cache.get(cache_key, 0.0))

                    from telegram.notifier import send
                    send(
                        f"🔒 Locking in Profits: {symbol} - SL moved to {ns} - Profit locked: {profit_locked:.2f}$"
                    )
                    logger.info(
                        f"[LOCK_PROFITS] Notified lock order_id={oid} symbol={symbol} new_sl={ns} profit_locked={profit_locked}"
                    )
                    sent_set.add(dedup_key)
        except Exception as e:
            logger.error(f"[LOCK_PROFITS] notification error: {e}")

        return True

    except Exception as e:
        logger.error(f"[SMART_PROFIT] MT5 SLTP modification failed: {e}")
        return False



# UNUSED (legacy, kept for reference only) - not called anywhere in the codebase.
# DO NOT wire this into any loop without adding a heartbeat/post-entry-active check first,
# otherwise it will independently close/modify positions without coordinating with PostEntryManager.

def _smart_profit_protection_step(qd_positions: list):
    """Smart profit protection (after entry only) based on MT5 direct SL/TP updates."""

    for pos in qd_positions:
        try:
            order_id = str(pos.get("id", pos.get("ticket", "")) or "")
            if not order_id:
                continue
            if _is_recently_closed(order_id):
                continue

            symbol = pos.get("symbol", "")
            if not symbol:
                continue

            comment = pos.get("comment", "") or ""
            ok_plan, Bp, Tp, Dp = _parse_bv3_plan(comment)
            if not ok_plan:
                continue

            direction_norm = _normalize_type_to_direction(pos.get("type", pos.get("direction", "")))
            profit_usd = _calc_profit_usd(pos)

            entry = float(pos.get("price_open", pos.get("price", 0)) or 0)
            current_sl = float(pos.get("sl", 0) or 0)

            tick = None
            try:
                if mt5 is not None:
                    tick = mt5.symbol_info_tick(symbol)
            except Exception:
                tick = None
            if tick is None:
                continue

            current_bid = float(getattr(tick, "bid", 0) or 0)
            current_ask = float(getattr(tick, "ask", 0) or 0)

            volume = float(pos.get("volume", pos.get("size", 0)) or 0)
            if volume <= 0:
                continue

            tp = pos.get("tp", pos.get("take_profit", None))
            if tp is None:
                continue
            tp = float(tp)

            info = None
            try:
                info = mt5.symbol_info(symbol) if mt5 is not None else None
            except Exception:
                info = None
            if info is None:
                continue

            pip = 0.01 if symbol.upper().endswith("JPY") else 0.0001
            tick_value = float(getattr(info, "tick_value", 0) or 0)
            tick_size = float(getattr(info, "tick_size", 0) or 0)
            if tick_value <= 0 or tick_size <= 0:
                continue

            pip_in_ticks = pip / tick_size
            pip_value_usd = tick_value * pip_in_ticks * volume
            if pip_value_usd <= 0:
                continue

            target_pips = abs(tp - entry) / pip if pip > 0 else 0
            target_profit_usd = target_pips * pip_value_usd
            if target_profit_usd <= 0:
                continue

            breakeven_usd = target_profit_usd * Bp / 100.0
            trail_start_usd = target_profit_usd * Tp / 100.0
            trail_distance_usd = target_profit_usd * Dp / 100.0

            distance_price = _distance_price_from_usd(symbol, volume, trail_distance_usd)
            if distance_price <= 0:
                continue

            stage_state = _plan_parse_cache.get(order_id, {}).get("stage", {})
            be_done = bool(stage_state.get(BREAKEVEN_DONE_KEY, False))
            trail_done = bool(stage_state.get(TRAILING_LOCKED_SL_KEY, False))

            # breakeven once
            if (not be_done) and profit_usd >= breakeven_usd:
                new_sl = entry
                better = (current_sl == 0) or (
                    (direction_norm == "buy" and new_sl > current_sl) or (direction_norm == "sell" and new_sl < current_sl)
                )
                if better and new_sl > 0:
                    # prepare profit locked amount for notification
                    try:
                        _lock_profit_amount_cache[f"{order_id}:{float(new_sl)}"] = float(profit_usd)
                    except Exception:
                        pass

                    ok = _apply_sltp_modification(order_id, symbol, direction_norm, new_sl=new_sl)
                    if ok:
                        logger.info(
                            f"[SMART_PROFIT] Breakeven SL sent directly to MT5 for {symbol} (order {order_id})"
                        )
                        stage_state[BREAKEVEN_DONE_KEY] = True
                        _plan_parse_cache[order_id] = {"ts": time.time(), "plan": {}, "stage": stage_state}

            # trailing once
            if profit_usd >= trail_start_usd:
                if direction_norm == "buy":
                    new_sl = current_bid - distance_price
                    better = (current_sl == 0) or (new_sl > current_sl)
                else:
                    new_sl = current_ask + distance_price
                    better = (current_sl == 0) or (new_sl < current_sl)

                if better and new_sl > 0 and (not trail_done):
                    # prepare profit locked amount for notification
                    try:
                        _lock_profit_amount_cache[f"{order_id}:{float(new_sl)}"] = float(profit_usd)
                    except Exception:
                        pass

                    ok = _apply_sltp_modification(order_id, symbol, direction_norm, new_sl=new_sl)
                    if ok:
                        logger.info(
                            f"[SMART_PROFIT] Trailing SL sent directly to MT5 for {symbol} (order {order_id})"
                        )
                        stage_state[TRAILING_LOCKED_SL_KEY] = True
                        _plan_parse_cache[order_id] = {"ts": time.time(), "plan": {}, "stage": stage_state}

        except Exception as e:
            logger.error(f"Smart profit protection step error: {e}")



# UNUSED (legacy, kept for reference only) - not called anywhere in the codebase.
# DO NOT wire this into any loop without adding a heartbeat/post-entry-active check first,
# otherwise it will independently close/modify positions without coordinating with PostEntryManager.

def check_profit_targets(qd_positions: list):
    """Apply fixed profit targets only for trades WITHOUT BV3 plan.

    - If BV3 plan exists in comment (ok_plan=True) => skip fixed limits.
    - Otherwise apply defensive fixed limits (PROFIT_TARGET_USD / TRAILING_TRIGGER).
    """


    for pos in qd_positions:
        try:
            order_id = str(pos.get("id", pos.get("ticket", "")))
            comment = pos.get("comment", "") or ""
            ok_plan, _, _, _ = _parse_bv3_plan(comment)
            if ok_plan:
                # BV3 safety/LLM plan already handled elsewhere; skip fixed profit logic
                continue


            symbol = pos.get("symbol", "")
            profit = float(pos.get("profit", 0))
            direction = _normalize_type_to_direction(pos.get("type", pos.get("direction", "")))

            if _is_recently_closed(order_id):
                continue

            # BV3 gate: if trade carries BV3 plan, skip fixed targets
            comment = pos.get("comment", "") or ""
            ok_plan, Bp, Tp, Dp = _parse_bv3_plan(comment)
            if ok_plan:
                continue

            actual_entry = float(pos.get("price_open", 0) or 0)
            actual_exit = float(pos.get("price_current", 0) or 0)

            expected_row = get_execution_dataset(order_id)
            expected_entry = None
            if expected_row:
                expected_entry = expected_row.get("expected_entry", None)

            if profit >= PROFIT_TARGET_USD:
                logger.info(f"🎯 Profit target reached: {symbol} profit=${profit:.2f} → closing")
                success = _close_trade_mt5(order_id)
                if success:
                    close_trade_db_by_order_id(order_id, pnl=profit)

                slippage = _calc_slippage_for_direction(expected_entry, actual_entry, direction)
                price_gap = (actual_entry - expected_entry) if (expected_entry not in (None, 0) and actual_entry not in (None, 0)) else None

                execution_quality_score = 0.0
                if slippage is not None:
                    execution_quality_score = max(0.0, 100.0 - abs(slippage) * 1000.0)

                upsert_execution_actual(
                    order_id=order_id,
                    actual_entry=actual_entry,
                    actual_exit=actual_exit,
                    actual_pnl=profit,
                    spread_at_entry=None,
                    slippage=slippage,
                    execution_delay_ms=None,
                    execution_quality_score=execution_quality_score,
                    price_gap=price_gap,
                    actual_indicators_json=None
                )

                if not _is_recently_closed(order_id):
                    _mark_closed(order_id)
                    _record_live_performance(order_id=order_id, actual_pnl=profit)
                    notify_trade_closed(
                        symbol=symbol,
                        direction=direction,
                        pnl=profit,
                        reason="✅ هدف الربح وصل",
                        size=pos.get("volume", 0),
                        entry=actual_entry,
                        exit_price=actual_exit
                    )
                continue

            trigger_amount = PROFIT_TARGET_USD * TRAILING_TRIGGER
            if profit >= trigger_amount:
                lock_amount = profit * TRAILING_LOCK_PERCENT
                if profit <= lock_amount and profit > 0:
                    logger.info(f"⚠️ Trailing stop hit: {symbol} profit=${profit:.2f} → closing")
                    success = _close_trade_mt5(order_id)
                    if success:
                        close_trade_db_by_order_id(order_id, pnl=profit)

                        actual_entry = float(pos.get("price_open", 0) or 0)
                        actual_exit = float(pos.get("price_current", 0) or 0)

                        expected_row = get_execution_dataset(order_id)
                        expected_entry = expected_row.get("expected_entry", None) if expected_row else None

                        slippage = _calc_slippage_for_direction(expected_entry, actual_entry, direction)
                        price_gap = (actual_entry - expected_entry) if (expected_entry not in (None, 0) and actual_entry not in (None, 0)) else None

                        execution_quality_score = 0.0
                        if slippage is not None:
                            execution_quality_score = max(0.0, 100.0 - abs(slippage) * 1000.0)

                        upsert_execution_actual(
                            order_id=order_id,
                            actual_entry=actual_entry,
                            actual_exit=actual_exit,
                            actual_pnl=profit,
                            spread_at_entry=None,
                            slippage=slippage,
                            execution_delay_ms=None,
                            execution_quality_score=execution_quality_score,
                            price_gap=price_gap,
                            actual_indicators_json=None
                        )

                        if not _is_recently_closed(order_id):
                            _mark_closed(order_id)
                            notify_trade_closed(
                                symbol=symbol,
                                direction=direction,
                                pnl=profit,
                                reason=f"🔒 Trailing Stop - ربح محمي ${lock_amount:.2f}",
                                size=pos.get("volume", 0),
                                entry=actual_entry,
                                exit_price=actual_exit
                            )

        except Exception as e:
            logger.error(f"Profit check error for {pos}: {e}")


# ⚠️ DEBUG/Guard: do not delete/rename this function.
# This function is referenced by automation/search and by reconciliation workflow.
# If you change its behavior, update all callers and the corresponding logic.
# [WARNING] PostEntryManager is responsible for all post-entry exit/modify logic.
# Reconciliation only handles DB↔MT5 orphan sync + SL-cross safety close.
# Do not add additional profit/SL modification logic here.

def check_news_conflict(qd_positions: list):
    # [WARNING] PostEntryManager is responsible for all post-entry exit/modify logic.
    # Reconciliation only handles DB↔MT5 orphan sync + SL-cross safety close.
    # Do not add additional profit/SL modification logic here.
    """المراقبة الذكية — يقفل الصفقة لو الأخبار عكست الاتجاه + dedup by order_id"""
    global _last_news_check

    if time.time() - _last_news_check < NEWS_MONITOR_INTERVAL:
        return

    _last_news_check = time.time()

    if not qd_positions:
        return

    try:
        from data.news.fetcher import fetch_rss_news
        from analysis.ai.deepseek import analyze_news

        news = fetch_rss_news()
        if not news:
            return

        for pos in qd_positions:
            try:
                order_id = str(pos.get("id", pos.get("ticket", "")))
                symbol = pos.get("symbol", "")
                profit = float(pos.get("profit", 0))
                direction_norm = _normalize_type_to_direction(pos.get("type", pos.get("direction", "")))

                if _is_recently_closed(order_id):
                    continue

                # snapshot is optional in this module; provide it if available and compatible.
                snapshot = None
                try:
                    # execution_dataset may carry snapshot-like cached indicators; best-effort only.
                    expected_row = get_execution_dataset(order_id) or {}
                    snapshot = expected_row.get("snapshot") or expected_row.get("market_snapshot") or None
                except Exception:
                    snapshot = None

                # analyze_news: snapshot may or may not be supported by the current function signature.
                # Requirement: pass snapshot if available; otherwise skip safely.
                try:
                    if snapshot is not None:
                        ai = analyze_news(news, symbol, snapshot)
                    else:
                        ai = analyze_news(news, symbol)
                except TypeError:
                    # Signature mismatch: call without snapshot safely.
                    ai = analyze_news(news, symbol)


                is_conflict = False
                if direction_norm == "buy" and ai.bias == "bearish" and ai.confidence >= MIN_CONFIDENCE_TO_EXIT:
                    is_conflict = True
                elif direction_norm == "sell" and ai.bias == "bullish" and ai.confidence >= MIN_CONFIDENCE_TO_EXIT:
                    is_conflict = True

                if is_conflict:
                    logger.info(f"🔄 News conflict: {symbol} pos={direction_norm} AI={ai.bias} conf={ai.confidence:.2f} → closing")
                    notify_alert(
                        f"📰 مراقبة ذكية: {symbol}\n"
                        f"الصفقة: {'شراء' if direction_norm == 'buy' else 'بيع'}\n"
                        f"الأخبار الجديدة: {ai.bias} ({ai.confidence:.0%})\n"
                        f"السبب: {ai.reason[:80]}\n"
                        f"القرار: إغلاق الصفقة"
                    )
                    # BV3/LLM safety: close via mt5 ticket-based closer
                    success = _close_trade_mt5(order_id)
                    if success:
                        close_trade_db_by_order_id(order_id, pnl=profit)

                        actual_entry = float(pos.get("price_open", 0) or 0)
                        actual_exit = float(pos.get("price_current", 0) or 0)

                        expected_row = get_execution_dataset(order_id)
                        expected_entry = expected_row.get("expected_entry", None) if expected_row else None

                        slippage = _calc_slippage_for_direction(expected_entry, actual_entry, direction_norm)
                        price_gap = (actual_entry - expected_entry) if (expected_entry not in (None, 0) and actual_entry not in (None, 0)) else None
                        execution_quality_score = None
                        if slippage is not None:
                            execution_quality_score = max(0.0, 100.0 - abs(slippage) * 1000.0)

                        upsert_execution_actual(
                            order_id=order_id,
                            actual_entry=actual_entry,
                            actual_exit=actual_exit,
                            actual_pnl=profit,
                            spread_at_entry=None,
                            slippage=slippage,
                            execution_delay_ms=None,
                            execution_quality_score=execution_quality_score,
                            price_gap=price_gap,
                            actual_indicators_json=None
                        )

                        if not _is_recently_closed(order_id):
                            _mark_closed(order_id)
                            notify_trade_closed(
                                symbol=symbol,
                                direction=direction_norm,
                                pnl=profit,
                                reason=f"📰 عكس الأخبار - {ai.bias} {ai.confidence:.0%}",
                                size=pos.get("volume", 0),
                                entry=actual_entry,
                                exit_price=actual_exit
                            )
                else:
                    logger.debug(f"News check {symbol}: AI={ai.bias} conf={ai.confidence:.2f} — no conflict")

            except Exception as e:
                logger.error(f"News conflict check error for {pos}: {e}")

    except Exception as e:
        logger.error(f"News monitor error: {e}")


def _mt5_positions_as_qd_like_dicts(positions) -> list:
    """Convert mt5.positions_get() namedtuples into QuantDinger-like dicts."""
    out = []
    try:
        if not positions:
            return out
        for p in positions:
            try:
                d = p._asdict() if hasattr(p, "_asdict") else dict(p.__dict__)
            except Exception:
                d = {}

            symbol = d.get("symbol") or d.get("Symbol") or ""
            ticket = d.get("ticket") or d.get("position_id") or d.get("id")
            order_id = ticket
            entry = d.get("price_open") or d.get("price_open")
            sl = d.get("sl")
            tp = d.get("tp")
            profit = d.get("profit") or d.get("pnl") or 0
            volume = d.get("volume") or d.get("Volume")
            comment = d.get("comment") or ""
            ptype = d.get("type")
            if isinstance(ptype, (int, float)):
                direction = "buy" if int(ptype) == 0 else "sell"
            else:
                direction = "buy" if str(ptype).lower() in ["0", "buy", "long"] else "sell"

            out.append(
                {
                    "id": str(order_id) if order_id is not None else "",
                    "ticket": str(order_id) if order_id is not None else "",
                    "symbol": symbol,
                    "type": direction,
                    "direction": direction,
                    "volume": float(volume) if volume is not None else 0,
                    "price_open": float(entry) if entry is not None else 0,
                    "sl": float(sl) if sl is not None else 0,
                    "tp": float(tp) if tp is not None else None,
                    "profit": float(profit) if profit is not None else 0,
                    "comment": comment or "",
                    "price_current": float(d.get("price_current") or d.get("price") or 0),
                }
            )
    except Exception:
        return []
    return out


def _close_trade_mt5(ticket) -> bool:
    """Close an MT5 position by ticket using mt5.order_send(action=DEAL)."""
    if mt5 is None:
        logger.error("MT5 library not available; cannot close")
        return False

    try:
        ticket_str = str(ticket)
        if not ticket_str.isdigit():
            return False
        ticket_int = int(ticket_str)

        pos_list = mt5.positions_get(ticket=ticket_int)
        if not pos_list:
            pos_list = mt5.positions_get()
            if not pos_list:
                return False
            pos_list = [p for p in pos_list if str(getattr(p, "ticket", "")) == ticket_str]

        if not pos_list:
            return False

        p = pos_list[0]
        d = p._asdict() if hasattr(p, "_asdict") else p.__dict__
        symbol = d.get("symbol")
        volume = d.get("volume")
        ptype = d.get("type")

        if not symbol or not volume:
            return False

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        is_buy = (int(ptype) == 0) if isinstance(ptype, (int, float, str)) else str(ptype).lower() in ["0", "buy"]
        order_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        price = float(tick.bid if is_buy else tick.ask)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "position": ticket_int,
            "price": price,
            "deviation": 20,
            "magic": 0,
            "comment": "Bot V3",
        }

        logger.info(f"[MT5_CLOSE] Sending DEAL close to MT5 for ticket={ticket_int} request={request}")
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"[MT5_CLOSE] order_send returned None last_error={mt5.last_error()}")
            return False
        if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
            logger.error(f"[MT5_CLOSE] Failed retcode={getattr(result,'retcode',None)} result={result}")
            return False
        return True

    except Exception as e:
        logger.error(f"[MT5_CLOSE] exception: {e}")
        return False


def _mt5_position_exists_by_ticket(order_id: str) -> bool:
    """Best-effort check: whether MT5 still has a position with the given ticket."""
    if mt5 is None:
        return False
    order_id_str = str(order_id).strip()
    if not order_id_str.isdigit():
        return False
    try:
        ticket_int = int(order_id_str)
        positions = mt5.positions_get(ticket=ticket_int)
        return bool(positions)
    except Exception:
        return False


def reconcile() -> dict:
    """Compare DB trades with MT5 positions and fix mismatches (idempotent by order_id)"""
    db_trades = get_open_trades()
    qd_positions = []
    mt5_positions_query_failed = False

    try:
        if mt5 is not None:
            mt5_positions = safe_positions_get(max_retries=3, delay=2.0)
            # IMPORTANT: when query fails, do NOT treat "0 positions" as "no longer exists"
            # otherwise DB trades can be incorrectly classified as orphans.
            if mt5_positions is None:
                mt5_positions_query_failed = True
                try:
                    logger.warning("[RECONCILE] safe_positions_get returned None (query failure). Will skip orphan checks this cycle.")
                except Exception:
                    logger.warning("[RECONCILE] safe_positions_get returned None (query failure).")
                qd_positions = []
            else:
                qd_positions = _mt5_positions_as_qd_like_dicts(mt5_positions)
    except Exception as e:
        mt5_positions_query_failed = True
        logger.warning(f"[RECONCILE] MT5 positions_get raised exception (query failure): {e}")
        qd_positions = []

    result = {
        "db_count": len(db_trades),
        "qd_count": len(qd_positions),
        "fixes": 0,
        "mismatches": []
    }

    if qd_positions:
        # Post-entry strategy is owned by PostEntryManager.
        # reconciliation.py is kept ONLY for:
        # 1) SL-cross safety close
        # 2) DB <-> MT5 orphan synchronization
        #
        # All other post-entry exit/modify logic is intentionally disabled.

        # SL-cross safety only
        from config import POST_ENTRY_LOOP_INTERVAL_SEC

        post_entry_inactive_after_sec = POST_ENTRY_LOOP_INTERVAL_SEC * 3  # = 15 sec

        extreme_sl_gap_multiplier = 5.0  # safety net threshold: abs(price-sl) > 5x expected_atr


        for pos in qd_positions:
            try:
                order_id = str(pos.get("id", pos.get("ticket", "")) or "")
                if not order_id:
                    continue

                symbol = pos.get("symbol", "") or ""
                if not symbol:
                    continue

                direction_norm = _normalize_type_to_direction(pos.get("type", pos.get("direction", "")))
                sl = pos.get("sl", None)
                if sl is None:
                    continue

                try:
                    sl_val = float(sl)
                except Exception:
                    continue
                if sl_val <= 0:
                    continue

                current_price = float(pos.get("price_current", pos.get("price", 0)) or 0)
                if current_price <= 0:
                    continue

                stop_hit = (direction_norm == "buy" and current_price <= sl_val) or (
                    direction_norm == "sell" and current_price >= sl_val
                )
                if not stop_hit:
                    continue

                # SL-cross safety gating against PostEntryManager closing/SL-modifying concurrently.
                try:
                    post_entry_inactive_sec = seconds_since_last_beat()
                    post_entry_inactive = post_entry_inactive_sec > post_entry_inactive_after_sec
                except Exception:
                    post_entry_inactive_sec = float("inf")
                    post_entry_inactive = True

                # Safety net: allow SL-cross close if the price is far beyond SL
                # using expected_atr from execution dataset as a natural volatility scale.
                allow_extreme_gap = False

                try:
                    expected_row = get_execution_dataset(order_id) or {}
                except Exception:
                    expected_row = {}

                expected_atr = None
                try:
                    expected_atr = expected_row.get("expected_atr", None)
                    if expected_atr is not None:
                        expected_atr = float(expected_atr)
                except Exception:
                    expected_atr = None

                if expected_atr and expected_atr > 0:
                    # distance beyond SL in price units
                    if direction_norm == "buy":
                        sl_gap = sl_val - current_price
                    else:
                        sl_gap = current_price - sl_val
                    if sl_gap > expected_atr * extreme_sl_gap_multiplier:
                        allow_extreme_gap = True


                if not (post_entry_inactive or allow_extreme_gap):
                    logger.warning(
                        f"[STOPLOSS] Blocked SL-cross close by PostEntryManager safety gate "
                        f"order_id={order_id} inactive_sec={post_entry_inactive_sec:.1f} "
                        f"inactive_after={post_entry_inactive_after_sec} allow_extreme_gap={allow_extreme_gap}"
                    )
                    continue


                current_profit = float(pos.get("profit", pos.get("pnl", 0)) or 0)
                logger.warning(
                    f"[STOPLOSS] Closing {symbol} order_id={order_id} dir={direction_norm} "
                    f"price_current={current_price} sl={sl_val} pnl={current_profit}"
                )

                ok = _close_trade_mt5(order_id)
                if ok:
                    try:
                        upsert_execution_actual(
                            order_id=order_id,
                            actual_entry=float(pos.get("price_open", pos.get("price", 0)) or 0),
                            actual_exit=current_price,
                            actual_pnl=current_profit,
                            spread_at_entry=None,
                            slippage=None,
                            execution_delay_ms=None,
                            execution_quality_score=None,
                            price_gap=None,
                            actual_indicators_json=None,
                            exit_reason="Stop Loss breached",
                            exit_probability=None,
                        )
                    except Exception:
                        pass

                    close_trade_db_by_order_id(order_id, pnl=current_profit)
                    if not _is_recently_closed(order_id):
                        _mark_closed(order_id)
                    # Feed Risk Governor with realized PnL (SL-cross close)
                    _feed_risk_governor(order_id, current_profit)
            except Exception as e:
                logger.error(f"[STOPLOSS] error: {e}")

    # If MT5 query failed (None/exception), do NOT perform orphan detection this cycle.
    # This prevents misclassifying DB trades as orphans when the MT5 snapshot is unavailable.
    if mt5_positions_query_failed:
        logger.warning("[RECONCILE] MT5 positions_get query failed; skipping orphan checks for this cycle.")
        return result

    qd_ids = {str(p.get("id", p.get("ticket", ""))) for p in qd_positions}


    for trade in db_trades:
        trade_id = trade.get("id")
        order_id = str(trade.get("order_id", "") or "")

        if not order_id or order_id in qd_ids:
            continue

        if _is_recently_closed(order_id):
            continue

        # Check if trade was opened recently (within 90 seconds)
        # If so, skip orphan check - it's likely a timing gap, not actual orphan
        trade_open_time = trade.get("time_open", None) or trade.get("open_time", None)
        if trade_open_time is None:
            # Try from execution dataset
            try:
                expected_row = get_execution_dataset(order_id)
                if expected_row:
                    trade_open_time = expected_row.get("time_open", None) or expected_row.get("created_at", None)
            except Exception:
                pass

        if _was_opened_recently(trade_open_time):
            logger.info(f"Ignoring recent DB order (timing gap): order_id={order_id}")
            continue

        logger.warning(f"Orphan trade in DB (no longer in QD): trade_id={trade_id} order_id={order_id}")

        # Startup safety window: if trade was opened shortly after bot start,
        # skip orphan logic entirely to avoid false orphans during sync.
        try:
            trade_open_time = trade.get("time_open", None) or trade.get("open_time", None)
            if trade_open_time is not None:
                if _was_opened_recently(trade_open_time):
                    logger.info(f"Ignoring orphan (opened recently): order_id={order_id}")
                    continue
        except Exception:
            pass

        # Also guard based on bot runtime if DB lacks open time info.
        try:
            if _BOT_START_EPOCH is not None:
                opened_at_epoch = None
                try:
                    # if execution_dataset has time_open as text, _was_opened_recently can parse
                    opened_at_epoch = trade.get("opened_at", None)
                except Exception:
                    opened_at_epoch = None

                # If we cannot reliably parse, fallback to bot start window.
                # Treat any trade opened within 90s from bot start as non-orphan.
                if opened_at_epoch is None:
                    pass
                else:
                    # already handled by _was_opened_recently above when possible
                    pass

                # bot-start-only fallback
                now = time.time()
                if now - _BOT_START_EPOCH <= _ORPHAN_IGNORE_WINDOW_SEC:
                    logger.info(f"Ignoring orphan during bot warmup window: order_id={order_id}")
                    continue
        except Exception:
            pass

        # a) If MT5 still has the position by ticket, ignore (it may be just a sync gap).
        if _mt5_position_exists_by_ticket(order_id):
            logger.info(f"Ignoring orphan (still exists on MT5): order_id={order_id}")
            # reset orphan counter because it reappeared
            orphan_tracker.pop(str(order_id), None)
            continue

        # b) Orphan pending confirmation cycles
        oid = str(order_id)
        tracker = orphan_tracker.get(oid) or {"count": 0, "last_seen": 0.0}
        tracker_count = int(tracker.get("count", 0) or 0) + 1
        tracker["count"] = tracker_count
        tracker["last_seen"] = time.time()
        orphan_tracker[oid] = tracker

        if tracker_count < _ORPHAN_CONFIRM_CYCLES:
            logger.warning(
                f"Orphan trade pending confirmation cycle {tracker_count}/{_ORPHAN_CONFIRM_CYCLES}: order_id={order_id}"
            )
            result["mismatches"].append(
                f"DB orphan pending: trade_id {trade_id} order_id {order_id} cycle {tracker_count}/{_ORPHAN_CONFIRM_CYCLES}"
            )
            continue

        # c) Confirmed orphan: close DB row (still keep exit_reason different and avoid exit_price=0 unless known)
        logger.warning(
            f"[RECONCILE] Orphan confirmed after {_ORPHAN_CONFIRM_CYCLES} cycles; closing DB: order_id={order_id}"
        )

        expected_row = None
        try:
            expected_row = get_execution_dataset(order_id)
        except Exception:
            expected_row = None

        actual_exit_expected = None
        actual_pnl_expected = None
        if isinstance(expected_row, dict):
            actual_exit_expected = expected_row.get("actual_exit", None)
            actual_pnl_expected = expected_row.get("actual_pnl", None)

        has_actual_close = (
            (actual_exit_expected is not None and float(actual_exit_expected) != 0) or
            (actual_pnl_expected is not None and float(actual_pnl_expected) != 0)
        )

        # Prefer pnl from execution_dataset if present; else keep pnl=0 (but don't send misleading exit_price)
        pnl_to_use = 0.0
        if has_actual_close and isinstance(expected_row, dict):
            try:
                pnl_to_use = float(expected_row.get("actual_pnl", 0) or 0)
            except Exception:
                pnl_to_use = 0.0

        try:
            ok = close_trade_db_by_order_id(order_id, pnl=pnl_to_use)
        except Exception:
            ok = False

        if ok:
            _mark_closed(order_id)
            orphan_tracker.pop(oid, None)
            # Feed Risk Governor with realized PnL (orphan/manual close at broker)
            _feed_risk_governor(order_id, float(pnl_to_use or 0))

            expected_entry = expected_row.get("expected_entry", None) if expected_row else None
            actual_entry = float(trade.get("entry_price", 0) or 0)

            actual_exit = 0
            slippage = _calc_slippage_for_direction(
                expected_entry,
                actual_entry,
                _normalize_type_to_direction(trade.get("direction", "")),
            )

            upsert_execution_actual(
                order_id=order_id,
                actual_entry=actual_entry,
                actual_exit=actual_exit,
                actual_pnl=float(pnl_to_use or 0),
                spread_at_entry=None,
                slippage=slippage,
                execution_delay_ms=None,
                execution_quality_score=None,
                price_gap=None,
                actual_indicators_json=None,
                exit_reason="Orphan confirmed (reconciled DB only)",
                exit_probability=None,
            )

            # Avoid exit_price=0 in message: if we don't know it, keep it 0 but make reason explicit.
            notify_trade_closed(
                symbol=trade.get("symbol", ""),
                direction=_normalize_type_to_direction(trade.get("direction", "")),
                pnl=float(pnl_to_use or 0),
                reason="⚠️ Orphan trade: closed after confirmation cycles (DB↔MT5 sync).",
                size=trade.get("size", 0),
                entry=actual_entry,
                exit_price=0
            )
            result["fixes"] += 1
            result["mismatches"].append(
                f"DB orphan closed after confirmation: trade_id {trade_id} order_id {order_id}"
            )
        else:
            result["mismatches"].append(
                f"DB orphan confirmed but not closed: trade_id {trade_id} order_id {order_id}"
            )

    db_order_ids = {str(t.get("order_id", "") or "") for t in db_trades}
    for pos in qd_positions:
        pid = str(pos.get("id", pos.get("ticket", "")))
        if pid and pid not in db_order_ids:
            # Check if position was opened recently (within 90 seconds)
            # If so, skip orphan check - it's likely a timing gap
            pos_open_time = pos.get("time_open", None) or pos.get("open_time", None)
            if _was_opened_recently(pos_open_time):
                logger.info(f"Ignoring recent QD position (timing gap): pid={pid}")
                continue

            logger.warning(f"Unknown position in QD: {pid} {pos.get('symbol')}")
            result["mismatches"].append(f"QD orphan: {pid}")

    if result["fixes"] > 0 or result["mismatches"]:
        logger.info(f"Reconciliation: {result}")

    return result


def reconciliation_loop(interval: int = 60):
    logger.info(f"Reconciliation loop started (every {interval}s)")
    while True:
        try:
            reconcile()
        except Exception as e:
            logger.error(f"Reconciliation error: {e}")
        time.sleep(interval)


def start_reconciliation(interval: int = 60):
    # Reconciliation remains for DB/MT5 orphan sync + SL-cross safety only.
    threading.Thread(target=reconciliation_loop, args=(interval,), daemon=True).start()
    logger.info("Reconciliation thread started (safety only)")




