# Trading Bot V3 - risk/risk_engine.py
# Central risk engine: can_trade check + correlation filter

from utils.logger import get_logger
from data.storage.database import get_daily_stats, get_open_trades, get_total_open_trades

from config import MIN_SCORE, AI_MIN_CONFIDENCE, MAX_OPEN_TRADES, STOP_AFTER_LOSSES, MAX_CORRELATION
from risk.drawdown import check_drawdown
from risk.position_sizing import get_dynamic_risk_multiplier
from core.exceptions import RiskLimitError
from config import MIN_SCORE, AI_MIN_CONFIDENCE, MAX_OPEN_TRADES, STOP_AFTER_LOSSES, MAX_CORRELATION, MAX_DAILY_LOSS_USD

logger = get_logger("risk_engine")

# Correlation groups: pairs that move together
CORRELATION_GROUPS = {
    "USD_SHORT": ["EURUSD", "GBPUSD", "XAUUSD"],  # All benefit from weak USD
    "USD_LONG": ["USDJPY", "USDCAD", "USDCHF"],    # All benefit from strong USD
}

def check_correlation(symbol: str, direction: str) -> tuple:
    """Check if new trade adds too much correlation to existing positions"""
    open_trades = get_open_trades()
    if not open_trades:
        return True, ""
    
    # Find which group this trade belongs to
    new_group = None
    for group_name, symbols in CORRELATION_GROUPS.items():
        if symbol in symbols:
            new_group = group_name
            break
    
    if not new_group:
        return True, ""  # Unknown group, allow
    
    # Check existing positions in same group
    same_direction_count = 0
    for trade in open_trades:
        trade_symbol = trade.get("symbol", "")
        trade_direction = trade.get("direction", "")
        if trade_symbol in CORRELATION_GROUPS.get(new_group, []):
            if trade_direction == direction:
                same_direction_count += 1
    
    # If 2+ positions in same direction in correlated group, block
    if same_direction_count >= 2:
        return False, f"Correlation limit: {same_direction_count} positions in {new_group}"
    
    return True, ""

def can_trade(symbol: str, direction: str, final_score: float,
              ai_confidence: float, equity: float) -> tuple:
    """Complete risk check before opening a trade"""

    # 1. Duplicate check (MT5-only) - block if symbol already has an open position
    # Requirement: use mt5.positions_get(symbol=symbol) and block regardless of direction.
    #
    # M-04: this used to say "allow trading if MT5 query fails" and treated
    # positions_get() returning None the same as "zero positions" — but the MT5
    # API documents None as ambiguous between "no results" and "an error
    # occurred" (check last_error() to tell them apart). Either path silently
    # skipped the one check that exists to stop a second position from opening
    # on a symbol that already has one, exactly when MT5 is least reliable. A
    # duplicate check that could not be verified is not a duplicate check that
    # passed, so both cases now reject the trade instead. Routed through the
    # shared session lock (data.market.mt5_session) rather than a raw import,
    # consistent with every other live MT5 read in this codebase.
    try:
        from data.market.mt5_session import ensure_session, mt5, mt5_call

        if not ensure_session():
            return False, "Duplicate check unavailable: MT5 session down"

        with mt5_call():
            positions = mt5.positions_get(symbol=symbol)
            last_err = mt5.last_error() if positions is None else None

        if positions is None:
            reason = f"Duplicate check unavailable: positions_get returned None ({last_err})"
            logger.error(f"Risk: {reason} ({symbol})")
            return False, reason

        if len(positions) > 0:
            logger.info(
                f"Risk: Duplicate symbol blocked via MT5 - {symbol} already has an open position"
            )
            return False, "Duplicate symbol blocked via MT5"
    except Exception as e:
        reason = f"Duplicate check failed: {e}"
        logger.error(f"Risk: {reason} ({symbol})")
        return False, reason



    # 2. Score check
    if final_score < MIN_SCORE:
        reason = f"Score {final_score:.1f} < {MIN_SCORE}"
        logger.info(f"Risk: Score too low ({symbol} {direction}) - {reason}")
        return False, reason

    # 2. AI confidence check
    if ai_confidence < AI_MIN_CONFIDENCE:
        reason = f"AI confidence {ai_confidence:.2f} < {AI_MIN_CONFIDENCE}"
        logger.info(f"Risk: AI confidence too low ({symbol} {direction}) - {reason}")
        return False, reason

    # 3. Daily loss check
    stats = get_daily_stats()
    starting_balance = equity - stats.get("total_pnl", 0)
    daily_loss_pct = abs(stats.get("total_pnl", 0)) / max(starting_balance, 1)
    if stats.get("total_pnl", 0) < 0 and daily_loss_pct >= 0.03:
        reason = f"Daily loss limit: {daily_loss_pct:.1%}"
        logger.warning(f"Risk: Daily loss limit exceeded ({symbol} {direction}) - {reason}")
        return False, reason

    # 4. Drawdown check
    dd_result = check_drawdown(stats.get("total_pnl", 0), equity)
    if dd_result["action"] in ["halt_day", "full_stop"]:
        reason = dd_result["reason"]
        logger.warning(f"Risk: Drawdown limit exceeded ({symbol} {direction}) - {reason}")
        return False, reason

    # 5. Max open trades
    open_count = get_total_open_trades()
    if open_count >= MAX_OPEN_TRADES:
        reason = f"Max open trades: {open_count}"
        logger.warning(f"Risk: Max open trades reached ({symbol} {direction}) - {reason}")
        return False, reason

    # 6. Consecutive losses
    if stats.get("consecutive_losses", 0) >= STOP_AFTER_LOSSES:
        reason = f"Consecutive losses: {stats['consecutive_losses']}"
        logger.warning(f"Risk: Consecutive losses limit exceeded ({symbol} {direction}) - {reason}")
        return False, reason

    # 7. Correlation filter (existing simple groups)
    corr_ok, corr_reason = check_correlation(symbol, direction)
    if not corr_ok:
        logger.info(f"Risk: Correlation filter blocked ({symbol} {direction}) - {corr_reason}")
        return False, corr_reason

    # 7b. Correlation Protection module (optional finer control)
    #
    # is_correlated_open() already catches its own internal errors and returns
    # False ("not correlated"). What used to reach this except block was
    # therefore only the two things it *doesn't* guard: a broken import, or
    # get_open_trades() failing against a locked/corrupt DB — and both were
    # swallowed into `return True, "OK"`, i.e. a database failure silently
    # became risk-check approval. A gate that cannot be evaluated is not a
    # gate that passed.
    try:
        from execution.risk_management.correlation_protection import is_correlated_open

        # open positions from DB
        open_positions = get_open_trades() or []
        if is_correlated_open(symbol=symbol, direction=direction, open_positions=open_positions):
            reason = "CorrelationProtection: correlated open trade exists"
            logger.info(f"Risk: CorrelationProtection blocked ({symbol} {direction})")
            return False, reason
    except Exception as exc:
        reason = f"CorrelationProtection check failed: {exc}"
        logger.error(f"Risk: {reason} ({symbol} {direction})")
        return False, reason

    return True, "OK"


