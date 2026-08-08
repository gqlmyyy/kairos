# Trading Bot V3 - data/storage/database.py
# SQLite database with WAL mode - all tables

import sqlite3
from datetime import datetime
from typing import List, Optional
from utils.logger import get_logger
from config import DB_FILE, INITIAL_WEIGHTS

logger = get_logger("database")

# How long a writer waits for a competing transaction before giving up.
# WAL allows one writer at a time; this turns an instant "database is locked"
# into a bounded wait, which is what the concurrent writers actually need.
DB_BUSY_TIMEOUT_SEC = 10.0

def get_conn():
    # busy_timeout matters here: five threads write to this database (main
    # cycle, reconciliation, post-entry manager, telegram, feedback). Without
    # it, a concurrent writer raises "database is locked" immediately instead
    # of waiting for the current transaction to finish — and several call
    # sites swallow that exception, silently dropping the write.
    conn = sqlite3.connect(DB_FILE, timeout=DB_BUSY_TIMEOUT_SEC)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={int(DB_BUSY_TIMEOUT_SEC * 1000)}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row

    # Run migrations
    c = conn.cursor()
    for col, typ in [
        ("exit_reason", "TEXT"),
        ("exit_probability", "REAL"),
        ("time_open", "TEXT"),
        ("expected_tp", "REAL"),
        ("expected_sl", "REAL"),
        ("risk_reward_ratio", "REAL"),
        ("trade_duration", "TEXT"),
        ("expected_indicators_json", "TEXT"),
        ("actual_indicators_json", "TEXT"),
        # Persistent MFE/MAE (in R) across restarts, consumed by trade management
        # and by the exit-model training pipeline.
        ("model_type", "TEXT"),
        ("mfe", "REAL DEFAULT 0.0"),
        ("mae", "REAL DEFAULT 0.0"),
        # Trade profile chosen at entry (Layer 6). Fixed for the trade's life so
        # a restart or a config change cannot silently re-profile a live position.
        ("entry_profile", "TEXT"),
        # Volume at entry, so the partial-TP ladder stays anchored to the
        # original size as the position is scaled out.
        ("expected_volume", "REAL"),
        # Partial-TP ladder levels already taken, as a comma-separated list of
        # indices (e.g. "0,1"). Held in memory only until now, so a restart
        # re-armed a level that had already executed and took the partial twice.
        ("partial_levels_done", "TEXT"),
    ]:
        try:
            # NOTE: typ may include "DEFAULT 0.0" already
            c.execute(f"ALTER TABLE execution_dataset ADD COLUMN {col} {typ}")
        except:
            pass

    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()

    # Trades - comprehensive
    c.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL, direction TEXT NOT NULL,
        size REAL, entry_price REAL, stop_loss REAL,
        take_profit REAL, atr REAL,
        final_score REAL, ai_score REAL, ai_confidence REAL,
        trend_score REAL, momentum_score REAL,
        sentiment_score REAL, volatility_score REAL,
        reason TEXT, status TEXT DEFAULT 'open',
        order_id TEXT, pnl REAL DEFAULT 0,
        opened_at TEXT, closed_at TEXT
    )""")

    # === Live execution dataset (expected vs actual) ===
    # order_id is the QuantDinger/MT5 mapping key
    c.execute("""CREATE TABLE IF NOT EXISTS execution_dataset (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_created_at TEXT,
        dataset_updated_at TEXT,

        order_id TEXT UNIQUE,

        symbol TEXT,
        direction TEXT,

        -- =========================
        -- expected snapshot (open)
        -- =========================
        expected_entry REAL,
        expected_final_score REAL,

        expected_rsi REAL,
        expected_macd REAL,
        expected_session TEXT,
        expected_spread REAL,
        expected_atr REAL,

        expected_trend_strength REAL,
        expected_momentum_score REAL,
        expected_volatility_score REAL,
        expected_market_regime TEXT,

        expected_ai_score REAL,
        expected_sentiment_score REAL,
        expected_news_impact_score REAL,

        expected_ai_confidence REAL,
        expected_trend_score REAL,
        expected_momentum_score_legacy REAL, -- keep backward compatibility with older code if any
        expected_sentiment_score_legacy REAL,
        expected_volatility_score_legacy REAL,

        expected_indicators_json TEXT,

        -- =========================
        -- actual facts (close/reconcile)
        -- =========================
        actual_entry REAL,
        actual_exit REAL,
        actual_pnl REAL,

        actual_rsi REAL,
        actual_macd REAL,
        actual_session TEXT,
        actual_spread REAL,
        actual_atr REAL,

        actual_trend_strength REAL,
        actual_momentum_score REAL,
        actual_volatility_score REAL,
        actual_market_regime TEXT,

        actual_ai_score REAL,
        actual_sentiment_score REAL,
        actual_news_impact_score REAL,

        spread_at_entry REAL,
        slippage REAL,
        execution_delay_ms INTEGER,
        execution_quality_score REAL,
        price_gap REAL,
        actual_indicators_json TEXT,

        -- =========================
        -- BV3 safety state (per order)
        -- =========================
        breakeven_done INTEGER DEFAULT 0,
        trailing_done INTEGER DEFAULT 0,

        status TEXT DEFAULT 'open',  -- open|closed|orphaned

        exit_reason TEXT,
        exit_probability REAL,

        model_type TEXT
    )""")

    # Migration: add exit_reason column if missing
    try:
        c.execute("ALTER TABLE execution_dataset ADD COLUMN exit_reason TEXT")
    except:
        pass
    try:
        c.execute("ALTER TABLE execution_dataset ADD COLUMN exit_probability REAL")
    except:
        pass

    # Decisions log
    c.execute("""CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, direction TEXT,
        final_score REAL, ai_score REAL,
        trend_score REAL, momentum_score REAL,
        sentiment_score REAL, volatility_score REAL,
        ai_confidence REAL, confidence REAL,
        mtf_aligned INTEGER, regime TEXT,
        reason TEXT, action TEXT, decided_at TEXT
    )""")

    # News analysis
    c.execute("""CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        headline TEXT, symbol TEXT, source TEXT,
        impact_score REAL, bias TEXT, confidence REAL,
        source_weight REAL, decay REAL,
        is_high_impact INTEGER, analyzed_at TEXT
    )""")

    # Analysis results per symbol per cycle
    c.execute("""CREATE TABLE IF NOT EXISTS analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, cycle INTEGER,
        ai_score REAL, ai_bias TEXT, ai_confidence REAL,
        sentiment_score REAL, sentiment_direction TEXT,
        trend_score REAL, trend_direction TEXT,
        momentum_score REAL, momentum_direction TEXT,
        volatility_score REAL, regime TEXT,
        mtf_aligned INTEGER, analyzed_at TEXT
    )""")

    # Signals
    c.execute("""CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, direction TEXT, score REAL,
        confidence REAL, source TEXT, cycle INTEGER,
        reason TEXT, created_at TEXT
    )""")

    # Daily stats
    c.execute("""CREATE TABLE IF NOT EXISTS daily_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT UNIQUE,
        total_trades INTEGER DEFAULT 0,
        winning_trades INTEGER DEFAULT 0,
        losing_trades INTEGER DEFAULT 0,
        total_pnl REAL DEFAULT 0,
        consecutive_losses INTEGER DEFAULT 0,
        max_drawdown REAL DEFAULT 0,
        best_symbol TEXT DEFAULT '',
        worst_symbol TEXT DEFAULT ''
    )""")

    # Weights history
    c.execute("""CREATE TABLE IF NOT EXISTS weights_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        ai_weight REAL, trend_weight REAL,
        momentum_weight REAL, sentiment_weight REAL,
        volatility_weight REAL,
        updated_at TEXT
    )""")

    # Risk events log
    c.execute("""CREATE TABLE IF NOT EXISTS risk_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT, symbol TEXT,
        reason TEXT, details TEXT DEFAULT '',
        created_at TEXT
    )""")

    # Performance metrics
    c.execute("""CREATE TABLE IF NOT EXISTS performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        batch_size INTEGER,
        win_rate REAL,
        profit_factor REAL,
        avg_win REAL, avg_loss REAL,
        sharpe REAL,
        created_at TEXT
    )""")

    # Positions sync table
    c.execute("""CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        qd_id TEXT UNIQUE, symbol TEXT, direction TEXT,
        size REAL, entry_price REAL,
        stop_loss REAL, take_profit REAL,
        pnl REAL DEFAULT 0, synced_at TEXT
    )""")

    conn.commit()
    conn.close()
    logger.info("V3 Database initialized with all tables")

# === Trade CRUD ===

def save_trade(symbol, direction, size, entry_price, sl, tp, atr,
               final_score, ai_score, ai_confidence, reason, order_id="",
               trend_score=0, momentum_score=0, sentiment_score=0, volatility_score=0):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""INSERT INTO trades (symbol, direction, size, entry_price,
        stop_loss, take_profit, atr, final_score, ai_score, ai_confidence,
        trend_score, momentum_score, sentiment_score, volatility_score,
        reason, order_id, opened_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (symbol, direction, size, entry_price, sl, tp, atr,
         final_score, ai_score, ai_confidence,
         trend_score, momentum_score, sentiment_score, volatility_score,
         reason, order_id, datetime.now().isoformat()))
    trade_id = c.lastrowid
    conn.commit()
    conn.close()
    return trade_id

def close_trade_db(trade_id, pnl):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE trades SET status='closed', pnl=?, closed_at=? WHERE id=?",
              (pnl, datetime.now().isoformat(), trade_id))
    conn.commit()
    conn.close()
    update_daily_stats(pnl)

def close_trade_db_by_order_id(order_id: str, pnl: float):
    """
    Close a trade by QuantDinger order_id (matches trades.order_id).
    Used by reconciliation to ensure correct row is closed.
    """
    if order_id is None:
        return False

    order_id = str(order_id).strip()
    if not order_id:
        return False

    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "UPDATE trades SET status='closed', pnl=?, closed_at=? "
        "WHERE order_id=? AND status='open'",
        (pnl, datetime.now().isoformat(), order_id)
    )
    updated = c.rowcount
    conn.commit()
    conn.close()

    if updated and updated > 0:
        update_daily_stats(pnl)
        return True

    return False

# === Execution dataset CRUD ===

def upsert_execution_expected(
    order_id: str,
    symbol: str,
    direction: str,
    expected_entry: float,
    expected_final_score: float,
    expected_ai_score: float,
    expected_ai_confidence: float,
    expected_trend_score: float,
    expected_momentum_score: float,
    expected_sentiment_score: float,
    expected_volatility_score: float,
    expected_rsi: float = None,
    expected_macd: float = None,
    expected_session: str = None,
    expected_spread: float = None,
    expected_atr: float = None,
    expected_trend_strength: float = None,
    expected_market_regime: str = None,
    expected_news_impact_score: float = None,
    expected_momentum_score_legacy: float = None,
    expected_sentiment_score_legacy: float = None,
    expected_volatility_score_legacy: float = None,
    expected_ai_score_legacy: float = None,
    expected_indicators_json: str = None,
    expected_sl: float = None,
    expected_tp: float = None,
    expected_volume: float = None,
    entry_profile: str = None,
    strategy: str = "V3"
):
    """
    Insert/update expected snapshot for an order_id.
    Uses entry_price returned from QuantDinger as expected_entry (per your approval).
    """
    if order_id is None:
        return False
    order_id = str(order_id).strip()
    if not order_id:
        return False

    now = datetime.now().isoformat()
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        INSERT INTO execution_dataset (
            dataset_created_at, dataset_updated_at,
            order_id, symbol, direction,
            expected_entry, expected_final_score,

            expected_rsi, expected_macd, expected_session,
            expected_spread, expected_atr,

            expected_trend_strength, expected_momentum_score, expected_volatility_score, expected_market_regime,

            expected_ai_score, expected_sentiment_score, expected_news_impact_score,

            expected_ai_confidence, expected_trend_score,

            expected_momentum_score_legacy, expected_sentiment_score_legacy, expected_volatility_score_legacy,

            expected_indicators_json,
            expected_sl, expected_tp, expected_volume, entry_profile,
            status
        ) VALUES (
            ?,?,
            ?,?,?,?,
            ? , ? , ?,
            ? , ?,
            ?,?,?,?,
            ?,?,?,
            ?,?,
            ?,?,?,?,
            ?,
            ?,?,?,?,
            'open'
        )
        ON CONFLICT(order_id) DO UPDATE SET
            dataset_updated_at=excluded.dataset_updated_at,
            symbol=excluded.symbol,
            direction=excluded.direction,

            expected_entry=excluded.expected_entry,
            expected_final_score=excluded.expected_final_score,

            expected_rsi=excluded.expected_rsi,
            expected_macd=excluded.expected_macd,
            expected_session=excluded.expected_session,
            expected_spread=excluded.expected_spread,
            expected_atr=excluded.expected_atr,

            expected_trend_strength=excluded.expected_trend_strength,
            expected_momentum_score=excluded.expected_momentum_score,
            expected_volatility_score=excluded.expected_volatility_score,
            expected_market_regime=excluded.expected_market_regime,

            expected_ai_score=excluded.expected_ai_score,
            expected_sentiment_score=excluded.expected_sentiment_score,
            expected_news_impact_score=excluded.expected_news_impact_score,

            expected_ai_confidence=excluded.expected_ai_confidence,
            expected_trend_score=excluded.expected_trend_score,

            expected_momentum_score_legacy=excluded.expected_momentum_score_legacy,
            expected_sentiment_score_legacy=excluded.expected_sentiment_score_legacy,
            expected_volatility_score_legacy=excluded.expected_volatility_score_legacy,

            expected_indicators_json=excluded.expected_indicators_json,

            -- Never destroy a value that is already recorded for this order.
            -- expected_tp in particular is populated for 504 historically
            -- imported rows; a live upsert that collides with one of them must
            -- not overwrite the imported figure. COALESCE keeps the first
            -- non-null write and lets later ones only fill genuine gaps.
            expected_sl=COALESCE(execution_dataset.expected_sl, excluded.expected_sl),
            expected_tp=COALESCE(execution_dataset.expected_tp, excluded.expected_tp),
            expected_volume=COALESCE(execution_dataset.expected_volume, excluded.expected_volume),
            entry_profile=COALESCE(execution_dataset.entry_profile, excluded.entry_profile),
            status='open'
    """, (
        now, now,
        order_id, symbol, direction,
        expected_entry, expected_final_score,

        # expected_rsi, expected_macd, expected_session
        expected_rsi, expected_macd, expected_session,

        # expected_spread, expected_atr
        expected_spread, expected_atr,

        # expected_trend_strength, expected_momentum_score, expected_volatility_score, expected_market_regime
        expected_trend_strength, expected_momentum_score, expected_volatility_score, expected_market_regime,

        # expected_ai_score, expected_sentiment_score, expected_news_impact_score
        expected_ai_score, expected_sentiment_score, expected_news_impact_score,

        # expected_ai_confidence, expected_trend_score
        expected_ai_confidence, expected_trend_score,

        # expected_momentum_score_legacy, expected_sentiment_score_legacy, expected_volatility_score_legacy
        expected_momentum_score_legacy, expected_sentiment_score_legacy, expected_volatility_score_legacy,

        expected_indicators_json,

        # expected_sl/tp: previously never written by the live path, which left
        # every downstream R-multiple calculation without a risk denominator.
        # expected_volume anchors the partial-TP ladder to the entry size.
        expected_sl, expected_tp, expected_volume, entry_profile
    ))

    conn.commit()
    conn.close()
    return True

def update_partial_levels_done(order_id: str, levels) -> bool:
    """Persist which partial-TP ladder levels have already executed.

    Stored as a sorted comma-separated index list ("0,1"). This must be written
    immediately after the broker confirms a partial close: the state lived only
    in memory before, so a restart re-armed a level that had already been taken
    and closed part of the position a second time.

    Returns True when a row was updated.
    """
    if order_id is None:
        return False
    order_id = str(order_id).strip()
    if not order_id:
        return False

    encoded = ",".join(str(int(level)) for level in sorted(set(levels or ())))

    # get_conn() is inside the try on purpose: the broker has already executed
    # the partial by the time this is called, so a database failure must be
    # reported to the caller as False — never raised into the management loop,
    # where it would abort the rest of the cycle for an already-done action.
    conn = None
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute(
            "UPDATE execution_dataset SET partial_levels_done=?, dataset_updated_at=? "
            "WHERE order_id=?",
            (encoded, datetime.now().isoformat(), order_id),
        )
        conn.commit()
        return c.rowcount > 0
    except Exception as exc:
        logger.error(
            "[DB] update_partial_levels_done failed order_id=%s: %s", order_id, exc
        )
        return False
    finally:
        if conn is not None:
            conn.close()


def update_breakeven_done(order_id: str, done: bool = True) -> bool:
    """Persist that the stop has been moved to break-even for this trade.

    The column existed and was read back on startup, but nothing ever wrote it:
    the live path only set an in-memory flag. A restart therefore re-armed
    break-even on a trade whose stop was already there. Harmless in isolation
    (the modify filter rejects a non-improving stop), but it made the restored
    state a lie, and the same gap did cause duplicate partial closes.
    """
    if order_id is None:
        return False
    order_id = str(order_id).strip()
    if not order_id:
        return False

    conn = None
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute(
            "UPDATE execution_dataset SET breakeven_done=?, dataset_updated_at=? "
            "WHERE order_id=?",
            (1 if done else 0, datetime.now().isoformat(), order_id),
        )
        conn.commit()
        return c.rowcount > 0
    except Exception as exc:
        logger.error("[DB] update_breakeven_done failed order_id=%s: %s", order_id, exc)
        return False
    finally:
        if conn is not None:
            conn.close()


def parse_partial_levels_done(raw) -> set:
    """Decode the stored ladder state back into a set of indices.

    Tolerant by design: an unparsable value yields an empty set, which re-arms
    the ladder. That is the conservative direction — worst case a level is
    retaken, whereas inventing indices could skip protection entirely.
    """
    if not raw:
        return set()
    levels = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            levels.add(int(part))
        except ValueError:
            logger.warning("[DB] ignoring unparsable partial level %r", part)
    return levels


def update_execution_mfe_mae(order_id: str, mfe: float, mae: float) -> bool:
    """Persist max favorable excursion (mfe) and max adverse excursion (mae) for an open trade.

    Called from PostEntry cycles before RedFlagDetector runs.
    """
    if order_id is None:
        return False
    order_id = str(order_id).strip()
    if not order_id:
        return False

    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute(
            "UPDATE execution_dataset SET mfe=?, mae=?, dataset_updated_at=? WHERE order_id=?",
            (float(mfe or 0.0), float(mae or 0.0), datetime.now().isoformat(), order_id),
        )
        conn.commit()
        updated = c.rowcount
        conn.close()
        return updated > 0
    except Exception:
        conn.close()
        return False


def upsert_execution_actual(
    order_id: str,
    actual_entry: float,
    actual_exit: float,
    actual_pnl: float,
    spread_at_entry: float = None,
    slippage: float = None,
    execution_delay_ms: int = None,
    execution_quality_score: float = None,
    price_gap: float = None,
    actual_indicators_json: str = None,
    exit_reason: str = None,
    exit_probability: float = None
):
    if order_id is None:
        return False
    order_id = str(order_id).strip()
    if not order_id:
        return False

    now = datetime.now().isoformat()
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        INSERT INTO execution_dataset (
            dataset_created_at, dataset_updated_at,
            order_id, actual_entry, actual_exit, actual_pnl,
            spread_at_entry, slippage, execution_delay_ms,
            execution_quality_score, price_gap, actual_indicators_json,
            breakeven_done, trailing_done,
            status,
            exit_reason, exit_probability
        ) VALUES (
            ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            COALESCE((SELECT breakeven_done FROM execution_dataset WHERE order_id=?),0),
            COALESCE((SELECT trailing_done FROM execution_dataset WHERE order_id=?),0),
            'closed',
            ?, ?
        )
        ON CONFLICT(order_id) DO UPDATE SET
            dataset_updated_at=excluded.dataset_updated_at,
            actual_entry=excluded.actual_entry,
            actual_exit=excluded.actual_exit,
            actual_pnl=excluded.actual_pnl,
            spread_at_entry=excluded.spread_at_entry,
            slippage=excluded.slippage,
            execution_delay_ms=excluded.execution_delay_ms,
            execution_quality_score=excluded.execution_quality_score,
            price_gap=excluded.price_gap,
            actual_indicators_json=excluded.actual_indicators_json,
            breakeven_done=excluded.breakeven_done,
            trailing_done=excluded.trailing_done,
            status='closed',
            exit_reason=excluded.exit_reason,
            exit_probability=excluded.exit_probability
    """, (
        now, now,
        order_id,
        actual_entry, actual_exit, actual_pnl,
        spread_at_entry, slippage, execution_delay_ms,
        execution_quality_score, price_gap, actual_indicators_json,
        order_id, order_id,
        exit_reason, exit_probability
    ))

    conn.commit()
    conn.close()
    return True

def get_execution_dataset(order_id: str):
    if not order_id:
        return None
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM execution_dataset WHERE order_id=?", (str(order_id).strip(),))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def get_open_trades():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM trades WHERE status='open'")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def get_trade_by_id(trade_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM trades WHERE id=?", (trade_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def is_symbol_open(symbol, direction):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id FROM trades WHERE symbol=? AND direction=? AND status='open'",
              (symbol, direction))
    row = c.fetchone()
    conn.close()
    return row is not None

def get_total_open_trades():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM trades WHERE status='open'")
    count = c.fetchone()[0]
    conn.close()
    return count

# === Daily Stats ===

def get_daily_stats():
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM daily_stats WHERE date=?", (today,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "date": row["date"], "total_trades": row["total_trades"],
            "winning_trades": row["winning_trades"],
            "losing_trades": row["losing_trades"],
            "total_pnl": row["total_pnl"],
            "consecutive_losses": row["consecutive_losses"],
            "max_drawdown": row["max_drawdown"],
            "best_symbol": row["best_symbol"],
            "worst_symbol": row["worst_symbol"]
        }
    return {"date": today, "total_trades": 0, "winning_trades": 0,
            "losing_trades": 0, "total_pnl": 0.0,
            "consecutive_losses": 0, "max_drawdown": 0.0,
            "best_symbol": "", "worst_symbol": ""}

def update_daily_stats(pnl):
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM daily_stats WHERE date=?", (today,))
    row = c.fetchone()
    if row:
        wins = row["winning_trades"] + (1 if pnl > 0 else 0)
        losses = row["losing_trades"] + (1 if pnl < 0 else 0)
        consecutive = 0 if pnl > 0 else row["consecutive_losses"] + 1
        new_pnl = row["total_pnl"] + pnl
        dd = min(row["max_drawdown"], new_pnl) if new_pnl < 0 else row["max_drawdown"]
        c.execute("""UPDATE daily_stats SET total_trades=total_trades+1,
            winning_trades=?, losing_trades=?, total_pnl=?,
            consecutive_losses=?, max_drawdown=? WHERE date=?""",
            (wins, losses, new_pnl, consecutive, dd, today))
    else:
        c.execute("""INSERT INTO daily_stats (date, total_trades, winning_trades,
            losing_trades, total_pnl, consecutive_losses, max_drawdown)
            VALUES (?,1,?,?,?,?,0)""",
            (today, 1 if pnl > 0 else 0, 1 if pnl < 0 else 0,
             pnl, 0 if pnl > 0 else 1))
    conn.commit()
    conn.close()

# === Decisions ===

def save_decision(symbol, direction, scores, ai_confidence, confidence,
                  mtf_aligned, regime, reason, action):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""INSERT INTO decisions (symbol, direction,
        final_score, ai_score, trend_score, momentum_score,
        sentiment_score, volatility_score, ai_confidence, confidence,
        mtf_aligned, regime, reason, action, decided_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (symbol, direction,
         scores.get("final", 0), scores.get("ai", 0),
         scores.get("trend", 0), scores.get("momentum", 0),
         scores.get("sentiment", 0), scores.get("volatility", 0),
         ai_confidence, confidence,
         1 if mtf_aligned else 0, regime,
         reason, action, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_last_decisions(limit=10):
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute("""SELECT symbol, direction, final_score, reason,
            action, decided_at FROM decisions
            ORDER BY decided_at DESC LIMIT ?""", (limit,))
        rows = [dict(r) for r in c.fetchall()]
    except:
        rows = []
    conn.close()
    return rows

# === Weights ===

def get_weights(symbol):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""SELECT * FROM weights_history WHERE symbol=?
        ORDER BY id DESC LIMIT 1""", (symbol,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"ai": row["ai_weight"], "trend": row["trend_weight"],
                "momentum": row["momentum_weight"],
                "sentiment": row["sentiment_weight"],
                "volatility": row["volatility_weight"]}
    return INITIAL_WEIGHTS.copy()

def save_weights(symbol, weights):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""INSERT INTO weights_history (symbol,
        ai_weight, trend_weight, momentum_weight,
        sentiment_weight, volatility_weight, updated_at)
        VALUES (?,?,?,?,?,?,?)""",
        (symbol, weights["ai"], weights["trend"],
         weights["momentum"], weights["sentiment"],
         weights["volatility"], datetime.now().isoformat()))
    conn.commit()
    conn.close()

# === Risk events ===

def save_risk_event(event_type, symbol, reason, details=""):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""INSERT INTO risk_events (event_type, symbol, reason, details, created_at)
        VALUES (?,?,?,?,?)""",
        (event_type, symbol, reason, details, datetime.now().isoformat()))
    conn.commit()
    conn.close()

# === Positions sync ===

def sync_position(qd_id, symbol, direction, size, entry_price, sl, tp, pnl):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO positions
        (qd_id, symbol, direction, size, entry_price,
         stop_loss, take_profit, pnl, synced_at)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (qd_id, symbol, direction, size, entry_price,
         sl, tp, pnl, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_synced_positions():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM positions")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def clear_positions():
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM positions")
    conn.commit()
    conn.close()

# === Performance ===

def save_performance(symbol, batch_size, win_rate, profit_factor,
                     avg_win, avg_loss, sharpe):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""INSERT INTO performance (symbol, batch_size,
        win_rate, profit_factor, avg_win, avg_loss, sharpe, created_at)
        VALUES (?,?,?,?,?,?,?,?)""",
        (symbol, batch_size, win_rate, profit_factor,
         avg_win, avg_loss, sharpe, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_recent_trades(symbol, limit=20):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""SELECT * FROM trades WHERE symbol=? AND status='closed'
        ORDER BY closed_at DESC LIMIT ?""", (symbol, limit))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows
# === Symbol Performance Analysis ===

# Note: keep only ONE implementation to avoid Python overriding issues.

def get_symbol_performance(symbol: str, limit: int = 50) -> dict:
    """تحليل أداء زوج معين"""
    conn = get_conn()
    c = conn.cursor()
    c.execute("""SELECT pnl, direction, final_score, ai_confidence
        FROM trades WHERE symbol=? AND status='closed'
        ORDER BY closed_at DESC LIMIT ?""", (symbol, limit))
    rows = c.fetchall()
    conn.close()

    if not rows:
        return {
            "symbol": symbol, "total": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "total_pnl": 0.0, "avg_win": 0.0,
            "avg_loss": 0.0, "profit_factor": 0.0,
            "best_trade": 0.0, "worst_trade": 0.0, "avg_score": 0.0
        }

    pnls = [row[0] for row in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    scores = [row[2] for row in rows if row[2]]

    return {
        "symbol": symbol,
        "total": len(pnls),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        "total_pnl": round(sum(pnls), 2),
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
        "profit_factor": round(abs(sum(wins) / sum(losses)), 2) if losses and sum(losses) != 0 else 0,
        "best_trade": round(max(pnls), 2),
        "worst_trade": round(min(pnls), 2),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0
    }

def get_all_symbols_performance() -> list:
    from config import SYMBOLS
    results = [get_symbol_performance(s) for s in SYMBOLS]
    results.sort(key=lambda x: x["total_pnl"], reverse=True)
    return results

def get_best_worst_symbols() -> dict:
    performances = get_all_symbols_performance()
    with_trades = [p for p in performances if p["total"] > 0]
    if not with_trades:
        return {"best": None, "worst": None}
    return {
        "best": max(with_trades, key=lambda x: x["total_pnl"]),
        "worst": min(with_trades, key=lambda x: x["total_pnl"])
    }

