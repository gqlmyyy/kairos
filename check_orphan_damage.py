from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from config import DB_FILE


def _get_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    return [r[1] for r in rows]


def _try_parse_dt(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    if isinstance(s, datetime):
        return s
    if isinstance(s, (int, float)):
        try:
            return datetime.fromtimestamp(float(s))
        except Exception:
            return None
    if isinstance(s, str):
        ss = s.strip()
        if not ss:
            return None
        # try common iso
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ]:
            try:
                return datetime.strptime(ss[: len(fmt)], fmt)
            except Exception:
                continue
        try:
            # last resort: fromisoformat
            return datetime.fromisoformat(ss)
        except Exception:
            return None
    return None


def _find_timestamp_col(cols: List[str]) -> Optional[str]:
    # prefer closed_at/opened_at
    for c in ["closed_at", "closed_time", "closedAt", "time_closed", "dataset_updated_at", "updated_at", "opened_at", "open_time", "time_open"]:
        if c in cols:
            return c
    return None


def _find_exit_reason_col(cols: List[str]) -> Optional[str]:
    for c in ["exit_reason", "reason", "close_reason", "exitReason"]:
        if c in cols:
            return c
    return None


def _find_order_id_col(cols: List[str]) -> Optional[str]:
    for c in ["order_id", "ticket", "orderId", "qd_order_id", "position_id"]:
        if c in cols:
            return c
    return None


def _find_trades_table(conn: sqlite3.Connection) -> Optional[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    # prefer exact 'trades'
    if "trades" in tables:
        return "trades"
    # fallback: something that looks like trades
    for t in tables:
        tl = t.lower()
        if "trade" in tl and "exec" not in tl:
            return t
    return None


def _compute_bucket_coverage(conn: sqlite3.Connection, table: str, ts_col: str, rows: List[sqlite3.Row], n_days: int) -> Tuple[int, int]:
    now = datetime.now()
    recent_cutoff = now - timedelta(days=n_days)
    recent = 0
    total = 0
    for r in rows:
        total += 1
        dt = _try_parse_dt(r[ts_col])
        if dt is None:
            continue
        if dt >= recent_cutoff:
            recent += 1
    return recent, total


def main(days: int = 30, limit_list: int = 200) -> None:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row

    trades_table = _find_trades_table(conn)
    if not trades_table:
        raise RuntimeError("Could not find trades-like table in DB")

    cols = _get_table_columns(conn, trades_table)
    ts_col = _find_timestamp_col(cols)
    order_col = _find_order_id_col(cols)
    exit_reason_col = _find_exit_reason_col(cols)

    print(f"DB_FILE={DB_FILE}")
    print(f"TABLE(trades-like)={trades_table}")
    print("Detected columns:")
    print(f"  ts_col={ts_col}")
    print(f"  order_col={order_col}")
    print(f"  exit_reason_col={exit_reason_col}")
    print("")

    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {trades_table}")
    rows = cur.fetchall()
    print(f"Total rows in {trades_table}: {len(rows)}")
    print("")

    # orphan heuristic:
    # In reconciliation.py orphan uses: close_trade_db_by_order_id(order_id,pnl=0)
    # and does NOT update trades.exit_reason directly.
    # So we use exit_reason/reason in trades if present; otherwise fallback to execution_dataset exit_reason.
    # We'll try both execution_dataset first (more reliable: upsert_execution_actual sets exit_reason).

    if exit_reason_col is None or ts_col is None or order_col is None:
        print("Insufficient columns in trades table for orphan filtering; switching to execution_dataset-based detection.")

    # Step A: execution_dataset-based orphan
    exec_cols = _get_table_columns(conn, "execution_dataset")
    exec_order_col = _find_order_id_col(exec_cols) or "order_id"
    exec_ts_col = _find_timestamp_col(exec_cols) or "dataset_updated_at"
    exec_exit_reason_col = _find_exit_reason_col(exec_cols) or "exit_reason"

    # execution_dataset contains 'status' and exit_reason; reconciliation orphan writes upsert_execution_actual with exit_reason? NOT in code.
    # but reconciliation orphan calls notify_trade_closed with reason "🎯 تم تنفيذ SL/TP (reconciled)".
    # In database.py, notify_trade_closed likely writes to trades.reason (not execution_dataset exit_reason).
    # Therefore we search for orphan via combinations:
    # - trades.status='closed'
    # - trades.reason contains 'reconciled' or 'orphan'
    # - OR execution_dataset.status='closed' and (exit_reason contains 'reconciled' OR status='orphan')

    print("---- execution_dataset scan ----")
    # detect status col
    status_col = "status" if "status" in exec_cols else None

    # find closed rows
    where = ["1=1"]
    if "status" in exec_cols:
        where.append("status='closed'")
    q = f"SELECT {exec_order_col} as order_id, {exec_exit_reason_col} as exit_reason, dataset_updated_at, dataset_created_at, actual_pnl, {status_col if status_col else 'NULL'} as status_col FROM execution_dataset WHERE {' AND '.join(where)}"
    cur.execute(q)
    all_closed = cur.fetchall()
    print(f"Closed execution_dataset rows: {len(all_closed)}")

    orphan_like = []
    now = datetime.now()
    cutoff = now - timedelta(days=days)

    for r in all_closed:
        oid = str(r["order_id"])
        er = r["exit_reason"]
        status_val = r["status_col"] if status_col else None
        dt = _try_parse_dt(r["dataset_updated_at"]) or _try_parse_dt(r["dataset_created_at"])
        if dt is None:
            continue
        if dt < cutoff:
            continue

        er_l = (str(er).lower() if er is not None else "")
        # orphan keyword heuristic
        if (status_val is not None and str(status_val).lower() == "orphan"):
            orphan_like.append(oid)
        elif "orphan" in er_l:
            orphan_like.append(oid)

    orphan_like_set = sorted(set([x for x in orphan_like if x]))
    print(f"Orphan-like (execution_dataset.status or exit_reason has orphan) in last {days} days: {len(orphan_like_set)}")
    if orphan_like_set:
        print("Sample order_ids:")
        print("  " + "\n  ".join(orphan_like_set[:limit_list]))

    # Step B: trades table scan for reconciled/orphan words
    if order_col and exit_reason_col and ts_col:
        print("\n---- trades scan (reason keywords) ----")
        # fetch closed trades within cutoff
        cur.execute(
            f"SELECT {order_col} AS order_id, {exit_reason_col} AS reason, {ts_col} AS ts, pnl, status FROM {trades_table} WHERE status='closed'"
        )
        closed_trades = cur.fetchall()
        keyword_orphan = []
        for r in closed_trades:
            oid = str(r["order_id"])
            reason = r["reason"]
            dt = _try_parse_dt(r["ts"])
            if dt is None:
                continue
            if dt < cutoff:
                continue
            s = str(reason).lower() if reason is not None else ""
            if ("orphan" in s) or ("reconciled" in s) or ("sl/tp" in s) or ("sl/" in s):
                keyword_orphan.append(oid)
        keyword_orphan_set = sorted(set([x for x in keyword_orphan if x]))
        print(f"Keyword orphan/reconciled trades in last {days} days: {len(keyword_orphan_set)}")
        if keyword_orphan_set:
            print("Sample order_ids:")
            print("  " + "\n  ".join(keyword_orphan_set[:limit_list]))

    conn.close()


if __name__ == "__main__":
    # default 30 days
    main(days=30, limit_list=200)

