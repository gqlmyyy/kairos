from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import DB_FILE


TARGETS = {
    # will try multiple possible column names per target
    "adx": [
        "expected_adx",
        "actual_adx",
        "adx",
        "entry_adx",
    ],
    "session": [
        "expected_session",
        "actual_session",
        "session",
    ],
    "trend_h1": [
        "expected_trend_h1",
        "actual_trend_h1",
        "trend_h1",
    ],
    "trend_h4": [
        "expected_trend_h4",
        "actual_trend_h4",
        "trend_h4",
    ],
}


TABLE_CANDIDATES = [
    "execution_dataset",
    "execution_dataset_v2",
]


NON_MISSING_NULL_LIKE = {"", "none", "null", "nan", "missing", "mISSING"}


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return True
        if s.lower() in NON_MISSING_NULL_LIKE:
            return True
        return False
    # numeric: treat 0 as present (we only care about non-null)
    return False


def _get_table_info(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    cols = [r[1] for r in rows]  # second field is name
    return cols


def _pick_column(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _find_timestamp_column(cols: List[str]) -> Optional[str]:
    # for recent-vs-old split
    for c in ["created_at", "dataset_created_at", "closed_at", "dataset_updated_at"]:
        if c in cols:
            return c
    return None


def _parse_timestamp_safe(v: Any) -> Optional[float]:
    """Return epoch seconds if parseable, else None.

    Accepts: int/float epoch, ISO strings, common sqlite datetime strings.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # numeric string
        try:
            if s.isdigit():
                return float(int(s))
        except Exception:
            pass
        # try iso formats
        from datetime import datetime

        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%f",
        ):
            try:
                ss = s.replace("Z", "").replace("+00:00", "")
                dt = datetime.strptime(ss[:19] if "." not in ss else ss, fmt)
                return dt.timestamp()
            except Exception:
                continue
    return None


def compute_coverage_for_table(table: str, rows_limit: Optional[int] = None) -> None:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cols = _get_table_info(conn, table)

    ts_col = _find_timestamp_column(cols)

    picked: Dict[str, Optional[str]] = {}
    for t, candidates in TARGETS.items():
        picked[t] = _pick_column(cols, candidates)

    total = 0
    non_missing: Dict[str, int] = {k: 0 for k in TARGETS.keys()}

    # recent/old split
    # we compute a threshold by median timestamp if possible
    ts_vals: List[float] = []
    row_cache: List[sqlite3.Row] = []

    cur = conn.cursor()
    sql = f"SELECT * FROM {table}"
    if rows_limit is not None:
        sql += f" LIMIT {int(rows_limit)}"
    cur.execute(sql)

    for r in cur.fetchall():
        row_cache.append(r)
        total += 1
        if ts_col:
            tv = _parse_timestamp_safe(r[ts_col])
            if tv is not None:
                ts_vals.append(tv)

        for t, col in picked.items():
            if col is None:
                continue
            v = r[col]
            if not _is_missing(v):
                non_missing[t] += 1

    conn.close()

    # print schema picks
    print(f"DB_FILE={DB_FILE}")
    print(f"TABLE={table}")
    print(f"timestamp_col={ts_col}")
    print("\nPicked columns (target -> actual col):")
    for t, col in picked.items():
        print(f"  - {t}: {col}")

    print("\nCoverage (all rows):")
    print("{:<12} | {:>10} | {:>12} | {:>8}".format("target", "total", "non_missing", "pct"))
    print("-" * 50)
    for t in TARGETS.keys():
        col = picked[t]
        if col is None:
            print(f"{t:<12} | {total:>10} | {0:>12} | {0.0:>7.1f}%")
            continue
        nm = non_missing[t]
        pct = (nm * 100.0 / total) if total else 0.0
        print("{:<12} | {:>10} | {:>12} | {:>7.1f}%".format(t, total, nm, pct))

    # recent vs old
    if ts_col and ts_vals:
        ts_vals_sorted = sorted(ts_vals)
        mid = len(ts_vals_sorted) // 2
        threshold = ts_vals_sorted[mid]

        # recompute with threshold
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cols2 = _get_table_info(conn, table)
        cur2 = conn.cursor()
        cur2.execute(f"SELECT * FROM {table}")

        total_recent = 0
        total_old = 0
        non_recent: Dict[str, int] = {k: 0 for k in TARGETS.keys()}
        non_old: Dict[str, int] = {k: 0 for k in TARGETS.keys()}

        for r in cur2.fetchall():
            tv = _parse_timestamp_safe(r[ts_col])
            if tv is None:
                continue
            bucket = "recent" if tv >= threshold else "old"
            if bucket == "recent":
                total_recent += 1
                for t, col in picked.items():
                    if col is None:
                        continue
                    if not _is_missing(r[col]):
                        non_recent[t] += 1
            else:
                total_old += 1
                for t, col in picked.items():
                    if col is None:
                        continue
                    if not _is_missing(r[col]):
                        non_old[t] += 1

        conn.close()

        print("\nCoverage (recent vs old by median timestamp):")
        print(f"threshold_epoch={threshold}")
        print("{:<12} | {:>10} | {:>10} | {:>10}".format("target", "old_pct", "new_pct", "note"))
        for t in TARGETS.keys():
            col = picked[t]
            if col is None:
                print(f"{t:<12} | {0.0:>10.1f} | {0.0:>10.1f} | missing_col")
                continue
            old_pct = (non_old[t] * 100.0 / total_old) if total_old else 0.0
            new_pct = (non_recent[t] * 100.0 / total_recent) if total_recent else 0.0
            print(f"{t:<12} | {old_pct:>10.1f} | {new_pct:>10.1f} | ts_col={ts_col}")


def main() -> None:
    # detect which table exists
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    conn.close()

    table = None
    for c in TABLE_CANDIDATES:
        if c in tables:
            table = c
            break

    if table is None:
        raise RuntimeError(f"None of tables exist: {TABLE_CANDIDATES} in DB_FILE={DB_FILE}")

    compute_coverage_for_table(table)


if __name__ == "__main__":
    main()

