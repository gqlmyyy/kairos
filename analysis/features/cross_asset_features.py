"""Cross-asset context (DXY, US yields, silver, oil, correlated pairs).

Research-only, daily resolution, and explicitly best-effort: this module
reports what it actually managed to fetch rather than assuming success.
`analysis.market.historical_fetcher.AlternativeDataFetchers` already wraps
yfinance/alpha_vantage/polygon.io calls but is not imported by anything — it
was written and never wired in. This module is the wiring, kept separate from
that one so a failed or unavailable fetch here cannot affect anything else.

Point-in-time rule
-------------------
A daily bar is indexed by its trading date, but "close" happens at a specific
time within that date that varies by instrument and exchange (COMEX silver
settles differently from ICE's DXY future, which trades nearly continuously).
Rather than encode each instrument's exact session close, this module uses one
deliberately conservative rule for all of them:

    a daily bar dated D becomes available at (D + 1 calendar day), 00:00 UTC

That is pessimistic for many instruments — DXY's close is typically known
same evening — but it can never be optimistic, which is the property that
matters. An H4 decision late on day D+1 sees D's close; an H4 decision on the
morning of D+1 does not, even though the real close may already be public.
Losing a few hours of legitimate freshness is an acceptable cost for a rule
that cannot leak.

Tickers are Yahoo Finance symbols, chosen for what is actually queryable there
without a paid feed:

    DX-Y.NYB   Dollar Index
    ^TNX       US 10-year yield (x10, e.g. 42.5 = 4.25%)
    ^IRX       US 13-week T-bill (2-year proxy is not directly free on Yahoo)
    SI=F       Silver futures
    CL=F       WTI crude futures
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

TICKERS: Dict[str, str] = {
    "dxy": "DX-Y.NYB",
    "us10y": "^TNX",
    "us13w": "^IRX",
    "silver": "SI=F",
    "oil": "CL=F",
}

# Only these get built for a given symbol — a EURUSD row gains nothing from
# an oil feature, and adding it would just be another chance to fit noise.
RELEVANT_TICKERS: Dict[str, tuple] = {
    "EURUSD": ("dxy", "us10y", "us13w"),
    "GBPUSD": ("dxy", "us10y", "us13w"),
    "XAUUSD": ("dxy", "us10y", "us13w", "silver", "oil"),
}

DAY_SECONDS = 86400


class FetchReport:
    """What actually happened, per ticker. Never fabricated."""

    def __init__(self) -> None:
        self.status: Dict[str, str] = {}
        self.rows: Dict[str, int] = {}
        self.span: Dict[str, tuple] = {}

    def record(self, key: str, status: str, series: Optional[List[Dict[str, float]]] = None
              ) -> None:
        self.status[key] = status
        if series:
            self.rows[key] = len(series)
            self.span[key] = (series[0]["t"], series[-1]["t"])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": dict(self.status),
            "rows": dict(self.rows),
            "span": {k: (datetime.fromtimestamp(a, tz=timezone.utc).date().isoformat(),
                        datetime.fromtimestamp(b, tz=timezone.utc).date().isoformat())
                    for k, (a, b) in self.span.items()},
        }


def fetch_daily_series(ticker: str, start: str, end: str) -> Optional[List[Dict[str, float]]]:
    """One ticker's daily closes for [start, end), or None on any failure.

    `start`/`end` are 'YYYY-MM-DD'. Returns rows sorted ascending with
    `t` = UTC midnight of the trading date and `close` = that day's close.
    Never raises — a failed fetch is reported by the caller via FetchReport,
    not by an exception the research pipeline would need to catch everywhere.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        data = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
    except Exception:  # noqa: BLE001 - network/library failures must not crash research
        return None

    if data is None or data.empty:
        return None

    rows: List[Dict[str, float]] = []
    for index, row in data.iterrows():
        date = index.to_pydatetime().date()
        t = datetime(date.year, date.month, date.day, tzinfo=timezone.utc).timestamp()
        close = row.get("Close")
        if close is None or close != close:  # NaN check without importing math here
            continue
        rows.append({"t": float(t), "close": float(close)})

    rows.sort(key=lambda r: r["t"])
    return rows or None


def fetch_all(start: str, end: str, tickers: Optional[Dict[str, str]] = None
             ) -> tuple:
    """Fetch every configured ticker. Returns (data, report).

    `data[key]` is present only for tickers that actually returned rows —
    absence IS the "NOT AVAILABLE" signal, checked by callers via `in data`
    rather than by a separate availability flag that could drift from reality.
    """
    tickers = tickers or TICKERS
    data: Dict[str, List[Dict[str, float]]] = {}
    report = FetchReport()

    for key, symbol in tickers.items():
        series = fetch_daily_series(symbol, start, end)
        if series is None:
            report.record(key, "NOT AVAILABLE (fetch failed or yfinance missing)")
            continue
        if len(series) < 30:
            report.record(key, "INSUFFICIENT HISTORY", series)
            continue
        report.record(key, "AVAILABLE", series)
        data[key] = series

    return data, report


def _available_at(bar_t: float) -> float:
    """The (D+1) 00:00 UTC rule, applied once so it cannot be re-derived
    differently in two places."""
    day_start = (bar_t // DAY_SECONDS) * DAY_SECONDS
    return day_start + DAY_SECONDS


def _last_known(series: Sequence[Dict[str, float]], at: float) -> Optional[float]:
    """Most recent close whose availability time is at or before `at`."""
    best: Optional[float] = None
    for row in series:
        if _available_at(row["t"]) <= at:
            best = row["close"]
        else:
            break
    return best


def build_cross_asset_features(
    data: Dict[str, List[Dict[str, float]]],
    *,
    symbol: str,
    decision_timestamp: float,
    lookback_days: int = 20,
) -> Optional[Dict[str, float]]:
    """Features knowable at `decision_timestamp` for `symbol`, or None.

    Each series must be sorted ascending by `t`. A None return means no
    relevant ticker had data available yet at this decision point — the
    caller drops the row rather than filling zeros that would look like a
    real reading of "no macro move".
    """
    wanted = RELEVANT_TICKERS.get(symbol, ())
    out: Dict[str, float] = {}
    any_available = False

    for key in wanted:
        series = data.get(key)
        if not series:
            continue

        visible = [r for r in series if _available_at(r["t"]) <= decision_timestamp]
        if len(visible) < 5:
            continue

        current = visible[-1]["close"]
        window = [r["close"] for r in visible[-lookback_days - 1:-1]]
        if not window:
            continue

        any_available = True
        mean = statistics.mean(window)
        stdev = statistics.pstdev(window)
        out[f"{key}_return_1d"] = (current - window[-1]) / window[-1] if window[-1] else 0.0
        out[f"{key}_zscore_{lookback_days}d"] = (
            (current - mean) / stdev if stdev > 0 else 0.0
        )

    return out if any_available else None
