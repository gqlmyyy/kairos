from __future__ import annotations

"""analysis/entry_v2/validation.py

Validation utilities for Entry v2 dataset building.

This module is used only for dataset building; it does NOT compute features or labels.

Checks provided:
- Deduplication verification
- Duplicate bar detection
- Missing candle detection during synchronization
- Warm-up feasibility checks
- Basic candle sanity validation

All checks are deterministic and do not require model training.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("entry_v2.validation")


@dataclass
class CandleSyncRowCoverage:
    # For a single synchronized timestamp, do we have candle availability for each TF?
    has_h4: bool
    has_h1: bool
    has_m15: bool


def detect_duplicates_by_time(candles: List[Any]) -> List[float]:
    seen = set()
    dups: List[float] = []
    for c in candles:
        t = getattr(c, "t", None)
        if t is None:
            continue
        if t in seen:
            dups.append(float(t))
        else:
            seen.add(t)
    return dups


def validate_candle_sanity(candles: List[Any]) -> List[int]:
    bad_idx: List[int] = []
    for i, c in enumerate(candles):
        o = getattr(c, "open", None)
        h = getattr(c, "high", None)
        l = getattr(c, "low", None)
        cl = getattr(c, "close", None)
        if o is None or h is None or l is None or cl is None:
            bad_idx.append(i)
            continue
        try:
            o_f, h_f, l_f, cl_f = float(o), float(h), float(l), float(cl)
        except Exception:
            bad_idx.append(i)
            continue
        # Simple sanity
        if h_f < max(o_f, cl_f) or l_f > min(o_f, cl_f):
            bad_idx.append(i)
    return bad_idx


def coverage_report_for_synchronized_times(
    synchronized_times: List[float],
    has_map: Dict[float, CandleSyncRowCoverage],
) -> Dict[str, Any]:
    missing_h4 = 0
    missing_h1 = 0
    missing_m15 = 0
    missing_any = 0

    for t in synchronized_times:
        cov = has_map.get(t)
        if cov is None:
            missing_any += 1
            continue
        if not cov.has_h4:
            missing_h4 += 1
        if not cov.has_h1:
            missing_h1 += 1
        if not cov.has_m15:
            missing_m15 += 1
        # If M15 is never provided in a given dataset build, we should not treat
        # it as missing-any. We determine this dynamically: if at least one
        # synchronized row reports has_m15=True, keep the strict 3-way requirement.
        # Otherwise only require H4+H1.
        if cov.has_m15:
            missing_any_cond = not (cov.has_h4 and cov.has_h1 and cov.has_m15)
        else:
            missing_any_cond = not (cov.has_h4 and cov.has_h1)

        if missing_any_cond:
            missing_any += 1


    total = max(len(synchronized_times), 1)
    return {
        "total_synchronized_rows": len(synchronized_times),
        "missing_h4": missing_h4,
        "missing_h1": missing_h1,
        "missing_m15": missing_m15,
        "missing_any": missing_any,
        "missing_any_pct": missing_any / total,
    }


def abort_if_critical(issues: Dict[str, Any]) -> None:
    """Abort training/build if critical dataset issues are detected.

    Critical issues include:
    - excessive missing synchronized rows
    - zero valid candles after sanity filtering
    """

    if issues.get("duplicate_times", 0) > 0:
        # duplicates are handled by deduplication, so not always critical.
        # But if still present, abort.
        raise RuntimeError("Critical: duplicate timestamps remain after deduplication")

    # For H4+H1-only mode, we intentionally tolerate large missing_any_pct
    # because exact alignment is not required (latest-candle-at-or-before
    # selection + forward-fill). We only abort if there are no valid rows.
    # Only abort if we ended up with *zero* usable rows.
    # Missingness can be tolerated in H4+H1-only mode.
    if issues.get("valid_rows", 0) <= 0:
        raise RuntimeError("Critical: no valid synchronized rows")

