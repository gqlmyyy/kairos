from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FlagResult:
    """Typed output for a single red-flag checker."""

    name: str
    triggered: bool
    score: float
    severity: int
    reason: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class RedFlagReport:
    """Unified output for RedFlagDetector."""

    flags: Dict[str, FlagResult]
    triggered_flags: Dict[str, FlagResult]
    flag_count: int
    total_score: float
    severity: int
    should_consult_exit_model: bool
    meta: Dict[str, Any]

