from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class SuggestedAction:
    action_type: str  # Continue|MoveSL|MoveTP|PartialClose|ClosePosition
    value: Optional[float] = None
    reason: str = ""


@dataclass
class RuleResult:
    rule_name: str
    rule_score: float
    reasons: List[str]
    suggested_actions: List[SuggestedAction]
    close_allowed: bool = False  # rules must not close directly

