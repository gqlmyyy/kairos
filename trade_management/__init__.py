"""Trade management.

Six layers, each in its own module with an explicit input/output contract, plus
an orchestrator that runs them in a fixed order and owns every side effect.

    Layer 1  baseline protection : initial SL/TP, risk gate, intrabar,
                                   break-even, minimum modify distance
    Layer 2  unified exit        : signal-flip hard override + one Exit Score
    Layer 3  adaptive trailing   : trend x volatility in one equation; also the
                                   target-extension mechanism
    Layer 4  trade age           : phase-based management, time stop inside
    Layer 5  partial take profit : R-ladder; MAE/MFE calibrates layer 3 only
    Layer 6  trade profile       : config selector, not decision logic

Every tunable lives in ``tm_config``.
"""

from .orchestrator import ManagementOutcome, TradeManagementOrchestrator
from .types import LayerResult, ModifyRequest, TradeContext

__all__ = [
    "TradeManagementOrchestrator",
    "ManagementOutcome",
    "TradeContext",
    "LayerResult",
    "ModifyRequest",
]
