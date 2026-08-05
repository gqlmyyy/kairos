"""Shared value types for the trade-management layers.

Every layer takes a ``TradeContext`` (plus its own settings) and returns a
``LayerResult``. Nothing else crosses layer boundaries, which is what makes each
layer testable in isolation: build a context dict, call the layer, assert on the
result.

Layers are pure. They never touch MT5, the database, Telegram or the network —
the orchestrator owns all side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TradeContext:
    """Everything a layer is allowed to know about an open trade.

    Prices are in symbol price units. ``r_distance`` is the price distance of
    1R, i.e. ``abs(entry_price - initial_sl)``; all R-multiples are derived from
    it so a layer never has to recompute risk.
    """

    order_id: str
    symbol: str
    direction: str                 # "buy" | "sell"
    entry_price: float
    current_price: float
    volume: float                  # currently open volume
    initial_volume: float          # volume at entry, for partial ladder maths
    sl: float                      # current stop, 0.0 when unset
    tp: Optional[float] = None
    initial_sl: float = 0.0
    r_distance: float = 0.0

    # Market context
    atr_now: float = 0.0
    atr_at_entry: float = 0.0
    trend_strength: float = 50.0   # 0..100
    regime: str = "unknown"
    point_size: float = 0.00001
    broker_stop_level_points: float = 0.0

    # Lifecycle
    bars_open: int = 0
    profile: str = "trend"

    # Excursions, in R
    mfe_r: float = 0.0
    mae_r: float = 0.0

    # State flags carried across loop passes
    breakeven_done: bool = False
    partial_levels_done: tuple = ()

    # Free-form extras (exit-model inputs, signal snapshot, ...)
    extras: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- helpers
    @property
    def is_buy(self) -> bool:
        return str(self.direction).strip().lower() in {"buy", "long", "0"}

    @property
    def signed_move(self) -> float:
        """Price movement in the trade's favour (negative when losing)."""
        if self.is_buy:
            return self.current_price - self.entry_price
        return self.entry_price - self.current_price

    @property
    def profit_r(self) -> float:
        """Open profit expressed in R. Zero when risk is unknown."""
        if self.r_distance <= 0:
            return 0.0
        return self.signed_move / self.r_distance

    def sl_is_improvement(self, candidate: float) -> bool:
        """True when ``candidate`` moves the stop towards profit, never back."""
        if candidate is None or candidate <= 0:
            return False
        if self.sl in (None, 0, 0.0):
            return True
        return candidate > self.sl if self.is_buy else candidate < self.sl


@dataclass
class LayerResult:
    """What a layer wants to happen. The orchestrator decides what actually does.

    ``close_fraction`` is a fraction of the *original* volume, so ladder levels
    stay stable as the position shrinks.
    """

    layer: str
    close_full: bool = False
    close_fraction: float = 0.0
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    # Set by hard overrides: stop evaluating any further layer.
    terminal: bool = False
    # Diagnostics for logging/telemetry; never drives behaviour.
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def wants_action(self) -> bool:
        return bool(
            self.close_full
            or self.close_fraction > 0
            or self.new_sl is not None
            or self.new_tp is not None
        )

    @classmethod
    def noop(cls, layer: str, reason: str = "") -> "LayerResult":
        return cls(layer=layer, reasons=[reason] if reason else [])


@dataclass(frozen=True)
class ModifyRequest:
    """A pending SL/TP write, before the minimum-distance filter runs."""

    order_id: str
    symbol: str
    direction: str
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    reasons: tuple = ()

    def with_sl(self, sl: Optional[float]) -> "ModifyRequest":
        return replace(self, new_sl=sl)
