"""M-03: the last numeric gate before an order reaches the broker.

Why a separate module
---------------------
``mt5_direct.open_trade`` already had a safety check, and it did not work for the
case that matters most:

.. code-block:: python

    if direction == "BUY":
        if sl >= live_price:
            raise ValueError(...)
        if tp <= live_price:
            raise ValueError(...)

Every comparison against NaN evaluates to ``False``. A NaN stop loss therefore
passed all four branches, was formatted into the request, and was sent. The
check was not merely incomplete — it was actively *inverted* for the one input
class that could not be recovered from, because MT5 accepts the order and the
position ends up live with no protective stop.

NaN is not a theoretical input here. It is produced by:

* ATR over a flat or gapped series (``0/0``),
* an indicator warm-up window shorter than its period,
* ``float()`` of a missing broker field,
* any arithmetic that touches one of the above.

Design
------
Pure functions over plain numbers, no MT5 imports, no I/O. That keeps the rules
testable without a broker and makes them reusable by the post-entry modify path,
which builds prices the same way.

Both validators **fail closed**: anything that cannot be proven finite and
correctly ordered is rejected. There is no clamping and no "best effort" repair —
a NaN stop is a bug upstream, and silently substituting a number would hide it
while still putting money at risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("order_validation")


class OrderValidationError(ValueError):
    """Raised when an order's numbers cannot be trusted."""


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str = ""

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise OrderValidationError(self.reason)


def is_finite_number(value: Any) -> bool:
    """True only for a real, finite number.

    ``bool`` is rejected deliberately: ``True`` is numerically 1.0, and a boolean
    arriving where a price or a lot size belongs is a wiring mistake, not a
    price. Strings that happen to parse are accepted — MT5 fields and DB columns
    both hand back numeric strings in practice.
    """
    if isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(numeric) or math.isinf(numeric))


def _first_non_finite(pairs: Iterable[Tuple[str, Any]]) -> Optional[str]:
    for name, value in pairs:
        if value is None:
            return f"{name} is missing"
        if not is_finite_number(value):
            return f"{name} is not a finite number: {value!r}"
    return None


def validate_order_inputs(
    symbol: str,
    direction: str,
    size: Any,
    sl_distance: Any,
    tp_distance: Any,
) -> ValidationResult:
    """Check what the caller supplied, before any broker call is made.

    Distances, not prices: ``open_trade`` takes distances and derives absolute
    levels from the live tick, so this runs first and
    :func:`validate_order_prices` runs after that derivation.
    """
    if not symbol or not str(symbol).strip():
        return ValidationResult(False, "symbol is empty")

    if str(direction).strip().upper() not in {"BUY", "SELL"}:
        return ValidationResult(False, f"direction is not BUY/SELL: {direction!r}")

    problem = _first_non_finite(
        (("size", size), ("sl_distance", sl_distance), ("tp_distance", tp_distance))
    )
    if problem:
        return ValidationResult(False, problem)

    if float(size) <= 0:
        return ValidationResult(False, f"size is not positive: {size}")

    # A zero stop distance means the position would be opened unprotected.
    if float(sl_distance) <= 0:
        return ValidationResult(False, f"sl_distance is not positive: {sl_distance}")

    # tp_distance == 0 is legitimate: the "trend" and "breakout" profiles run
    # with USE_FIXED_TP false and let the trailing stop be the exit.
    if float(tp_distance) < 0:
        return ValidationResult(False, f"tp_distance is negative: {tp_distance}")

    return ValidationResult(True)


def validate_order_prices(
    live_price: Any,
    sl: Any,
    tp: Any,
    direction: str,
    *,
    allow_zero_tp: bool = True,
) -> ValidationResult:
    """Check the absolute levels derived from the live tick.

    ``tp == 0.0`` is MT5's "no take profit" sentinel and is accepted by default;
    it is how a trailing-only profile is expressed on the wire.
    """
    direction_upper = str(direction).strip().upper()
    if direction_upper not in {"BUY", "SELL"}:
        return ValidationResult(False, f"direction is not BUY/SELL: {direction!r}")

    problem = _first_non_finite((("live_price", live_price), ("sl", sl), ("tp", tp)))
    if problem:
        return ValidationResult(False, problem)

    price = float(live_price)
    sl_price = float(sl)
    tp_price = float(tp)

    if price <= 0:
        return ValidationResult(False, f"live_price is not positive: {price}")
    if sl_price <= 0:
        return ValidationResult(False, f"sl is not positive: {sl_price}")

    if tp_price < 0:
        return ValidationResult(False, f"tp is negative: {tp_price}")
    if tp_price == 0.0 and not allow_zero_tp:
        return ValidationResult(False, "tp is zero but a target is required")
    tp_is_disabled = tp_price == 0.0

    # Ordering. Written as positive assertions on known-finite numbers, so a
    # NaN can no longer slip through by making every comparison False.
    if direction_upper == "BUY":
        if not sl_price < price:
            return ValidationResult(
                False, f"BUY: sl ({sl_price}) must be below live_price ({price})"
            )
        if not tp_is_disabled and not tp_price > price:
            return ValidationResult(
                False, f"BUY: tp ({tp_price}) must be above live_price ({price})"
            )
    else:
        if not sl_price > price:
            return ValidationResult(
                False, f"SELL: sl ({sl_price}) must be above live_price ({price})"
            )
        if not tp_is_disabled and not tp_price < price:
            return ValidationResult(
                False, f"SELL: tp ({tp_price}) must be below live_price ({price})"
            )

    return ValidationResult(True)


def validate_market_data(
    *,
    atr: Any = None,
    spread: Any = None,
    equity: Any = None,
    risk_amount: Any = None,
    probability: Any = None,
    confidence: Any = None,
) -> ValidationResult:
    """Check the upstream quantities an order is derived from.

    Every argument is optional so a caller can validate only what it holds.
    ``None`` means "not supplied" and is skipped; a supplied value must be
    finite and within its natural range.

    This runs *before* sizing, so a NaN ATR is caught at the point it enters the
    system rather than after it has been multiplied into a stop distance.
    """
    supplied = [
        (name, value)
        for name, value in (
            ("atr", atr), ("spread", spread), ("equity", equity),
            ("risk_amount", risk_amount), ("probability", probability),
            ("confidence", confidence),
        )
        if value is not None
    ]

    problem = _first_non_finite(supplied)
    if problem:
        return ValidationResult(False, problem)

    values = {name: float(value) for name, value in supplied}

    if "atr" in values and values["atr"] <= 0:
        # A zero ATR gives a zero stop distance, i.e. an unprotected position.
        return ValidationResult(False, f"atr is not positive: {values['atr']}")
    if "spread" in values and values["spread"] < 0:
        return ValidationResult(False, f"spread is negative: {values['spread']}")
    if "equity" in values and values["equity"] <= 0:
        return ValidationResult(False, f"equity is not positive: {values['equity']}")
    if "risk_amount" in values and values["risk_amount"] < 0:
        return ValidationResult(False, f"risk_amount is negative: {values['risk_amount']}")

    for name in ("probability", "confidence"):
        if name in values and not 0.0 <= values[name] <= 1.0:
            return ValidationResult(False, f"{name} is outside 0..1: {values[name]}")

    return ValidationResult(True)
