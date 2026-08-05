"""Layer 1 - Initial protection.

Computes the SL/TP that a trade is opened with, from ATR and the prevailing
market regime. This runs once, at entry, before the order is sent.

Input : symbol, atr, regime, equity, optional profile overrides
Output : InitialProtection(sl_distance, tp_distance, ...)

Distances rather than absolute prices: the caller applies them to the live
execution price, which keeps the broker's fill and our records consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

from . import tm_config as C

logger = get_logger("tm.initial_protection")


@dataclass(frozen=True)
class InitialProtection:
    sl_distance: float
    tp_distance: float
    sl_multiplier: float
    tp_multiplier: float
    regime: str
    capped: bool
    use_fixed_tp: bool

    def apply_to(self, entry_price: float, direction: str) -> tuple:
        """Turn distances into absolute (sl, tp) for a given fill price."""
        is_buy = str(direction).strip().lower() in {"buy", "long", "0"}
        if is_buy:
            sl = entry_price - self.sl_distance
            tp = entry_price + self.tp_distance
        else:
            sl = entry_price + self.sl_distance
            tp = entry_price - self.tp_distance
        return round(sl, 5), (round(tp, 5) if self.use_fixed_tp else 0.0)


def _regime_factors(regime: str) -> tuple:
    key = str(regime or "").strip().lower().replace(" ", "_")
    return C.REGIME_SLTP_FACTORS.get(key, C.REGIME_SLTP_FACTORS["unknown"])


def _max_sl_distance(symbol: str, atr: float, equity: Optional[float]) -> float:
    """Broker- and account-aware ceiling on stop distance.

    Delegates to risk.symbol_info when available (it knows the broker's point
    size and stop level); otherwise falls back to a pip-count ceiling.
    """
    try:
        from risk.symbol_info import get_max_sl_distance

        return float(
            get_max_sl_distance(
                symbol,
                max_sl_pips=C.MAX_SL_PIPS,
                atr=atr,
                account_equity=equity,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive, broker layer absent
        logger.warning("[TM_L1] symbol_info unavailable for %s (%s); using ATR ceiling", symbol, exc)
        return float(atr) * C.ATR_SL_BASE_MULTIPLIER * 3.0


def compute_initial_protection(
    symbol: str,
    atr: float,
    regime: str = "normal",
    account_equity: Optional[float] = None,
    settings: Optional[dict] = None,
) -> InitialProtection:
    """Compute entry SL/TP distances.

    ``settings`` is the resolved profile settings mapping from Layer 6; when
    omitted the module defaults apply.
    """
    settings = settings or {}
    atr = max(float(atr or 0.0), 0.0)

    sl_base = float(settings.get("ATR_SL_BASE_MULTIPLIER", C.ATR_SL_BASE_MULTIPLIER))
    tp_base = float(settings.get("ATR_TP_BASE_MULTIPLIER", C.ATR_TP_BASE_MULTIPLIER))
    use_fixed_tp = bool(settings.get("USE_FIXED_TP", C.USE_FIXED_TP))

    sl_factor, tp_factor = _regime_factors(regime)
    sl_mult = sl_base * sl_factor
    tp_mult = tp_base * tp_factor

    raw_sl = atr * sl_mult
    ceiling = _max_sl_distance(symbol, atr, account_equity)
    sl_distance = min(raw_sl, ceiling) if ceiling > 0 else raw_sl
    tp_distance = atr * tp_mult

    result = InitialProtection(
        sl_distance=sl_distance,
        tp_distance=tp_distance,
        sl_multiplier=sl_mult,
        tp_multiplier=tp_mult,
        regime=str(regime),
        capped=bool(ceiling > 0 and raw_sl > ceiling),
        use_fixed_tp=use_fixed_tp,
    )

    logger.info(
        "[TM_L1] %s atr=%.5f regime=%s sl_mult=%.2f tp_mult=%.2f "
        "sl_dist=%.5f tp_dist=%.5f capped=%s fixed_tp=%s",
        symbol, atr, regime, sl_mult, tp_mult,
        sl_distance, tp_distance, result.capped, use_fixed_tp,
    )
    return result
