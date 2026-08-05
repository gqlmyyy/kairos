"""Layer 6 - Trade Profile Management.

The governing frame, and deliberately *not* decision logic: this layer is a
config selector. It reads why a trade was entered and returns the settings
mapping that layers 2-5 will use for that trade's whole life.

Profiles:
    trend          -> wide trailing, no fixed TP (the trail is the exit)
    breakout       -> medium trailing
    mean_reversion -> tight trailing, fixed TP, quick exit
    range          -> short targets

Resolution happens once, at entry. The resolved mapping is stored with the
trade so a restart, or a change to the defaults, cannot silently re-profile a
position that is already running.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from utils.logger import get_logger

from . import tm_config as C

logger = get_logger("tm.trade_profile")

LAYER = "trade_profile"


def classify_entry(
    regime: Optional[str] = None,
    mtf_aligned: Optional[bool] = None,
    trend_strength: Optional[float] = None,
) -> str:
    """Infer the profile from the entry context.

    Intentionally conservative: anything unrecognised falls back to the
    configured default rather than guessing.
    """
    key = str(regime or "").strip().lower().replace(" ", "_")

    if key in {"trending", "trend", "strong_uptrend", "strong_downtrend"}:
        # A strong aligned trend earns the open-ended profile; a weak one does not.
        if mtf_aligned is False:
            return "breakout"
        return "trend"

    if key in {"breakout", "expansion", "high_volatility", "volatile"}:
        return "breakout"

    if key in {"mean_reversion", "reversion"}:
        return "mean_reversion"

    if key in {"ranging", "range", "sideways", "consolidation"}:
        return "range"

    if key in {"weak_trend"}:
        try:
            if trend_strength is not None and float(trend_strength) >= C.TRAILING_TREND_HIGH:
                return "trend"
        except (TypeError, ValueError):
            pass
        return "breakout"

    return C.DEFAULT_PROFILE


def resolve_settings(profile: Optional[str]) -> Dict[str, Any]:
    """Return the full settings mapping for ``profile``.

    Module defaults first, profile overrides on top. Layers read every tunable
    through this mapping, so a profile can change any of them without the layer
    knowing profiles exist.
    """
    name = str(profile or "").strip().lower()
    if name not in C.PROFILE_OVERRIDES:
        if name:
            logger.warning("[TM_L6] unknown profile %r; falling back to %s", profile, C.DEFAULT_PROFILE)
        name = C.DEFAULT_PROFILE

    # Start from every module-level constant, then apply the profile's overrides.
    settings: Dict[str, Any] = {
        key: getattr(C, key)
        for key in dir(C)
        if key.isupper() and not key.startswith("_")
    }
    settings.update(C.PROFILE_OVERRIDES.get(name, {}))
    settings["PROFILE"] = name
    return settings


def profile_for_trade(
    stored_profile: Optional[str] = None,
    regime: Optional[str] = None,
    mtf_aligned: Optional[bool] = None,
    trend_strength: Optional[float] = None,
) -> tuple:
    """Return ``(profile_name, settings)`` for a trade.

    A stored profile always wins: once a trade is open its profile is fixed.
    """
    if stored_profile and str(stored_profile).strip().lower() in C.PROFILE_OVERRIDES:
        name = str(stored_profile).strip().lower()
    else:
        name = classify_entry(regime, mtf_aligned, trend_strength)
        logger.info(
            "[TM_L6] classified entry regime=%s aligned=%s strength=%s -> profile=%s",
            regime, mtf_aligned, trend_strength, name,
        )

    return name, resolve_settings(name)
