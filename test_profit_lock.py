"""Unit test for Profit Lock (Feature 4).

Verifies that the locked profit value is logical relative to trade size and ATR.
"""

from __future__ import annotations

import pytest

from execution.post_entry.rules.protect_open_profit_rule import ProtectOpenProfitRule
from execution.post_entry.rules.rule_types import RuleResult, SuggestedAction


def _make_snapshot(
    direction: str = "buy",
    entry_price: float = 100.0,
    sl: float = 99.0,
    profit_points: float = 60.0,
    price_current: float = 101.0,
) -> dict:
    return {
        "trade": {
            "direction": direction,
            "entry_price": entry_price,
            "sl": sl,
            "profit_points": profit_points,
            "price_current": price_current,
            "symbol": "XAUUSD",
            "order_id": "12345",
        }
    }


def test_profit_lock_activates_after_threshold():
    """Profit lock should trigger when profit_points >= PROFIT_PROTECT_TRIGGER_POINTS."""
    rule = ProtectOpenProfitRule()
    snap = _make_snapshot(profit_points=60.0)  # >= 50 trigger
    result = rule(snap, "OPENED")

    assert result.rule_score > 0
    assert any(sa.action_type == "MoveSL" for sa in result.suggested_actions)


def test_profit_lock_does_not_activate_below_threshold():
    """Profit lock should NOT trigger when profit_points < threshold."""
    rule = ProtectOpenProfitRule()
    snap = _make_snapshot(profit_points=10.0)  # < 50 trigger
    result = rule(snap, "OPENED")

    assert result.rule_score == 0.0
    assert not any(sa.action_type == "MoveSL" for sa in result.suggested_actions)


def test_profit_lock_sl_is_logical_for_buy():
    """For a BUY, locked SL must be above entry (protecting profit) and below current price."""
    rule = ProtectOpenProfitRule()
    entry = 100.0
    snap = _make_snapshot(direction="buy", entry_price=entry, profit_points=60.0, price_current=102.0)
    result = rule(snap, "OPENED")

    move_sl = [sa for sa in result.suggested_actions if sa.action_type == "MoveSL"]
    assert len(move_sl) == 1
    new_sl = float(move_sl[0].value)

    # Locked SL must be above entry (protecting profit) but below current price
    assert new_sl > entry
    assert new_sl < float(snap["trade"]["price_current"])


def test_profit_lock_sl_is_logical_for_sell():
    """For a SELL, locked SL must be below entry (protecting profit) and above current price."""
    rule = ProtectOpenProfitRule()
    entry = 100.0
    snap = _make_snapshot(direction="sell", entry_price=entry, sl=101.0, profit_points=60.0, price_current=98.0)
    result = rule(snap, "OPENED")

    move_sl = [sa for sa in result.suggested_actions if sa.action_type == "MoveSL"]
    assert len(move_sl) == 1
    new_sl = float(move_sl[0].value)

    # Locked SL must be below entry (protecting profit) but above current price
    assert new_sl < entry
    assert new_sl > float(snap["trade"]["price_current"])


def test_profit_lock_units_are_price_units_not_pips():
    """Verify locked profit is in price units (not pips) - critical for XAUUSD.

    For XAUUSD, pip = 0.1, so 10 points = 1.0 price unit.
    The lock offset (PROFIT_PROTECT_LOCK_POINTS=10) should be 10 * pip = 1.0 price unit.
    """
    from config import PROFIT_PROTECT_LOCK_POINTS, PIP_VALUES

    rule = ProtectOpenProfitRule()
    entry = 100.0
    snap = _make_snapshot(direction="buy", entry_price=entry, profit_points=60.0)
    result = rule(snap, "OPENED")

    move_sl = [sa for sa in result.suggested_actions if sa.action_type == "MoveSL"]
    assert len(move_sl) == 1
    new_sl = float(move_sl[0].value)

    # Locked SL = entry + lock_points * pip_value
    pip = PIP_VALUES.get("XAUUSD", 0.1)
    expected_sl = entry + float(PROFIT_PROTECT_LOCK_POINTS) * pip
    assert abs(new_sl - expected_sl) < 1e-6


def test_profit_lock_never_moves_sl_backwards():
    """SL must never move backwards (reduce protection)."""
    rule = ProtectOpenProfitRule()

    # BUY with existing SL already above the proposed lock level
    entry = 100.0
    existing_sl = 101.5  # already locked higher than proposed (100 + 10*0.1 = 101.0)
    snap = _make_snapshot(direction="buy", entry_price=entry, sl=existing_sl, profit_points=60.0)
    result = rule(snap, "OPENED")

    # Should NOT suggest moving SL backwards
    assert result.rule_score == 0.0
    assert not any(sa.action_type == "MoveSL" for sa in result.suggested_actions)