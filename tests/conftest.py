"""Shared fixtures for trade-management layer tests."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trade_management.layer6_trade_profile import resolve_settings  # noqa: E402
from trade_management.types import TradeContext  # noqa: E402


@pytest.fixture
def settings():
    """Module defaults, with no profile overrides applied."""
    base = resolve_settings("trend")
    # Neutralise the trend profile's overrides so tests exercise the defaults.
    base.update(
        {
            "TRAILING_BASE_ATR_MULTIPLIER": 2.0,
            "TRAILING_MAX_ATR_MULTIPLIER": 4.0,
            "EXIT_SCORE_THRESHOLD": 0.75,
            "TIME_STOP_MAX_BARS": 15,
            "USE_FIXED_TP": True,
        }
    )
    return base


def make_ctx(**overrides) -> TradeContext:
    """A long trade, 1.0 ATR of risk, currently flat.

    entry 100.0, initial SL 99.0 -> r_distance = 1.0, so profit_r reads
    directly off the price move.
    """
    defaults = dict(
        order_id="1001",
        symbol="EURUSD",
        direction="buy",
        entry_price=100.0,
        current_price=100.0,
        volume=1.0,
        initial_volume=1.0,
        sl=99.0,
        tp=103.0,
        initial_sl=99.0,
        r_distance=1.0,
        atr_now=1.0,
        atr_at_entry=1.0,
        trend_strength=50.0,
        regime="normal",
        point_size=0.01,
        broker_stop_level_points=0.0,
        bars_open=1,
        profile="trend",
    )
    defaults.update(overrides)
    return TradeContext(**defaults)


@pytest.fixture
def ctx():
    return make_ctx()
