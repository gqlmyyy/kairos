from __future__ import annotations

import types

from execution.post_entry.post_entry_manager import PostEntryManager, EMERGENCY_LOSS_PIPS


class _FakeBus:
    def __init__(self):
        self.published = []

    def publish(self, evt):
        self.published.append(evt)


class _FakeExecutor:
    def __init__(self):
        self.closed = []

    def close_position(self, order_id):
        self.closed.append(str(order_id))
        return True


class _FakeTradeMonitor:
    def __init__(self, positions):
        self._positions = positions

    def get_open_positions(self):
        return self._positions


def test_emergency_stop_closes_when_loss_exceeds_threshold():
    # Use a loss far worse than -30 points:
    # For buy: loss_points = (current - entry)/point, so set current - entry = 100 * point => -100 points? actually (current-entry)/point
    # We'll set point=1 and use buy direction with current far below entry.
    # Because loss_points for buy uses (current - entry)/point. If current < entry => negative => <= -30
    positions = [
        {
            "order_id": "123",
            "symbol": "EURUSD",
            "direction": "buy",
            "entry_price": 1.2000,
            "price_current": 1.1900,
            "profit": -500,
            "type": 0,
        }
    ]

    mgr = PostEntryManager(loop_interval_sec=60)

    # monkeypatch critical deps
    mgr._trade_monitor = _FakeTradeMonitor(positions)
    mgr._executor = _FakeExecutor()
    mgr._bus = _FakeBus()

    # Ensure no MT5 needed: make mt5 import fail path by setting module attribute on manager's file
    # (the production code does try to import MetaTrader5 globally; in unit tests it may be None)
    # To be safe, we force symbol_info point to None by setting mt5 to None if present.
    # If mt5 exists in runtime, tests still pass because point_size fallback uses raw price diff as "points".
    import execution.post_entry.post_entry_manager as pem

    pem.mt5 = None

    mgr.run_once()

    assert mgr._executor.closed == ["123"], "Emergency stop should close the losing position immediately."
    assert len(mgr._bus.published) >= 1, "Expected at least one TradeClosed event to be published."


if __name__ == "__main__":
    test_emergency_stop_closes_when_loss_exceeds_threshold()
    print("OK")
