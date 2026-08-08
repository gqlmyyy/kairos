"""H-03 regression: management state must survive a restart.

``partial_levels_done`` lived only in ``TradeRuntimeState``. A bot restarted
while holding a position that had already taken its +2R partial came back with
an empty ladder, saw profit_r >= 2.0, and closed another 30% of the original
volume. ``breakeven_done`` had the mirror-image gap: the column was read on
startup but never written.

These tests use a temporary database, so nothing here touches the live file.
"""

from __future__ import annotations

import os
import sqlite3
import sys

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """A throwaway execution_dataset with one open trade."""
    path = str(tmp_path / "test.db")

    import config
    import data.storage.database as db

    monkeypatch.setattr(config, "DB_FILE", path)
    monkeypatch.setattr(db, "DB_FILE", path)

    db.init_db()

    db.upsert_execution_expected(
        order_id="TEST-1", symbol="XAUUSD", direction="BUY",
        expected_entry=4000.0, expected_final_score=70.0, expected_ai_score=70.0,
        expected_ai_confidence=0.8, expected_trend_score=70.0,
        expected_momentum_score=60.0, expected_sentiment_score=55.0,
        expected_volatility_score=50.0, expected_sl=3990.0, expected_tp=4030.0,
        expected_volume=0.10, entry_profile="trend",
    )
    return db


class TestPartialLadderPersistence:
    def test_levels_round_trip(self, temp_db):
        assert temp_db.update_partial_levels_done("TEST-1", {0}) is True
        row = temp_db.get_execution_dataset("TEST-1")
        assert temp_db.parse_partial_levels_done(row["partial_levels_done"]) == {0}

    def test_multiple_levels_round_trip(self, temp_db):
        temp_db.update_partial_levels_done("TEST-1", {0, 1})
        row = temp_db.get_execution_dataset("TEST-1")
        assert temp_db.parse_partial_levels_done(row["partial_levels_done"]) == {0, 1}

    def test_restart_does_not_retake_a_completed_level(self, temp_db):
        """The exact scenario from the finding."""
        from trade_management.layer5_partial_tp import next_due_level
        from trade_management.layer6_trade_profile import resolve_settings
        from trade_management.types import TradeContext

        settings = resolve_settings("trend")

        # +2R reached, level 0 taken and persisted.
        temp_db.update_partial_levels_done("TEST-1", {0})

        # Restart: rebuild state from the database only.
        row = temp_db.get_execution_dataset("TEST-1")
        restored = temp_db.parse_partial_levels_done(row["partial_levels_done"])

        ctx = TradeContext(
            order_id="TEST-1", symbol="XAUUSD", direction="buy",
            entry_price=4000.0, current_price=4020.0,
            volume=0.07, initial_volume=0.10,
            sl=3990.0, initial_sl=3990.0, r_distance=10.0,
            partial_levels_done=tuple(sorted(restored)),
        )
        assert ctx.profit_r == pytest.approx(2.0)

        due = next_due_level(ctx, settings)
        assert due is None, "level 0 was retaken after restart"

    def test_without_persistence_the_level_would_be_retaken(self, temp_db):
        """Proves the test above is actually exercising the fix."""
        from trade_management.layer5_partial_tp import next_due_level
        from trade_management.layer6_trade_profile import resolve_settings
        from trade_management.types import TradeContext

        ctx = TradeContext(
            order_id="TEST-1", symbol="XAUUSD", direction="buy",
            entry_price=4000.0, current_price=4020.0,
            volume=0.07, initial_volume=0.10,
            sl=3990.0, initial_sl=3990.0, r_distance=10.0,
            partial_levels_done=(),  # state lost
        )
        due = next_due_level(ctx, resolve_settings("trend"))
        assert due is not None and due.index == 0

    def test_level_1_still_available_after_level_0(self, temp_db):
        from trade_management.layer5_partial_tp import next_due_level
        from trade_management.layer6_trade_profile import resolve_settings
        from trade_management.types import TradeContext

        temp_db.update_partial_levels_done("TEST-1", {0})
        restored = temp_db.parse_partial_levels_done(
            temp_db.get_execution_dataset("TEST-1")["partial_levels_done"]
        )
        ctx = TradeContext(
            order_id="TEST-1", symbol="XAUUSD", direction="buy",
            entry_price=4000.0, current_price=4030.0,   # +3R
            volume=0.07, initial_volume=0.10,
            sl=3990.0, initial_sl=3990.0, r_distance=10.0,
            partial_levels_done=tuple(sorted(restored)),
        )
        due = next_due_level(ctx, resolve_settings("trend"))
        assert due is not None and due.index == 1

    # --- edge cases ---
    def test_unknown_order_id_reports_failure(self, temp_db):
        assert temp_db.update_partial_levels_done("NOPE", {0}) is False

    def test_empty_and_null_decode_to_empty_set(self, temp_db):
        assert temp_db.parse_partial_levels_done(None) == set()
        assert temp_db.parse_partial_levels_done("") == set()

    def test_corrupt_value_decodes_conservatively(self, temp_db):
        """Unparsable -> empty, which re-arms rather than skipping protection."""
        assert temp_db.parse_partial_levels_done("x,y,z") == set()
        assert temp_db.parse_partial_levels_done("0,junk,1") == {0, 1}

    def test_db_write_failure_is_reported_not_swallowed(self, temp_db, monkeypatch):
        """Broker already executed; the caller must learn the write failed."""
        def boom():
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(temp_db, "get_conn", boom)
        assert temp_db.update_partial_levels_done("TEST-1", {0}) is False


class TestBreakevenPersistence:
    def test_breakeven_round_trips(self, temp_db):
        assert temp_db.update_breakeven_done("TEST-1", True) is True
        row = temp_db.get_execution_dataset("TEST-1")
        assert bool(row["breakeven_done"]) is True

    def test_breakeven_defaults_to_false(self, temp_db):
        row = temp_db.get_execution_dataset("TEST-1")
        assert bool(row["breakeven_done"]) is False

    def test_restored_breakeven_stops_the_layer_re_arming(self, temp_db):
        from trade_management import layer1_breakeven
        from trade_management.layer6_trade_profile import resolve_settings
        from trade_management.types import TradeContext

        temp_db.update_breakeven_done("TEST-1", True)
        row = temp_db.get_execution_dataset("TEST-1")

        ctx = TradeContext(
            order_id="TEST-1", symbol="XAUUSD", direction="buy",
            entry_price=4000.0, current_price=4015.0,
            volume=0.10, initial_volume=0.10,
            sl=3990.0, initial_sl=3990.0, r_distance=10.0,
            breakeven_done=bool(row["breakeven_done"]),
        )
        assert layer1_breakeven.apply_breakeven(ctx, resolve_settings("trend")).new_sl is None


class TestEntryStatePersistence:
    """Everything the manager needs to rebuild a trade after a restart."""

    @pytest.mark.parametrize(
        "column,expected",
        [
            ("expected_entry", 4000.0),
            ("expected_sl", 3990.0),
            ("expected_tp", 4030.0),
            ("expected_volume", 0.10),
            ("entry_profile", "trend"),
            ("symbol", "XAUUSD"),
            ("direction", "BUY"),
        ],
    )
    def test_field_survives(self, temp_db, column, expected):
        row = temp_db.get_execution_dataset("TEST-1")
        assert row[column] == expected

    def test_r_distance_is_reconstructible(self, temp_db):
        """1R must be derivable from persisted values alone."""
        row = temp_db.get_execution_dataset("TEST-1")
        r = abs(float(row["expected_entry"]) - float(row["expected_sl"]))
        assert r == pytest.approx(10.0)
