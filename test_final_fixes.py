#!/usr/bin/env python3
"""Tests for: migration, dedup halt sources, R-multiple unification."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_1_migration_old_state_file():
    """Test #1: Migration of old halt_reason to halt_sources."""
    print("\n" + "=" * 70)
    print("TEST #1: Migration of old halt_reason to halt_sources")
    print("=" * 70)

    state_file = tempfile.mktemp(suffix="_old_state.json")
    try:
        old_state = {
            "halted": True,
            "halt_reason": "MT5 connection lost - candle boundary unavailable",
            "cumulative_loss_r": 0.0,
            "consecutive_losses": 0,
            "daily_loss_usd": 0.0,
            "date": "2026-08-04",
        }
        with open(state_file, "w") as f:
            json.dump(old_state, f)
        print(f"\n  Step (a): Created old-format state file")
        print(f"  {json.dumps(old_state, indent=2)}")

        from risk.risk_governor import RiskGovernor
        gov = RiskGovernor(state_file=state_file)

        print(f"\n  Step (b)+(c): Loaded RiskGovernor from old state file")
        print(f"  is_halted() = {gov.is_halted()}")
        print(f"  get_halt_sources() = {gov.get_halt_sources()}")

        assert gov.is_halted() == True
        assert "mt5_disconnect" in gov.get_halt_sources()
        assert len(gov.get_halt_sources()) == 1

        print(f"\n  PASS: Old halt_reason migrated to halt_sources=['mt5_disconnect']")
        return True
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


def test_2_dedup_halt_sources():
    """Test #2: Prevent duplicate halt sources."""
    print("\n" + "=" * 70)
    print("TEST #2: Prevent duplicate halt sources")
    print("=" * 70)

    state_file = tempfile.mktemp(suffix="_dedup_test.json")
    try:
        from risk.risk_governor import RiskGovernor
        gov = RiskGovernor(state_file=state_file)

        print(f"\n  Calling halt(source='mt5_disconnect') 50 times...")
        for i in range(50):
            gov.halt(f"MT5 disconnected attempt {i}", source="mt5_disconnect")

        sources = gov.get_halt_sources()
        print(f"  get_halt_sources() = {sources}")
        print(f"  len(get_halt_sources()) = {len(sources)}")

        assert len(sources) == 1
        assert sources[0] == "mt5_disconnect"

        print(f"\n  PASS: halt() deduplicates sources correctly (50 calls -> 1 entry)")
        return True
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


def test_3_r_multiple_unification():
    """Test #3: R-multiple unification."""
    print("\n" + "=" * 70)
    print("TEST #3: R-multiple Unification")
    print("=" * 70)

    from risk.r_multiple import calculate_r_multiple, calculate_risk_amount_usd
    from config import PIP_VALUES
    from risk.position_sizing import get_pip_value_per_lot

    symbol = "EURUSD"
    entry = 1.1000
    sl = 1.0970
    trade_size = 0.50
    pnl = -50.0

    pip = PIP_VALUES.get(symbol, 0.0001)
    pip_value_per_lot = get_pip_value_per_lot(symbol)
    sl_distance = abs(entry - sl)
    expected_risk = calculate_risk_amount_usd(sl_distance, symbol, trade_size)
    expected_r = calculate_r_multiple(pnl, sl_distance, symbol, trade_size)

    print(f"\n  Inputs: symbol={symbol}, entry={entry}, sl={sl}, size={trade_size}, pnl={pnl}")
    print(f"  pip={pip}, pip_value_per_lot=${pip_value_per_lot}")
    print(f"  sl_distance={sl_distance:.5f}")
    print(f"\n  Shared function result:")
    print(f"  calculate_risk_amount_usd() = ${expected_risk:.2f}")
    print(f"  calculate_r_multiple() = {expected_r:.2f} R")

    assert abs(expected_risk - 150.0) < 0.001, f"Expected $150, got ${expected_risk:.6f}"
    assert abs(expected_r - (-0.333)) < 0.01

    # Verify reconciliation.py uses the shared function
    print(f"\n  Checking reconciliation.py...")
    with open("execution/reconciliation.py", "r") as f:
        recon_content = f.read()
    assert "from risk.r_multiple import calculate_risk_amount_usd" in recon_content
    assert "calculate_risk_amount_usd(sl_distance, trade_symbol, trade_size)" in recon_content
    print(f"  CONFIRMED: reconciliation.py uses shared function")

    # Verify post_entry_manager.py computes risk_amount_usd
    print(f"\n  Checking post_entry_manager.py...")
    with open("execution/post_entry/post_entry_manager.py", "r") as f:
        pem_content = f.read()
    assert "from risk.r_multiple import calculate_risk_amount_usd" in pem_content
    assert "_compute_risk_amount_usd" in pem_content
    assert "risk_amount_usd=_risk_amt" in pem_content
    print(f"  CONFIRMED: post_entry_manager.py computes and passes risk_amount_usd")

    print(f"\n  PASS: R-multiple calculation unified across all callers")
    return True


def test_migration_edge_cases():
    """Test migration edge cases."""
    print("\n" + "=" * 70)
    print("TEST #1b: Migration edge cases")
    print("=" * 70)

    state_file = tempfile.mktemp(suffix="_empty.json")
    try:
        # Edge case 1: Empty file
        with open(state_file, "w") as f:
            f.write("")
        from risk.risk_governor import RiskGovernor
        gov = RiskGovernor(state_file=state_file)
        assert not gov.is_halted()
        print(f"\n  Empty file: is_halted()={gov.is_halted()} OK")

        # Edge case 2: File with halt_reason but not halted
        with open(state_file, "w") as f:
            json.dump({"halted": False, "halt_reason": "some reason"}, f)
        gov2 = RiskGovernor(state_file=state_file)
        assert not gov2.is_halted()
        assert len(gov2.get_halt_sources()) == 0
        print(f"  Not halted with reason: is_halted()={gov2.is_halted()}, sources={gov2.get_halt_sources()} OK")

        # Edge case 3: Old format with cumulative loss reason
        with open(state_file, "w") as f:
            json.dump({
                "halted": True,
                "halt_reason": "cumulative_loss_r=6.50 >= 6.0",
                "cumulative_loss_r": 6.5,
            }, f)
        gov3 = RiskGovernor(state_file=state_file)
        assert gov3.is_halted()
        assert "risk_limit" in gov3.get_halt_sources()
        print(f"  Cumulative loss reason: sources={gov3.get_halt_sources()} OK")

        print(f"\n  PASS: All migration edge cases handled correctly")
        return True
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


if __name__ == "__main__":
    results = []
    results.append(("Migration old state", test_1_migration_old_state_file()))
    results.append(("Migration edge cases", test_migration_edge_cases()))
    results.append(("Dedup halt sources", test_2_dedup_halt_sources()))
    results.append(("R-multiple unification", test_3_r_multiple_unification()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    if all(r[1] for r in results):
        print("\n  All tests PASSED!")
    else:
        print("\n  Some tests FAILED!")
        sys.exit(1)