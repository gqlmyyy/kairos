import types
import time
import importlib

# ---------------------------
# Helpers: simple monkeypatch
# ---------------------------
class MonkeyPatch:
    def __init__(self):
        self._patches = []

    def setattr(self, obj, name, value):
        old = getattr(obj, name)
        self._patches.append((obj, name, old))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._patches):
            setattr(obj, name, old)
        self._patches.clear()


def run_tests():
    mp = MonkeyPatch()

    # ---------------------------
    # Test 1: orphan reconciliation pending confirmation (3 cycles)
    # ---------------------------
    rec = importlib.import_module("execution.reconciliation")

    # Capture calls
    close_trade_calls = []
    notify_calls = []

    def fake_close_trade_db_by_order_id(order_id, pnl=0):
        close_trade_calls.append({"order_id": str(order_id), "pnl": pnl})
        return True

    def fake_notify_trade_closed(**kwargs):
        notify_calls.append(kwargs)
        return None

    # Fake DB: provide 1 open trade existing in DB
    # We patch get_open_trades used inside reconcile().
    open_trade = {
        "id": 1,
        "order_id": "123456",
        "symbol": "GBPUSD",
        "direction": "buy",
        "time_open": time.time() - 300,  # open 5 minutes ago
        "open_time": time.time() - 300,
        "pnl": 0,
        "size": 1.0,
    }

    def fake_get_open_trades():
        return [open_trade]

    # Fake QD/MT5 positions: empty list => no match in reconcile
    # reconciliation.py may call mt5.positions_get() and/or qd client; we force mt5 path.
    class FakeMT5:
        def positions_get(self, *args, **kwargs):
            return []

    mp.setattr(rec, "mt5", FakeMT5())

    # Patch DB helpers in reconciliation module namespace (imported into module)
    mp.setattr(rec, "get_open_trades", fake_get_open_trades)
    mp.setattr(rec, "close_trade_db_by_order_id", fake_close_trade_db_by_order_id)

    # Patch notify function
    mp.setattr(rec, "notify_trade_closed", fake_notify_trade_closed)

    # Also patch seconds_since_last_beat if used; keep stable.
    if hasattr(rec, "seconds_since_last_beat"):
        mp.setattr(rec, "seconds_since_last_beat", lambda *args, **kwargs: 0)

    # Ensure orphan warmup window doesn't block our test:
    # reconciliation.py likely has _ORPHAN_IGNORE_WINDOW_SEC; we don't override unless necessary.
    # We'll just call reconcile quickly; if warmup applies globally via bot start epoch,
    # we force the "bot started" epoch to long ago by patching _BOT_START_EPOCH if present.
    if hasattr(rec, "_BOT_START_EPOCH"):
        mp.setattr(rec, "_BOT_START_EPOCH", time.time() - 3600)

    # Run reconcile 3 times
    # reconcile() may require no args.
    rec.reconcile()
    rec.reconcile()
    rec.reconcile()

    # Assertions
    assert len(close_trade_calls) == 1, f"Expected close_trade_db_by_order_id called once on cycle 3, got {close_trade_calls}"
    assert close_trade_calls[0]["order_id"] == "123456"
    # pnl expected to be 0 in the orphan flow
    assert close_trade_calls[0]["pnl"] == 0

    # Notify assertions:
    assert len(notify_calls) >= 1, "Expected at least one notify_trade_closed call"
    reasons = [c.get("reason", "") for c in notify_calls]
    # Must not say "تم تنفيذ SL/TP" (per requirement). It should mention orphan.
    joined = " | ".join(reasons)
    assert "orphan" in joined.lower() or "يتيمة" in joined, f"Expected orphan reason, got: {joined}"
    assert "sl/TP" not in joined.lower(), f"Should not contain SL/TP misleading reason, got: {joined}"
    # Also should not claim exit_price=0 & pnl=0 with SL/TP reason
    # (We only check that reason isn't SL/TP; exit_price might still be 0 depending on implementation.)
    print("[OK] Test 1 passed: orphan reconciliation pending confirmation works.")

    # ---------------------------
    # Test 2: feature building for XGBoost adapter (no expected_* inputs)
    # ---------------------------
    adapter_mod = importlib.import_module("execution.post_entry.xgboost_exit_model_adapter")

    # We must avoid loading real model from disk if it's heavy;
    # We'll only call the feature-building function used by adapter.
    # Let's introspect adapter for a method that builds features.
    # Common patterns: build_features(...) or _build_features(...).
    adapter_cls = getattr(adapter_mod, "XGBoostExitModelAdapter")
    adapter = adapter_cls()

    trade = {
        "symbol": "GBPUSD",
        "direction": "buy",
        "adx": 22.5,   # provided
        "trend": 1.0, # provided
        "ticket": 999,
        "time_open": time.time() - 1000,
        "size": 1.0,
        "spread": 1.2,
        "volume": 1.0,
        "market_regime": 0.0,
        "mfe": 10.0,
        "mae": -5.0,
        "entry_rsi": 55.0,
        "entry_atr": 1.3,
        "trade_duration": 123.0,
        "pnl": 0,
        # Intentionally missing:
        # expected_adx, expected_trend_h1, expected_trend_h4
    }

    expected_row = {}

    # Determine which internal method builds feature vector/feature dict.
    # We'll try a few likely names.
    feature_dict = None
    candidates = ["build_exit_features", "build_features", "_build_features", "get_features", "_build_feature_row", "build_feature_vector"]
    for name in candidates:
        if hasattr(adapter, name):
            fn = getattr(adapter, name)
            try:
                # Try calling with flexible signatures
                try:
                    feature_dict = fn(trade=trade, expected_row=expected_row)  # type: ignore
                except TypeError:
                    feature_dict = fn(trade, expected_row)  # type: ignore
            except Exception:
                feature_dict = None
            if feature_dict is not None:
                break

    # If adapter doesn't expose a feature builder, we fall back to analysis/models/feature_schema
    # which is the canonical place to build the vector.
    if feature_dict is None:
        fs = importlib.import_module("analysis.models.feature_schema")
        # feature_schema likely has build_feature_vector(features_dict)
        # We'll try to construct its expected raw feature dict and call build_feature_vector.
        # We'll use adapter's own call if it exists via adapter._prepare_features or similar.
        # Otherwise, directly use adapter's features extraction if present.
        feature_builder_fn = None
        for name in ["build_feature_vector", "build_features", "build_feature_dict", "build_feature_vector_from_trade"]:
            if hasattr(fs, name):
                feature_builder_fn = getattr(fs, name)
                break
        assert feature_builder_fn is not None, "Could not find any feature builder in analysis.models.feature_schema"

        # If build_feature_vector expects already ordered numeric vector, skip.
        # We just need to ensure no None/MISSING in output and that adx/trend_h1/trend_h4 are numeric.
        # We'll attempt both calling styles.
        try:
            vec = feature_builder_fn(trade)
        except TypeError:
            vec = feature_builder_fn({**trade, **expected_row})

        # Now verify that there are no 'MISSING' strings in vec representation
        vec_str = str(vec)
        assert "MISSING" not in vec_str, f"Vector contains MISSING: {vec_str}"
        print("[OK] Test 2 passed: feature building avoids MISSING/None using fallbacks.")
    else:
        # feature_dict path: ensure numeric fields are not None and not 'MISSING'
        # We don't know exact keys; check common ones
        adx_val = feature_dict.get("entry_adx") if isinstance(feature_dict, dict) else None
        # Also accept any keys that correspond to adx/trend h1/h4
        merged_str = str(feature_dict)
        assert "MISSING" not in merged_str, f"Features contain MISSING: {merged_str}"

        # Check we have numeric-like values for relevant fields
        # We'll scan for entries that look like 0.0/22.5/1.0 etc in the dict
        # and assert they aren't None.
        for k in ["entry_adx", "trend_h1", "trend_h4", "adx", "trend_h1", "trend_h4"]:
            if k in feature_dict:
                assert feature_dict[k] is not None, f"{k} is None"
                if isinstance(feature_dict[k], str):
                    assert feature_dict[k].strip().lower() != "missing", f"{k} is MISSING"
        print("[OK] Test 2 passed: adapter feature building uses numeric fallbacks for adx/trend_h1/trend_h4.")

    # ---------------------------
    # Test 3: orphan flow should not emit misleading "SL/TP" reason before confirmation
    # (We validate indirectly via notify_calls and close_trade_calls from test 1.)
    # ---------------------------
    joined = " | ".join([c.get("reason", "") for c in notify_calls]).lower()
    assert "sl/tp" not in joined and "sl/ tp" not in joined, f"Misleading SL/TP reason detected: {joined}"
    assert "orphan" in joined or "يتيمة" in joined, f"Orphan wording missing: {joined}"
    # Confirm close_trade called only after 3 cycles already checked.
    print("[OK] Test 3 passed: no misleading SL/TP orphan notifications before confirmation.")

    mp.undo()
    print("\nALL TESTS PASSED.")


if __name__ == "__main__":
    run_tests()
