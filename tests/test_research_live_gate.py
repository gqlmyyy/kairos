"""The live gate: research_v2 + PRODUCTION_ELIGIBLE only, or blocked.

These tests pin the two constraints that make this route safe on a real
account — the generation and the status — and prove the gate cannot be
talked into widening either, or into reaching the legacy artifact.
"""

from __future__ import annotations

import json

import pytest

from analysis.research import live_gate as gate
from analysis.research import model_loader as loader
from analysis.research import model_registry as reg


@pytest.fixture(autouse=True)
def _clean():
    gate.reset_cache()
    yield
    gate.reset_cache()


SYMBOLS = ("EURUSD", "GBPUSD", "XAUUSD")
TIMEFRAMES = ("M15", "H1", "H4")


def write_registry(tmp_path, entries):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"models": entries}), encoding="utf-8")
    return path


def entry(symbol, timeframe, *, status, version="research_v2", path="models/research/x"):
    """A registry entry carrying every field the real schema requires — an
    incomplete one is rejected at parse time, which would mask whether the
    status/version pinning under test actually fired."""
    return {
        "model_id": f"{version}__{symbol}__{timeframe}",
        "symbol": symbol, "timeframe": timeframe, "version": version,
        "status": status, "path": path,
        "feature_count": 80,
        "feature_schema_version": "research-1.2.0",
        "model_hash": "f0ded2146f85c584b5ad7a5d03d660819141e993a855c04c45726efb4f588896",
        "research_verdict": "NO_SIGNAL",
        "target": "TP before SL within the horizon",
    }


class TestPinnedConstraints:
    def test_the_gate_pins_generation_and_status_as_constants(self):
        """Not parameters with defaults — a caller must not be able to widen
        either one."""
        assert gate.VERSION == "research_v2"
        assert gate.REQUIRED_STATUSES == (reg.PRODUCTION_ELIGIBLE,)

    def test_predict_entry_exposes_no_status_or_version_argument(self):
        import inspect
        params = set(inspect.signature(gate.predict_entry).parameters)
        assert "statuses" not in params, "a caller could serve RESEARCH models"
        assert "version" not in params, "a caller could serve another generation"

    def test_load_production_model_exposes_no_status_or_version_argument(self):
        import inspect
        params = set(inspect.signature(gate.load_production_model).parameters)
        assert "statuses" not in params
        assert "version" not in params


class TestShippedStateBlocksEverything:
    """Against the real registry.json as shipped: nothing is
    PRODUCTION_ELIGIBLE, so all nine slots must block."""

    @pytest.mark.parametrize("symbol", SYMBOLS)
    @pytest.mark.parametrize("timeframe", TIMEFRAMES)
    def test_every_slot_blocks_today(self, symbol, timeframe):
        result = gate.predict_entry(
            symbol=symbol, timeframe=timeframe, row={}, entry_direction="BUY")
        assert result["available"] is False
        assert result["p_win"] is None
        assert result["status"] == gate.STATUS_MODEL_MISSING

    def test_the_reason_names_the_required_status(self):
        result = gate.predict_entry(
            symbol="XAUUSD", timeframe="H1", row={}, entry_direction="BUY")
        assert "PRODUCTION_ELIGIBLE" in result["reason"]

    def test_no_shipped_model_is_production_eligible(self):
        registry = reg.load_registry()
        assert registry.by_status(reg.PRODUCTION_ELIGIBLE) == [], (
            "a model became PRODUCTION_ELIGIBLE — the live gate would now serve "
            "it, so this test is the intended tripwire, not a failure to fix "
            "by deleting")


class TestStatusIsEnforced:
    def test_a_research_status_model_is_refused(self, tmp_path):
        path = write_registry(tmp_path, [entry("XAUUSD", "H1", status=reg.RESEARCH)])
        result = gate.predict_entry(
            symbol="XAUUSD", timeframe="H1", row={}, entry_direction="BUY",
            registry_path=path)
        assert result["status"] == gate.STATUS_MODEL_MISSING
        assert result["available"] is False

    def test_a_candidate_status_model_is_refused(self, tmp_path):
        path = write_registry(tmp_path, [entry("XAUUSD", "H1", status=reg.CANDIDATE)])
        result = gate.predict_entry(
            symbol="XAUUSD", timeframe="H1", row={}, entry_direction="BUY",
            registry_path=path)
        assert result["status"] == gate.STATUS_MODEL_MISSING

    @pytest.mark.parametrize("status", [reg.RESEARCH, reg.CANDIDATE, reg.VALIDATED])
    def test_no_servable_status_short_of_production_eligible_passes(self, tmp_path, status):
        path = write_registry(tmp_path, [entry("XAUUSD", "H1", status=status)])
        result = gate.predict_entry(
            symbol="XAUUSD", timeframe="H1", row={}, entry_direction="BUY",
            registry_path=path)
        assert result["available"] is False


class TestGenerationIsEnforced:
    def test_research_v3_is_not_served_even_when_production_eligible(self, tmp_path):
        """The user's constraint: research_v2 is the single source. A v3 model
        at the right status must still be refused."""
        path = write_registry(tmp_path, [
            entry("XAUUSD", "H1", status=reg.PRODUCTION_ELIGIBLE, version="research_v3"),
        ])
        result = gate.predict_entry(
            symbol="XAUUSD", timeframe="H1", row={}, entry_direction="BUY",
            registry_path=path)
        assert result["available"] is False
        assert result["status"] == gate.STATUS_MODEL_MISSING

    def test_both_generations_registered_still_resolves_only_v2(self, tmp_path):
        """Two generations at the same slot would make an unpinned registry
        raise. Pinning to research_v2 must resolve cleanly instead."""
        path = write_registry(tmp_path, [
            entry("XAUUSD", "H1", status=reg.PRODUCTION_ELIGIBLE, version="research_v2",
                   path=str(tmp_path / "nonexistent_v2")),
            entry("XAUUSD", "H1", status=reg.PRODUCTION_ELIGIBLE, version="research_v3",
                   path=str(tmp_path / "nonexistent_v3")),
        ])
        with pytest.raises(loader.ModelNotCompatible) as excinfo:
            gate.load_production_model("XAUUSD", "H1", registry_path=path)
        # It got past resolution (no "2 models registered" ambiguity error) and
        # failed on the missing artifact instead — proving v2 was selected.
        assert "nonexistent_v2" in str(excinfo.value)


def write_activation(tmp_path, records):
    path = tmp_path / gate.ACTIVATION_FILENAME
    path.write_text(json.dumps({"activated": records}), encoding="utf-8")
    return path


class TestActivationIsSeparateFromEligibility:
    """production_gate: "turning a model on remains a separate, human,
    out-of-band act". Eligibility is necessary; activation is the second,
    independent condition."""

    def test_nothing_is_activated_by_default(self):
        assert gate.load_activations() == {}

    def test_an_eligible_but_unactivated_model_is_refused(self, tmp_path):
        path = write_registry(tmp_path, [
            entry("XAUUSD", "H1", status=reg.PRODUCTION_ELIGIBLE,
                   path=str(tmp_path / "artifact")),
        ])
        result = gate.predict_entry(
            symbol="XAUUSD", timeframe="H1", row={}, entry_direction="BUY",
            registry_path=path)
        assert result["available"] is False
        # It resolved past the status check and stopped somewhere after it —
        # eligibility alone did not serve the model.
        assert result["status"] != "OK"

    def test_activation_must_name_the_artifact_hash(self):
        assert "model_hash" in gate.REQUIRED_ACTIVATION_FIELDS

    def test_an_activation_record_missing_fields_is_ignored(self, tmp_path):
        write_registry(tmp_path, [])
        write_activation(tmp_path, [{"model_id": "research_v2__XAUUSD__H1"}])
        loaded = gate.load_activations(tmp_path / "registry.json")
        assert loaded == {}, "an incomplete activation record must not activate"

    def test_a_complete_activation_record_is_read(self, tmp_path):
        write_registry(tmp_path, [])
        write_activation(tmp_path, [{
            "model_id": "research_v2__XAUUSD__H1",
            "model_hash": "ABCDEF",
            "activated_by": "operator",
            "activated_at_utc": "2026-08-27T00:00:00+00:00",
        }])
        loaded = gate.load_activations(tmp_path / "registry.json")
        assert loaded == {"research_v2__XAUUSD__H1": "abcdef"}

    def test_an_unreadable_activation_file_activates_nothing(self, tmp_path):
        write_registry(tmp_path, [])
        (tmp_path / gate.ACTIVATION_FILENAME).write_text("{ broken", encoding="utf-8")
        assert gate.load_activations(tmp_path / "registry.json") == {}

    def test_activation_without_eligibility_still_blocks(self, tmp_path):
        """The other direction: activating a RESEARCH-status model must not
        serve it either. Both conditions, or nothing."""
        path = write_registry(tmp_path, [entry("XAUUSD", "H1", status=reg.RESEARCH)])
        write_activation(tmp_path, [{
            "model_id": "research_v2__XAUUSD__H1",
            "model_hash": "f0ded2146f85c584b5ad7a5d03d660819141e993a855c04c45726efb4f588896",
            "activated_by": "operator",
            "activated_at_utc": "2026-08-27T00:00:00+00:00",
        }])
        result = gate.predict_entry(
            symbol="XAUUSD", timeframe="H1", row={}, entry_direction="BUY",
            registry_path=path)
        assert result["available"] is False
        assert result["status"] == gate.STATUS_MODEL_MISSING


class TestNoLegacyFallback:
    def test_the_gate_never_references_the_legacy_artifact(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(gate))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                  ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docstrings.add(doc)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in docstrings:
                    continue
                assert "entry_model.json" not in node.value, (
                    f"live gate has a live reference to the legacy artifact: "
                    f"{node.value!r}")

    def test_a_missing_registry_blocks_rather_than_falling_back(self, tmp_path):
        missing = tmp_path / "absent.json"
        result = gate.predict_entry(
            symbol="XAUUSD", timeframe="H1", row={}, entry_direction="BUY",
            registry_path=missing)
        assert result["available"] is False
        assert result["p_win"] is None

    def test_an_empty_registry_blocks_rather_than_falling_back(self, tmp_path):
        path = write_registry(tmp_path, [])
        result = gate.predict_entry(
            symbol="XAUUSD", timeframe="H1", row={}, entry_direction="BUY",
            registry_path=path)
        assert result["status"] == gate.STATUS_MODEL_MISSING
        assert result["available"] is False


class TestResultShape:
    def test_the_blocked_result_matches_what_main_consumes(self):
        result = gate.predict_entry(
            symbol="XAUUSD", timeframe="H1", row={}, entry_direction="BUY")
        # main.py reads v2_result["p_win"] and v2_result["available"]
        for key in ("p_win", "available", "status", "reason"):
            assert key in result
        assert isinstance(result["available"], bool)

    def test_p_win_is_none_whenever_unavailable(self, tmp_path):
        path = write_registry(tmp_path, [entry("XAUUSD", "H1", status=reg.RESEARCH)])
        result = gate.predict_entry(
            symbol="XAUUSD", timeframe="H1", row={}, entry_direction="BUY",
            registry_path=path)
        assert result["available"] is False
        assert result["p_win"] is None, (
            "a numeric default here would be indistinguishable from a real "
            "low probability at the call site")


class TestConfigWiring:
    def test_research_is_an_accepted_entry_model_version(self):
        import config
        assert "research" in config.SUPPORTED_ENTRY_MODEL_VERSIONS

    def test_v1_remains_the_default(self):
        import config
        assert config.ENTRY_MODEL_VERSION == "v1", (
            "the default must stay v1 — switching to the research path is a "
            "deliberate operator action")

    def test_main_routes_research_to_the_live_gate(self):
        """The wiring itself: main.py must dispatch ENTRY_MODEL_VERSION ==
        'research' to live_gate, passing symbol and timeframe."""
        source = open("main.py", encoding="utf-8-sig").read()
        assert 'ENTRY_MODEL_VERSION == "research"' in source
        assert "live_gate.predict_entry(" in source
        assert "symbol=symbol" in source
        assert "timeframe=TF_DECISION" in source
