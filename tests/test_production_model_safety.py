"""Nothing may replace the served entry model by accident.

The audit found four trainers writing to `models/entry/entry_model.json`, each
with its own schema, none checking what was there. The deployed artifact is
whichever one ran last. These tests pin the repairs so that cannot recur:

  * training code cannot write to the production path at all,
  * promotion requires an explicit operator opt-in,
  * a candidate that disagrees with its metadata, or with the live feature
    contract, is refused,
  * the daily retrain thread is off unless switched on, and its trigger is no
    longer a constant True wearing the shape of a decision.
"""

from __future__ import annotations

import json
import os

import pytest

from analysis.models import entry_model_metadata as md
from analysis.models import production_model_guard as guard
from analysis.models import entry_feature_spec as spec


def _payload(**overrides):
    payload = md.build(
        model_version="test-1",
        training_pipeline_id="tests/test_production_model_safety.py",
        feature_names=list(spec.FEATURE_NAMES),
        dataset_fingerprint="sha256:" + "a" * 64,
        training_start="2024-01-01T00:00:00+00:00",
        training_end="2024-06-01T00:00:00+00:00",
        validation_start="2024-06-01T00:00:00+00:00",
        validation_end="2024-09-01T00:00:00+00:00",
        test_start="2024-09-01T00:00:00+00:00",
        test_end="2024-12-01T00:00:00+00:00",
        symbol_scope=["EURUSD"],
        timeframe_scope=["H4", "H1"],
        target_definition="first touch of TP before SL",
        target_horizon=24,
        git_commit="b" * 40,
    )
    payload.update(overrides)
    return payload


@pytest.fixture
def candidate(tmp_path):
    xgb = pytest.importorskip("xgboost")
    np = pytest.importorskip("numpy")
    X = np.random.rand(80, spec.FEATURE_COUNT)
    y = (np.random.rand(80) > 0.5).astype(float)
    d = xgb.DMatrix(X, label=y, feature_names=list(spec.FEATURE_NAMES))
    booster = xgb.train({"objective": "binary:logistic", "max_depth": 2, "seed": 1},
                        d, num_boost_round=3)
    path = str(tmp_path / "candidate.json")
    booster.save_model(path)
    return path


class TestTrainersCannotWriteProduction:
    def test_the_production_path_is_rejected(self):
        with pytest.raises(guard.ModelInstallRefused, match="production entry model"):
            guard.assert_not_production(guard.PRODUCTION_MODEL_PATH)

    def test_a_relative_detour_to_the_same_file_is_rejected(self):
        """`models/entry/../entry/entry_model.json` is the same file."""
        sneaky = os.path.join("models", "entry", "..", "entry", "entry_model.json")
        with pytest.raises(guard.ModelInstallRefused):
            guard.assert_not_production(sneaky)

    def test_a_research_path_is_allowed(self, tmp_path):
        guard.assert_not_production(str(tmp_path / "entry_model.json"))

    @pytest.mark.parametrize("module_name,attr", [
        ("analysis.models.xgboost_trainer", "DEFAULT_MODEL_PATH"),
        ("analysis.entry_v2.entry_xgboost_trainer", "DEFAULT_OUTPUT_MODEL_PATH"),
    ])
    def test_legacy_trainers_no_longer_default_to_production(self, module_name, attr):
        import importlib

        module = importlib.import_module(module_name)
        configured = os.path.realpath(os.path.abspath(getattr(module, attr)))
        production = os.path.realpath(os.path.abspath(guard.PRODUCTION_MODEL_PATH))
        assert configured != production, (
            f"{module_name}.{attr} still points at the production model")


class TestInstallRequiresOptIn:
    def test_install_refuses_without_the_env_var(self, candidate, monkeypatch):
        monkeypatch.delenv(guard.INSTALL_ENV_VAR, raising=False)
        with pytest.raises(guard.ModelInstallRefused, match=guard.INSTALL_ENV_VAR):
            guard.install(candidate, _payload())

    def test_the_production_model_is_untouched_by_a_refused_install(
            self, candidate, monkeypatch):
        monkeypatch.delenv(guard.INSTALL_ENV_VAR, raising=False)
        before = (guard.sha256_of(guard.PRODUCTION_MODEL_PATH)
                  if os.path.exists(guard.PRODUCTION_MODEL_PATH) else None)
        with pytest.raises(guard.ModelInstallRefused):
            guard.install(candidate, _payload())
        after = (guard.sha256_of(guard.PRODUCTION_MODEL_PATH)
                 if os.path.exists(guard.PRODUCTION_MODEL_PATH) else None)
        assert before == after


class TestInstallValidatesTheCandidate:
    """Even with the opt-in set, a wrong model must not be installable."""

    @pytest.fixture(autouse=True)
    def _opt_in(self, monkeypatch, tmp_path):
        monkeypatch.setenv(guard.INSTALL_ENV_VAR, "1")
        # Redirect production so a passing install cannot touch the real file.
        monkeypatch.setattr(guard, "PRODUCTION_MODEL_PATH",
                            str(tmp_path / "prod" / "entry_model.json"))
        monkeypatch.setattr(guard, "BACKUP_ROOT", str(tmp_path / "backup"))

    def test_a_mismatched_feature_count_is_refused(self, candidate):
        names = list(spec.FEATURE_NAMES) + ["extra"]
        with pytest.raises(guard.ModelInstallRefused, match="disagrees"):
            guard.install(candidate, _payload(
                feature_names=names, feature_count=len(names),
                feature_order=list(range(len(names)))))

    def test_reordered_names_are_refused(self, candidate):
        """Caught against the artifact's own names, which this booster carries."""
        names = list(spec.FEATURE_NAMES)
        names[0], names[-1] = names[-1], names[0]
        with pytest.raises(guard.ModelInstallRefused, match="disagrees"):
            guard.install(candidate, _payload(feature_names=names))

    def test_reordered_names_are_refused_even_when_the_artifact_is_nameless(
            self, tmp_path):
        """The case that actually matters.

        Every model file in this repository has an empty `feature_names` list,
        so there is nothing in the artifact to compare against. The sidecar is
        then the only description of what the columns mean, and it must be
        checked against the live contract — otherwise a permuted vector would
        install cleanly and score confidently on the wrong columns.
        """
        xgb = pytest.importorskip("xgboost")
        np = pytest.importorskip("numpy")
        d = xgb.DMatrix(np.random.rand(80, spec.FEATURE_COUNT),
                        label=(np.random.rand(80) > 0.5).astype(float))
        booster = xgb.train({"objective": "binary:logistic", "seed": 1}, d,
                            num_boost_round=3)
        path = str(tmp_path / "nameless.json")
        booster.save_model(path)
        assert not (booster.feature_names or ()), "fixture no longer nameless"

        names = list(spec.FEATURE_NAMES)
        names[0], names[-1] = names[-1], names[0]
        with pytest.raises(guard.ModelInstallRefused, match="cannot be served"):
            guard.install(path, _payload(feature_names=names))

    def test_a_non_json_extension_is_refused(self, candidate, tmp_path):
        staged = str(tmp_path / "candidate.bin")
        os.rename(candidate, staged)
        with pytest.raises(guard.ModelInstallRefused, match=r"\.json"):
            guard.install(staged, _payload())

    def test_a_valid_candidate_installs_and_writes_its_sidecar(self, candidate):
        result = guard.install(candidate, _payload())
        assert result["installed"] is True
        assert os.path.exists(guard.PRODUCTION_MODEL_PATH)
        sidecar = md.metadata_path_for(guard.PRODUCTION_MODEL_PATH)
        assert os.path.exists(sidecar), "installed a model with no provenance"
        assert md.load(guard.PRODUCTION_MODEL_PATH).feature_count == spec.FEATURE_COUNT

    def test_installing_backs_up_what_it_replaces(self, candidate):
        guard.install(candidate, _payload())
        first = guard.sha256_of(guard.PRODUCTION_MODEL_PATH)
        result = guard.install(candidate, _payload(model_version="test-2"))
        assert result["backup"] is not None
        backed_up = os.path.join(result["backup"], "entry_model.json")
        assert guard.sha256_of(backed_up) == first


class TestDormantRetrainingIsDisabled:
    def test_auto_retrain_is_off_by_default(self, monkeypatch):
        from analysis.models import system_orchestrator as orch

        monkeypatch.delenv(orch.AUTO_RETRAIN_ENV_VAR, raising=False)
        assert orch.auto_retrain_enabled() is False

    def test_the_daily_cycle_trains_nothing_when_disabled(self, monkeypatch):
        """The whole point: a fired thread must be a no-op, not a retrain."""
        from analysis.models import system_orchestrator as orch

        monkeypatch.delenv(orch.AUTO_RETRAIN_ENV_VAR, raising=False)
        monkeypatch.setattr(orch, "_get_execution_dataset_stats",
                            lambda: {"total_rows": 100_000,
                                     "new_rows_count": 100_000,
                                     "last_updated_at": None})

        called = []
        monkeypatch.setattr(orch, "train_model_from_db",
                            lambda **kw: called.append(kw) or {"ok": True})
        monkeypatch.setattr(orch, "load_latest_model",
                            lambda **kw: called.append(kw) or (None, None))

        orch.run_daily_cycle()
        assert called == [], "the disabled daily cycle still trained or reloaded"

    def test_the_orchestrator_thread_is_not_started_by_main(self):
        """It is imported; it must not be launched."""
        import re

        source = open("main.py", encoding="utf-8-sig").read()
        calls = re.findall(r"^[^#\n]*start_daily_orchestrator_thread\s*\(",
                           source, re.MULTILINE)
        assert not calls, f"main.py starts the retrain thread: {calls}"


class TestShouldRetrainIsNoLongerAlwaysTrue:
    def test_unknown_last_train_time_does_not_authorise_a_retrain(self):
        from analysis.models.xgboost_trainer import should_retrain

        assert should_retrain(new_rows_count=0, last_train_ts=None) is False

    def test_no_new_rows_does_not_retrain_however_old_the_model(self):
        from analysis.models.xgboost_trainer import should_retrain

        assert should_retrain(new_rows_count=0, last_train_ts=0.0) is False

    def test_enough_new_rows_still_triggers(self):
        from analysis.models.xgboost_trainer import should_retrain

        assert should_retrain(new_rows_count=500, last_train_ts=None) is True

    def test_a_negative_delta_is_a_bug_not_a_trigger(self):
        from analysis.models.xgboost_trainer import should_retrain

        with pytest.raises(ValueError):
            should_retrain(new_rows_count=-1, last_train_ts=None)


class TestEntryV2IsQuarantined:
    @pytest.fixture(autouse=True)
    def _no_override(self, monkeypatch):
        from analysis import entry_v2

        monkeypatch.delenv(entry_v2.QUARANTINE_ENV_VAR, raising=False)

    @pytest.mark.parametrize("module_name,func", [
        ("analysis.entry_v2.dataset_builder", "build_dataset"),
        ("analysis.entry_v2.feature_engineering", "generate_features"),
        ("analysis.entry_v2.entry_labels", "generate_entry_labels_v2"),
        ("analysis.entry_v2.entry_xgboost_trainer", "train_entry_xgboost"),
    ])
    def test_the_invalidated_entrypoints_refuse_to_run(self, module_name, func):
        import importlib

        from analysis.entry_v2 import InvalidatedPipelineError

        module = importlib.import_module(module_name)
        with pytest.raises(InvalidatedPipelineError):
            getattr(module, func)(**_kwargs_for(func))

    def test_the_package_is_still_importable_for_inspection(self):
        """Quarantine must not mean 'cannot be audited'."""
        import importlib

        assert importlib.import_module("analysis.entry_v2.feature_schema")

    def test_entry_model_version_v2_is_rejected(self, monkeypatch):
        import importlib

        import config

        monkeypatch.setenv("ENTRY_MODEL_VERSION", "v2")
        with pytest.raises(ValueError, match="quarantined"):
            importlib.reload(config)
        monkeypatch.delenv("ENTRY_MODEL_VERSION")
        importlib.reload(config)
        assert config.ENTRY_MODEL_VERSION == "v1"


def _kwargs_for(func: str) -> dict:
    """Minimal kwargs so the refusal, not a TypeError, is what fires."""
    return {
        "build_dataset": {"output_dir": "/tmp/kairos-quarantine-test"},
        "generate_features": {"dataset_csv_path": "/nonexistent.csv",
                              "output_dir": "/tmp/kairos-quarantine-test"},
        "generate_entry_labels_v2": {},
        "train_entry_xgboost": {"dataset_csv_path": "/nonexistent.csv",
                                "output_dir": "/tmp/kairos-quarantine-test"},
    }[func]
