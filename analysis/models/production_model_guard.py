"""The only sanctioned way to replace the production entry model.

Four trainers used to write `models/entry/entry_model.json` directly, each with
its own schema, none of them checking what was already there. The deployed
artifact is the one that won the last race — not the one that was best, or even
valid.

This module makes that impossible to repeat by accident. Installing a model is
no longer "copy a file"; it requires:

  * a metadata sidecar that the live loader would accept,
  * agreement between that metadata and the booster being installed,
  * an explicit opt-in from the operator,
  * a timestamped backup of whatever is being replaced.

Research code stays research code. A trainer that wants to keep its output puts
it under `models/entry/research/<experiment>/` and never touches the production
path; promoting it is a separate, deliberate, human-triggered act.

The opt-in is an environment variable rather than a function argument on
purpose. An argument can be passed by a scheduler, a retry loop, or a module
imported for an unrelated reason. `KAIROS_ALLOW_MODEL_INSTALL=1` has to be set
by whoever is at the keyboard.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime, timezone
from typing import Optional

from utils.logger import get_logger

from analysis.models import entry_model_metadata as metadata

logger = get_logger("production_model_guard")

PRODUCTION_MODEL_PATH = os.path.join("models", "entry", "entry_model.json")
RESEARCH_ROOT = os.path.join("models", "entry", "research")
BACKUP_ROOT = os.path.join("models_backup", "entry")

INSTALL_ENV_VAR = "KAIROS_ALLOW_MODEL_INSTALL"


class ModelInstallRefused(Exception):
    """Raised when an install does not meet the contract. Never caught to retry."""


def install_allowed() -> bool:
    return os.environ.get(INSTALL_ENV_VAR, "").strip() in {"1", "true", "TRUE", "yes"}


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def research_dir(experiment_id: str) -> str:
    """Where a trainer may write freely. Never the production path."""
    if not experiment_id or os.path.sep in experiment_id or experiment_id in {".", ".."}:
        raise ValueError(f"invalid experiment_id: {experiment_id!r}")
    path = os.path.join(RESEARCH_ROOT, experiment_id)
    os.makedirs(path, exist_ok=True)
    return path


def assert_not_production(path: str) -> None:
    """Guard for trainers: refuse to write to the production artifact.

    Call this at the top of any code that saves a model. It compares resolved
    absolute paths, so a relative path, a symlink or a `..` detour is caught.
    """
    target = os.path.realpath(os.path.abspath(path))
    production = os.path.realpath(os.path.abspath(PRODUCTION_MODEL_PATH))
    if target == production:
        raise ModelInstallRefused(
            f"{path} is the production entry model. Training code must write to "
            f"{RESEARCH_ROOT}/<experiment_id>/ instead; promote deliberately via "
            f"analysis.models.production_model_guard.install()."
        )


def backup_current(reason: str = "replaced") -> Optional[str]:
    """Copy the live model and its sidecar aside. Returns the backup dir."""
    if not os.path.exists(PRODUCTION_MODEL_PATH):
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    destination = os.path.join(BACKUP_ROOT, f"{stamp}_{reason}")
    os.makedirs(destination, exist_ok=True)
    shutil.copy2(PRODUCTION_MODEL_PATH, os.path.join(destination, "entry_model.json"))
    sidecar = metadata.metadata_path_for(PRODUCTION_MODEL_PATH)
    if os.path.exists(sidecar):
        shutil.copy2(sidecar, os.path.join(destination,
                                           os.path.basename(sidecar)))
    logger.info("[ML_CONTRACT] backed up current production model to %s", destination)
    return destination


def install(
    model_path: str,
    metadata_payload: dict,
    *,
    allow: Optional[bool] = None,
) -> dict:
    """Promote a trained model to production, or refuse with a reason.

    `model_path` must be a `.json` file. XGBoost picks its serialisation format
    from the extension, so staging under any other suffix silently writes
    binary UBJSON that the loader cannot parse — a mistake this project has
    already made once.
    """
    if allow is None:
        allow = install_allowed()
    if not allow:
        raise ModelInstallRefused(
            f"install refused: set {INSTALL_ENV_VAR}=1 to promote a model. "
            f"This is deliberate — no scheduler, retry loop or import should be "
            f"able to replace the production model on its own."
        )

    if not os.path.exists(model_path):
        raise ModelInstallRefused(f"no model at {model_path}")
    if not model_path.endswith(".json"):
        raise ModelInstallRefused(
            f"{model_path} must end in .json; XGBoost infers its serialisation "
            f"format from the extension and would write unparseable binary.")

    # The metadata must be something the live loader would accept, checked
    # before anything is moved.
    meta = metadata.parse(metadata_payload)

    try:
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(model_path)
    except Exception as exc:  # noqa: BLE001
        raise ModelInstallRefused(f"candidate at {model_path} is not loadable: {exc}") from exc

    mismatch = metadata.validate_against_booster(meta, booster)
    if mismatch is not None:
        raise ModelInstallRefused(f"candidate disagrees with its metadata: {mismatch}")

    from analysis.models.entry_feature_spec import FEATURE_NAMES as LIVE_NAMES
    serving = metadata.validate_for_serving(meta, live_feature_names=LIVE_NAMES)
    if serving is not None:
        raise ModelInstallRefused(
            f"candidate cannot be served by the live path: {serving}")

    backup = backup_current("replaced")

    os.makedirs(os.path.dirname(PRODUCTION_MODEL_PATH) or ".", exist_ok=True)
    shutil.copy2(model_path, PRODUCTION_MODEL_PATH)
    metadata.write(PRODUCTION_MODEL_PATH, metadata_payload)

    installed_sha = sha256_of(PRODUCTION_MODEL_PATH)
    if installed_sha != sha256_of(model_path):
        raise ModelInstallRefused("copy verification failed: checksums differ")

    logger.info("[ML_CONTRACT] installed %s -> %s sha256=%s",
                model_path, PRODUCTION_MODEL_PATH, installed_sha)
    return {
        "installed": True,
        "path": PRODUCTION_MODEL_PATH,
        "sha256": installed_sha,
        "backup": backup,
        "metadata": meta.describe(),
    }
