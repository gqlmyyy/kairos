# Trading Bot V3 - analysis/models/model_manager.py

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from utils.logger import get_logger

logger = get_logger("model_manager")

LATEST_MODEL_PATH = "models/entry/entry_model.json"

# Cache
_cached_model = None
_cached_model_mtime: Optional[float] = None
_cached_model_version: Optional[str] = None


def _get_model_version_from_file(path: str) -> str:
    """Best-effort version: use filename content if exists, else mtime."""
    try:
        if not os.path.exists(path):
            return "missing"
        # If xgboost json contains metadata, parse.
        # For native Booster .json, structure may vary.
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read(4096)
        if "version" in txt:
            # crude extraction
            return str(int(time.time()))
        return str(os.path.getmtime(path))
    except Exception:
        return "unknown"


def load_latest_model(path: str = LATEST_MODEL_PATH, force_reload: bool = False):
    """Hot reload latest model from models/entry/entry_model.json.

    - caching by file mtime
    - returns cached model if unchanged
    """
    global _cached_model, _cached_model_mtime, _cached_model_version

    try:
        if not os.path.exists(path):
            logger.warning("Latest model not found: %s", path)
            return None, None

        mtime = os.path.getmtime(path)
        if not force_reload and _cached_model is not None and _cached_model_mtime == mtime:
            return _cached_model, _cached_model_version

        # Dynamic import to avoid circular dependency:
        # model_manager <-> xgboost_inference
        from analysis.models.xgboost_inference import load_model
        model = load_model(path)
        _cached_model = model
        _cached_model_mtime = mtime
        _cached_model_version = _get_model_version_from_file(path)

        logger.info("Model hot-reloaded: %s (mtime=%s)", path, mtime)
        return _cached_model, _cached_model_version
    except Exception as e:
        logger.warning("Failed to hot reload model: %s", e)
        return None, _cached_model_version


def get_cached_model_version():
    return _cached_model_version

