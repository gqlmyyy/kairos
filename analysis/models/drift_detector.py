# Trading Bot V3 - analysis/models/drift_detector.py

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("drift_detector")


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / max(len(vals), 1)
    return m, math.sqrt(var)


def detect_feature_drift(
    current_features: List[List[float]],
    training_stats: Optional[Dict[str, Any]],
    threshold: float = 0.35,
) -> Dict[str, Any]:
    """Light drift detection.

    - current_features: X rows (list of 12-d vectors)
    - training_stats: expected json with per-feature mean/std

    Returns:
      {"drift": bool, "score": float, "details": {...}}
    """
    if not current_features:
        return {"drift": False, "score": 0.0}

    # compute current mean/std per feature index
    n_feat = len(current_features[0])
    cols: List[List[float]] = [[] for _ in range(n_feat)]
    for row in current_features:
        for i in range(n_feat):
            cols[i].append(float(row[i]))

    current = {str(i): _mean_std(cols[i]) for i in range(n_feat)}

    if not training_stats:
        return {"drift": False, "score": 0.0, "reason": "no_training_stats"}

    # training_stats expected format: {"features": {"0": {"mean":...,"std":...}, ...}}
    feat_stats = training_stats.get("features", {}) if isinstance(training_stats, dict) else {}

    score = 0.0
    details = {}
    for i in range(n_feat):
        key = str(i)
        tr = feat_stats.get(key)
        if not tr:
            continue
        tr_mean = float(tr.get("mean", 0.0))
        tr_std = float(tr.get("std", 0.0))
        cur_mean, cur_std = current[key]

        denom = tr_std if abs(tr_std) > 1e-9 else 1.0
        z = abs(cur_mean - tr_mean) / denom
        score = max(score, z)
        details[key] = {"z": z, "cur_mean": cur_mean, "tr_mean": tr_mean}

    drift = score >= threshold
    return {"drift": drift, "score": float(score), "details": details}

