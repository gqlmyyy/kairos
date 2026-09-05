"""Feature redundancy report.

Purely diagnostic. It measures how much the feature set duplicates itself
and prints the worst offenders — it **never removes a feature**. High
correlation is not by itself a reason to drop anything: two correlated
features can still split differently in a tree, and which one to keep is a
question for the ablation / feature-selection stage with a target in hand,
not for a correlation matrix computed before any model exists.

Reported per pair:
  * Pearson |r|   -- linear duplication
  * Spearman |rho| -- monotone duplication, which catches pairs that are the
    same information on a different scale (e.g. a raw and an ATR-normalised
    version of the same quantity) even when the linear fit is poor.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Pairs at or above this |r| are listed as "highly redundant". Chosen to be
# informative rather than prescriptive -- nothing is dropped at any value.
HIGH_REDUNDANCY_THRESHOLD = 0.95
MODERATE_REDUNDANCY_THRESHOLD = 0.85
# |r| at or above this is treated as an EXACT duplicate: the two features are
# the same number, usually because two config blocks computed the same
# formula under different names. Reported separately from merely-correlated
# pairs because the finding is different in kind -- but still never removed
# automatically.
EXACT_DUPLICATE_THRESHOLD = 0.999999


@dataclass
class RedundancyPair:
    feature_a: str
    feature_b: str
    pearson: float
    spearman: float
    same_timeframe: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RedundancyReport:
    feature_count: int
    rows_analyzed: int
    exact_duplicates: list = field(default_factory=list)
    high: list = field(default_factory=list)
    moderate: list = field(default_factory=list)
    constant_features: list = field(default_factory=list)
    high_threshold: float = HIGH_REDUNDANCY_THRESHOLD
    moderate_threshold: float = MODERATE_REDUNDANCY_THRESHOLD

    def as_dict(self) -> dict:
        return {
            "feature_count": self.feature_count,
            "rows_analyzed": self.rows_analyzed,
            "high_threshold": self.high_threshold,
            "moderate_threshold": self.moderate_threshold,
            "constant_features": self.constant_features,
            "exact_duplicates": [p.as_dict() for p in self.exact_duplicates],
            "high": [p.as_dict() for p in self.high],
            "moderate": [p.as_dict() for p in self.moderate],
        }


def _timeframe_of(name: str, specs_by_name: dict) -> str:
    spec = specs_by_name.get(name)
    return spec.timeframe if spec is not None else "?"


def analyze_redundancy(
    dataset: pd.DataFrame,
    feature_names: list[str],
    specs: list | None = None,
    high_threshold: float = HIGH_REDUNDANCY_THRESHOLD,
    moderate_threshold: float = MODERATE_REDUNDANCY_THRESHOLD,
) -> RedundancyReport:
    specs_by_name = {s.name: s for s in (specs or [])}
    frame = dataset[feature_names].astype(float)

    # A constant column has no correlation with anything (undefined), so it
    # is reported separately rather than silently producing NaN pairs.
    nunique = frame.nunique(dropna=True)
    constant = sorted(nunique[nunique <= 1].index.tolist())
    varying = [c for c in feature_names if c not in set(constant)]

    report = RedundancyReport(
        feature_count=len(feature_names), rows_analyzed=len(frame),
        constant_features=constant,
        high_threshold=high_threshold, moderate_threshold=moderate_threshold,
    )
    if len(varying) < 2:
        return report

    sub = frame[varying]
    pearson = sub.corr(method="pearson").abs()
    spearman = sub.corr(method="spearman").abs()

    # Upper triangle only: each unordered pair considered once.
    mask = np.triu(np.ones(pearson.shape, dtype=bool), k=1)
    for i, a in enumerate(varying):
        for j, b in enumerate(varying):
            if not mask[i, j]:
                continue
            r = pearson.iat[i, j]
            rho = spearman.iat[i, j]
            if np.isnan(r) and np.isnan(rho):
                continue
            worst = np.nanmax([r, rho])
            if worst < moderate_threshold:
                continue
            pair = RedundancyPair(
                feature_a=a, feature_b=b,
                pearson=float(r) if not np.isnan(r) else float("nan"),
                spearman=float(rho) if not np.isnan(rho) else float("nan"),
                same_timeframe=_timeframe_of(a, specs_by_name) == _timeframe_of(b, specs_by_name),
            )
            if worst >= EXACT_DUPLICATE_THRESHOLD:
                report.exact_duplicates.append(pair)
            elif worst >= high_threshold:
                report.high.append(pair)
            else:
                report.moderate.append(pair)

    report.exact_duplicates.sort(key=lambda p: (p.feature_a, p.feature_b))
    report.high.sort(key=lambda p: -np.nanmax([p.pearson, p.spearman]))
    report.moderate.sort(key=lambda p: -np.nanmax([p.pearson, p.spearman]))
    return report


def format_report(report: RedundancyReport, max_rows: int = 40) -> str:
    lines = [
        "# Feature redundancy report",
        "",
        f"Features analyzed: {report.feature_count}   Rows: {report.rows_analyzed}",
        f"Thresholds: high >= {report.high_threshold}, moderate >= {report.moderate_threshold}",
        "",
        "**Nothing is removed by this report.** Correlated features are listed so a",
        "human can decide during ablation / feature selection. Two correlated inputs",
        "can still be useful to a tree model, and which one to keep depends on the",
        "target -- a question this report deliberately does not answer.",
        "",
    ]
    if report.constant_features:
        lines += [
            f"## Constant features ({len(report.constant_features)})",
            "",
            "No variance in this dataset, so no correlation is defined. Worth checking:",
            "a constant feature carries no information for the model.",
            "",
        ] + [f"- `{c}`" for c in report.constant_features] + [""]

    if report.exact_duplicates:
        lines += [
            f"## Exact duplicates ({len(report.exact_duplicates)} pairs)",
            "",
            "|r| == 1: these are the SAME number under two names. Usually two config",
            "blocks computing one formula. Still not removed here -- but this is the",
            "first place to look during feature selection, since one of each pair adds",
            "no information at all.",
            "",
            "| Feature A | Feature B | Same TF |",
            "|---|---|---|",
        ]
        for p in report.exact_duplicates:
            lines.append(f"| `{p.feature_a}` | `{p.feature_b}` | {'yes' if p.same_timeframe else 'no'} |")
        lines.append("")

    for title, pairs in (("Highly redundant", report.high), ("Moderately redundant", report.moderate)):
        lines += [f"## {title} ({len(pairs)} pairs)", ""]
        if not pairs:
            lines += ["_None._", ""]
            continue
        lines += ["| Feature A | Feature B | \\|Pearson\\| | \\|Spearman\\| | Same TF |",
                  "|---|---|---|---|---|"]
        for p in pairs[:max_rows]:
            lines.append(
                f"| `{p.feature_a}` | `{p.feature_b}` | {p.pearson:.4f} | "
                f"{p.spearman:.4f} | {'yes' if p.same_timeframe else 'no'} |"
            )
        if len(pairs) > max_rows:
            lines.append(f"| _... {len(pairs) - max_rows} more pairs, see the JSON report_ | | | | |")
        lines.append("")
    return "\n".join(lines)
