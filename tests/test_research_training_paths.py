"""One canonical path per artifact, and no path may write over another's.

The defect this file guards against is documented all over this repository:
four trainers wrote three different feature schemas to one filename, and
nothing noticed until live inference had been gated on unrelated numbers for
months. Separation is only real if it is checked.

The rule for research models
----------------------------
KAIROS does not train them. Training lives in the xgbooost research repository,
which owns the datasets, the walk-forward protocol and the null tests. KAIROS's
only entry point is ``scripts/import_research_model.py``, which copies a
finished artifact and generates its card. That is a single source of truth by
construction: there is no second way to produce a research artifact here, so
there is nothing for it to drift against.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

RESEARCH_STORE = "models/research"
LEGACY_STORES = ("models/entry", "models/entry_v2", "models/exit")

#: The ONE module allowed to write into models/research/.
CANONICAL_IMPORTER = "scripts/import_research_model.py"


def _python_files():
    for base in ("scripts", "analysis", "execution", "trade_management", "risk"):
        yield from sorted((ROOT / base).rglob("*.py"))
    yield from sorted(ROOT.glob("*.py"))


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _read(path: Path) -> str:
    # Some pre-existing files in this repository carry a UTF-8 BOM.
    return path.read_text(encoding="utf-8-sig")


def _code_lines(path: Path):
    """(lineno, text) for real code only — docstrings and comments removed.

    A module docstring that says "this never writes to models/entry/" is
    documentation, not a write, and a scan that cannot tell the two apart
    would either miss real calls or force the prose out of the docstrings.
    """
    text = _read(path)
    skip = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT):
                skip.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    return [(i, line) for i, line in enumerate(text.splitlines(), 1) if i not in skip]


def test_only_the_import_script_writes_into_the_research_store():
    """A legacy trainer must never be able to overwrite a research artifact."""
    writers = []
    for path in _python_files():
        rel = _rel(path)
        if rel == CANONICAL_IMPORTER or rel.startswith("analysis/research/"):
            continue
        for i, line in _code_lines(path):
            if RESEARCH_STORE in line:
                writers.append(f"{rel}:{i}")
    assert not writers, (
        f"only {CANONICAL_IMPORTER} may touch {RESEARCH_STORE}/: {writers}")


def test_the_research_package_never_writes_into_a_legacy_store():
    """The boundary holds in both directions."""
    offenders = []
    for path in sorted((ROOT / "analysis" / "research").glob("*.py")):
        for i, line in _code_lines(path):
            for store in LEGACY_STORES:
                if store in line:
                    offenders.append(f"{_rel(path)}:{i} -> {store}")
    assert not offenders, f"research code reaches into a legacy store: {offenders}"


def test_kairos_contains_no_research_model_trainer():
    """Training belongs to the research repo; KAIROS imports finished artifacts.

    A trainer here would be a second source of truth for the same artifact,
    and the two would diverge exactly the way the legacy schemas did.
    """
    offenders = []
    for path in _python_files():
        rel = _rel(path)
        if rel == CANONICAL_IMPORTER:
            continue
        try:
            tree = ast.parse(_read(path), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if name in ("dump",) and isinstance(fn, ast.Attribute):
                    mod = getattr(fn.value, "id", "")
                    if mod == "joblib":
                        offenders.append(f"{rel}:{node.lineno} joblib.dump")
    assert not offenders, (
        f"a .joblib writer outside the importer suggests a second research-model "
        f"producer: {offenders}")


def test_the_research_package_does_not_import_the_legacy_entry_path():
    """Two vocabularies must not be able to reach each other.

    Importing the legacy spec here is how someone would eventually map
    `trend_score` onto `trend_score` — same name, different arithmetic.
    """
    banned = {"analysis.models.entry_feature_spec",
              "analysis.models.xgboost_v2_inference",
              "analysis.models.entry_feature_contract",
              "analysis.entry_v2.inference"}
    offenders = []
    for path in sorted((ROOT / "analysis" / "research").glob("*.py")):
        tree = ast.parse(_read(path), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in banned:
                offenders.append(f"{_rel(path)}:{node.lineno} {node.module}")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in banned:
                        offenders.append(f"{_rel(path)}:{node.lineno} {a.name}")
    assert not offenders, f"research code imports the legacy entry path: {offenders}"


def test_legacy_artifacts_are_preserved_not_replaced():
    """The legacy model must still be on disk and untouched by this work."""
    legacy = ROOT / "models" / "entry" / "entry_model.json"
    if not legacy.exists():
        pytest.skip("models/entry/ is not populated in this checkout")
    assert legacy.stat().st_size > 0
    # And it must still be the 65-feature artifact the legacy tests describe:
    # this integration did not retrain, replace or repair it.
    xgb = pytest.importorskip("xgboost")
    booster = xgb.Booster()
    booster.load_model(str(legacy))
    assert booster.num_features() == 65


def test_the_deprecated_entry_point_still_only_prints_a_pointer():
    """`train_from_historical.py` is a signpost, not a pipeline."""
    text = _read(ROOT / "train_from_historical.py")
    assert "has been replaced" in text
    tree = ast.parse(text)
    imports = {a.name.split(".")[0] for n in ast.walk(tree)
               if isinstance(n, ast.Import) for a in n.names}
    imports |= {n.module.split(".")[0] for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module}
    assert imports <= {"sys"}, f"a deprecated entry point should import nothing: {imports}"
