from __future__ import annotations

import re
from pathlib import Path

TOKENS = [
    "xgboost",
    "import xgboost",
    "from xgboost",
    "Booster",
    "XGBClassifier",
    "XGBRegressor",
    "load_model",
    "joblib.load",
    "pickle.load",
    "predict_with_entry_v2",
    "predict_with_v2",
    "predict_exit_probability",
    "XGBoostExitModelAdapter",
    "entry_model.json",
    "exit_model.json",
    "predict_proba",
    "predict(",
]

# Build regex list: some tokens are already substrings; use escaped match.
PATTERNS = [(re.compile(re.escape(t), re.IGNORECASE), t) for t in TOKENS]

ROOT = Path(".")

py_files = [p for p in ROOT.rglob("*.py")]

hits: list[tuple[str, str, int, str]] = []

def scan_file(path: Path) -> None:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return

    for i, line in enumerate(txt.splitlines(), 1):
        # Note: do NOT attempt to be clever; just substring/regex token match.
        for pat, tok in PATTERNS:
            if pat.search(line):
                hits.append((str(path), tok, i, line.strip()))
                break


for p in py_files:
    scan_file(p)

for file_path, tok, ln, snippet in hits:
    # Single-line, easy to parse.
    print(f"{file_path}:{ln}:{tok}:{snippet}")

print(f"TOTAL_HITS={len(hits)}")
print(f"TOTAL_PY_FILES_SCANNED={len(py_files)}")

