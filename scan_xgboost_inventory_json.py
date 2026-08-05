from __future__ import annotations

import json
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

PATTERNS = [(re.compile(re.escape(t), re.IGNORECASE), t) for t in TOKENS]

ROOT = Path(".")

py_files = sorted([p for p in ROOT.rglob("*.py")])

hits = []
for path in py_files:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue

    for i, line in enumerate(txt.splitlines(), 1):
        for pat, tok in PATTERNS:
            if pat.search(line):
                hits.append(
                    {
                        "file": str(path),
                        "line": int(i),
                        "token": tok,
                        "snippet": line.strip(),
                    }
                )
                break

out = {
    "total_py_files_scanned": len(py_files),
    "total_hits": len(hits),
    "hits": hits,
}

out_path = ROOT / "scan_xgboost_inventory.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print(f"WROTE={out_path}")
print(f"TOTAL_PY_FILES_SCANNED={len(py_files)}")
print(f"TOTAL_HITS={len(hits)}")

