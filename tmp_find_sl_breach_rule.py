from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEARCH_DIRS = [
    ROOT / "execution",
    ROOT / "core",
]

PATTERNS = [
    "stop_loss_breach_rule",
    "sl_breach",
    "sl_breach_rule",
    "breach_rule",
    "stop loss breach",
    "breached",
    "stop_loss",
    "order_send",
    "position.sl",
    "position_sl",
    "request.sl",
]


def iter_py_files():
    for d in SEARCH_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*.py"):
            yield p


def main() -> None:
    print(f"ROOT={ROOT}")
    matches = []
    for p in iter_py_files():
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lower = txt.lower()
        for pat in PATTERNS:
            if pat.lower() in lower:
                # report first line containing pat
                lines = txt.splitlines()
                for i, line in enumerate(lines, 1):
                    if pat.lower() in line.lower():
                        snippet = line.strip()
                        matches.append((str(p), i, pat, snippet))
                        break
    # de-dup by (file,line,pat)
    seen = set()
    dedup = []
    for m in matches:
        key = (m[0], m[1], m[2])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(m)

    # sort
    dedup.sort(key=lambda x: (x[0], x[1], x[2]))

    print("\n=== Matches ===")
    for file, line, pat, snippet in dedup:
        print(f"{file}:{line}  pattern={pat}  line={snippet}")


if __name__ == "__main__":
    main()

