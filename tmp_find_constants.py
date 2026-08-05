import os

targets = [
    "MAX_DAILY_LOSS",
    "MAX_DAILY_LOSS_USD",
    "MAX_DRAWDOWN_HALT",
]

# Only match exact variable names as tokens
token_targets = {t: None for t in targets}

def iter_py_files(root="."):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)

def find_in_file(path: str):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except Exception:
        return []

    hits = []
    for t in targets:
        # token-ish search: occurrences of the name
        # We still output the full line containing the match.
        if t in txt:
            for i, line in enumerate(txt.splitlines()):
                if t in line:
                    hits.append((t, line.strip()))
    # De-dup (target, line) while preserving order
    seen = set()
    out = []
    for t, line in hits:
        key = (t, line)
        if key not in seen:
            seen.add(key)
            out.append((t, line))
    return out

results = []
for p in iter_py_files("."):
    hits = find_in_file(p)
    if hits:
        print("FILE:", p)
        for t, line in hits:
            print(f"  MATCH: {t} :: {line}")
        print()
