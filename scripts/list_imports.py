import os
import re
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# simple detection: import ... OR from ... import ...
IMPORT_RE = re.compile(r'^\s*(import\s+|from\s+).*$', re.IGNORECASE)

py_files = []
imps = []

for dirpath, _, filenames in os.walk(ROOT):
    for fn in filenames:
        if fn.endswith('.py'):
            py_files.append(os.path.join(dirpath, fn))

for path in py_files:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                s = line.strip()
                if s.startswith('import ') or s.startswith('from '):
                    # capture only real import statements (exclude comments)
                    if not s.startswith('#'):
                        imps.append((path, i, s))
    except Exception:
        pass

c = Counter([p for p, _, _ in imps])
files_with_import = len(c)

print(f"ROOT={ROOT}")
print(f"py_files={len(py_files)}")
print(f"import_lines={len(imps)}")
print(f"files_with_import={files_with_import}")

print("--- Top files by import line count ---")
for p, cnt in c.most_common(50):
    print(f"{cnt}\t{os.path.relpath(p, ROOT)}")

print("--- All import lines (relative paths) ---")
for path, i, s in imps:
    rel = os.path.relpath(path, ROOT)
    print(f"{rel}:{i}: {s}")

