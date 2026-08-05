from pathlib import Path
import re

path = Path('execution/post_entry/post_entry_manager.py')
text = path.read_text(encoding='utf-8', errors='ignore')
lines = text.splitlines()

patterns = [
    ('import time', re.compile(r'\bimport\s+time\b')),
    ('time =', re.compile(r'^\s*time\s*=')),
    ('time=', re.compile(r'\btime\s*=\s*[^=]')),
    ('for time in', re.compile(r'^\s*for\s+time\s+in\b')),
]

print('FILE', path)
print('--- matched lines ---')
for idx, line in enumerate(lines, 1):
    for name, pat in patterns:
        if pat.search(line):
            print(idx, line.rstrip())
            break

# show defs near emergency/precheck/per-position markers
keywords = ['emergency', 'precheck', 'per-position', 'per_position', 'for pos in positions']
print('--- defs/snippets near keywords ---')
for idx, line in enumerate(lines, 1):
    low = line.lower()
    if any(k in low for k in ['def ']):
        if any(k in low for k in keywords):
            print('DEF', idx, line.rstrip())

