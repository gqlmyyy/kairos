import os

SKIP_DIRS = {'.git', 'node_modules', '__pycache__', 'models_backup'}

targets = {
    'sltp': [],
    'entry_features_call': [],
    'entry_signal_logic': [],
}

for dirpath, dirnames, filenames in os.walk('.'):
    # prune
    if any(x in dirpath for x in SKIP_DIRS):
        continue

    for fn in filenames:
        if not fn.endswith('.py'):
            continue
        path = os.path.join(dirpath, fn)
        try:
            with open(path, encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 1) sltp real definition
            if 'def ' in content and (
                'sl_tp' in content or 'def sltp' in content or 'calculate_sl_tp' in content or 'compute_sl_tp' in content
            ):
                targets['sltp'].append(path)

            # 2) entry v2 feature generation calls
            # (heuristic: file mentions feature_engineering and is not under entry_v2 folder)
            if 'feature_engineering' in content and 'entry_v2' not in path:
                targets['entry_features_call'].append(path)

            # 3) entry signal logic used by live trading
            if 'signal_engine' in content or 'SIGNAL_FINAL' in content:
                targets['entry_signal_logic'].append(path)
        except Exception:
            pass

for k, v in targets.items():
    # dedupe but keep order
    seen = set()
    out = []
    for p in v:
        if p not in seen:
            out.append(p)
            seen.add(p)

    print(k + ' ->')
    for p in out:
        print('  ' + p)
    print('  (count=' + str(len(out)) + ')')

