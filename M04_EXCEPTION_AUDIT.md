# M-04 — Exception Handler Audit

Scope: every `except` clause in the source tree (332 total, `tests/` and
`scripts/` counted but treated as lower priority — offline/one-shot tooling,
not the live trading path). Goal: find handlers where a caught error silently
becomes **trade approval** rather than a rejection, and fix those. Everything
else is classified and left as-is with the reasoning recorded here.

## Method

A handler was read in its call context (not just the `except` line) and
placed in one of five buckets:

| Class | Meaning |
|---|---|
| **RECOVER** | Falls back to a safe, conservative default and continues (e.g. a fallback point-size when the broker doesn't expose one). |
| **RETRY** | Already handled by a bounded retry loop elsewhere (session reconnect, IPC hiccup). |
| **REJECT** | Correctly turns the error into "this specific trade/action does not proceed," without affecting anything else. |
| **HALT** | Correctly stops new entries system-wide (Risk Governor). |
| **FAIL-OPEN (bug)** | The error was caught and the code proceeded as if the check had *passed* — the class this audit exists to find. |

Raw counts (`ast`-based scan, `tests/`/`scripts/` excluded from the "critical
path" figures below):

- 332 handlers total; 11 bare `except:`, 321 `except Exception`.
- 49 are silent (`pass`, no log, no re-raise).
- 194 log nothing at all.
- Heaviest files: `execution/reconciliation.py` (60), `main.py` (21),
  `execution/mt5_direct.py` (19), `execution/post_entry/post_entry_manager.py` (11).

A raw count is not a severity score — most of these are legitimate RECOVER
patterns (JSON-repair fallback chains, best-effort Telegram notifications,
idempotent `ALTER TABLE` migrations that are supposed to fail once the column
exists). The scan's job was to surface candidates for manual reading, not to
be graded directly.

## FAIL-OPEN bugs found and fixed

### 1. `risk/risk_engine.py::can_trade` — MT5 duplicate-position check

Before:

```python
try:
    import MetaTrader5 as mt5
    positions = mt5.positions_get(symbol=symbol)
    if positions and len(positions) > 0:
        return False, "Duplicate symbol blocked via MT5"
except Exception as e:
    # Defensive requirement: allow trading if MT5 query fails.
    logger.error(f"Risk: MT5 duplicate check failed (allowing trade): {e}")
```

Two separate fail-open paths, both reachable in production:

- Any exception (raw, unlocked `import MetaTrader5` + `positions_get()`,
  outside the shared session lock, so it can race the post-entry loop,
  reconciliation and the watchdog on the same IPC channel) → explicitly
  logged as "allowing trade."
- `positions_get()` returning `None` — which the MT5 API documents as
  ambiguous between "zero results" and "an error occurred," distinguishable
  only via `last_error()` — was treated identically to "zero positions,"
  i.e. also allowed.

This is the one check that exists to stop a second position opening on a
symbol that already has one. Both paths meant it stopped working exactly when
MT5 connectivity was worst — which is also when a stale/duplicate signal is
most likely to fire.

**Fix:** both cases now reject the trade. The call is also routed through
`data.market.mt5_session.mt5_call()` (the H-01 shared lock) instead of a bare
`import MetaTrader5`, so it no longer races the other threads on the IPC
channel while it's at it.

Tests: `tests/test_risk_engine_fail_closed.py::TestDuplicatePositionCheckFailsClosed`
(5 tests — normal pass, real duplicate blocks, exception rejects, `None`
rejects, session-down rejects before calling, lock is taken).

### 2. `risk/risk_engine.py::can_trade` — Correlation Protection module

Before:

```python
try:
    from execution.risk_management.correlation_protection import is_correlated_open
    open_positions = get_open_trades() or []
    if is_correlated_open(...):
        return False, reason
except Exception:
    pass
return True, "OK"
```

`is_correlated_open()` already catches its own internal errors and returns
`False`. What reached this outer handler was therefore only what it *doesn't*
guard: a broken import, or `get_open_trades()` failing against a locked or
corrupt database — and both fell through to `return True, "OK"`. A database
failure during risk checking was silently becoming risk-check approval.

**Fix:** the outer handler now returns `False, "CorrelationProtection check
failed: <exc>"` instead of falling through.

Tests: `tests/test_risk_engine_fail_closed.py::TestCorrelationProtectionFailsClosed`
(4 tests — normal pass, broken import rejects, DB read failure rejects, a
real correlated position still blocks after the fix).

### 3. `main.py` — Risk Governor halt during an MT5 outage (visibility only)

```python
try:
    governor.halt("MT5 connection lost - candle boundary unavailable", ...)
except Exception:
    pass
return
```

Not fail-open in the trade-approval sense: the `return` immediately below
runs unconditionally (`current_candle_ts is None` is what triggered this
branch, and that's re-checked independently every cycle), so this specific
cycle stays blocked either way. What a swallowed failure hides is the halt
*not persisting* — an operator watching for a halt notification during a real
outage would see nothing. Now logs the exception instead of discarding it.

## What was reviewed and left as RECOVER (no change)

- **`execution/reconciliation.py`** (60 handlers, the largest file): every
  `pass`-only handler was read in context. All fall into: best-effort
  telemetry/notification writes that run *after* the money-moving action
  (`_apply_sltp_modification`, `close_trade_db_by_order_id`) has already
  happened; parsing fallbacks that leave a value at a safe `None` which
  downstream code already treats conservatively; or defensive loop
  continuations. None were found where a caught reconciliation error causes a
  position to be treated as safe/closed/protected when it isn't.
- **`execution/post_entry/event_bus.py::publish`** — wraps each listener call
  in `except Exception: pass`, explicitly commented `# fail-safe: do not
  break management loop`. Listeners are notification/analytics-only
  (Telegram, DB audit log, structured logger); the actual SL/TP/close actions
  run before the event is published, not inside a listener.
- **`risk/symbol_info.py`** (`get_symbol_point`, `get_symbol_stops_level`) —
  falls back to a static per-symbol table when MT5's `symbol_info` isn't
  available. Conservative RECOVER; the fallback table is deliberately
  present for this purpose.
- **`data/storage/database.py`** — the bare-`except: pass` handlers around
  `ALTER TABLE ... ADD COLUMN` are idempotent schema migrations, expected to
  raise once the column already exists on every run after the first.
  Legitimate RECOVER, though the bare `except:` (catches
  `KeyboardInterrupt`/`SystemExit` too, not just the expected
  `OperationalError`) is a style issue worth narrowing in a future pass — not
  a safety defect, so left out of this remediation's scope.
- **`analysis/ai/deepseek.py`** — JSON-repair fallback chain when the LLM
  response isn't clean JSON. The final failure path already returns a neutral
  advisory score rather than treating unparseable output as a signal; this
  was verified in an earlier remediation step, not re-litigated here.
- **Everything under `scripts/`** — offline tooling (historical import,
  dataset building, one-off training), not on the live trading path.

## Result

```
python -m pytest tests/ -q
569 passed, 1 failed  (the pre-existing, deliberately-deferred 13-vs-12
                       exit-model feature-count mismatch — see KNOWN_ISSUES.md
                       item 1 / ROADMAP.md; ML_EXIT_ENABLED stays False)
```

10 new tests added for this item (`tests/test_risk_engine_fail_closed.py`).
Combined with M-01/M-03/M-05 earlier in this same task
(`test_config_single_source.py` 36, `test_order_validation.py` 134,
`test_startup_script.py` 16) and H-05's integration suite
(`test_integration_execution_path.py` 19), this task added 215 tests, all
passing. Full suite: `588 passed, 1 failed` (the pre-existing deferred exit-
model mismatch).
