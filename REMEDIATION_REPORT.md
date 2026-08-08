# Kairos — Remediation Report

Closes the findings from the forensic audit that preceded this work. Baseline
recorded in `REMEDIATION_BASELINE.md` (branch `claude/drop-quantdinger`,
commit `211543d3086e1949c35aa95aa142d5f5d9ddda7d`, `1 failed, 217 passed`).
This branch (`claude/system-analysis-review-m2it05`) was restarted from
`origin/main` after `drop-quantdinger`'s PRs merged, then carries the
remediation forward — see "Branch note" below.

Current state: **`594 passed, 1 failed`** in `pytest tests/ -q` (595
collected). The one failure is the same pre-existing, deliberately-deferred
gate present in the baseline (see item 2 below) — everything the baseline had
green is still green, and 377 new tests were added.

---

## 1. Findings closed, with evidence

Each item: what was wrong, what changed, what proves it.

### C-01 — Entry model fed the wrong feature vector and predicted anyway

**Was:** `models/entry/entry_model.json` expects 65 features;
`predict_with_v2()` sent 10. `booster.predict()` silently zero-fills missing
slots and returns a number — no error, no rejection. A sensitivity sweep in
the baseline showed the model's output was insensitive to `direction`
(identical `p_win` for BUY and SELL), proof the prediction was not
meaningful.

**Fix:** `analysis/models/entry_feature_contract.py` — a single authoritative
gate (`validate_features`, `contract_from_booster`) checked before any
prediction is trusted: feature count → name/order → NaN/Inf/non-numeric →
probability range. Statuses `OK` / `ML_GATE_INVALID` / `ML_MODEL_MISSING` /
`ML_PREDICTION_ERROR`; `GateResult.allowed` requires both `OK` status and a
non-`None` `p_win`. Wired into both `analysis/models/xgboost_v2_inference.py`
(the default `ENTRY_MODEL_VERSION=v1` path) and `analysis/entry_v2/inference.py`
(`ENTRY_MODEL_VERSION=v2`).

**Evidence:** `tests/test_entry_ml_contract.py` (30 tests). **Operational
consequence — read item 0 in `KNOWN_ISSUES.md`:** the gate now correctly
refuses every live prediction, because the underlying mismatch was never
fixed at the data level, only safely contained. This is by design, not a
regression — see the dedicated section below.

### C-02 — Duplicate-order exposure on a lost broker reply

**Was:** `execution/mt5_direct.py`'s filling-mode retry loop `continue`d on
both an `order_send` exception and a `None` result, with no reconciliation
against the broker before resending — a lost reply could produce two live
positions for one signal.

**Fix:** `execution/order_idempotency.py` — deterministic per-signal `magic`
number (`SignalIdentity`, sha256-derived from symbol/direction/signal_ts),
`ExecutionRecord` state machine
(`NEW→SUBMITTING→UNKNOWN→RECONCILING→CONFIRMED_EXECUTED/NOT_EXECUTED`),
`find_position_for_signal()` (magic first, then symbol+side+recency within a
120s window), `resolve_unknown_outcome()`, `may_send_another_order()`. Any
ambiguous outcome reconciles against the broker before the loop is allowed to
try another filling mode; an unresolvable outcome (broker also unreachable)
refuses to retry rather than guess.

**Evidence:** `tests/test_order_idempotency.py` (24 tests),
`tests/test_integration_execution_path.py::TestDuplicateSignalSubmission` and
`::TestBrokerLevelOutcomes` (real retry-loop behaviour against a broker
double: rejection, timeout-with-no-execution, timeout-but-actually-executed,
ambiguous-and-unreachable).

### H-01 / M-02 — MT5 session ownership and DB concurrency

**Was:** four threads (main cycle, post-entry manager, reconciliation,
watchdog) called the `MetaTrader5` library directly and concurrently over a
single, non-thread-safe IPC channel (198 `No IPC connection` errors, 726
watchdog disconnects in the live logs). `reconciliation.py` additionally
called `mt5.shutdown()` then `mt5.initialize()` while other threads were
mid-call, manufacturing the very failures it was meant to repair. SQLite had
no busy_timeout, so a concurrent writer got an instant lock error instead of
a bounded wait.

**Fix:** `data/market/mt5_session.py` — single owner of session lifecycle
(`ensure_session`, `mt5_call()` reentrant lock, `is_healthy()` using a cheap
local `account_info()` read instead of a broker round trip). Every other
module now goes through it; the direct `mt5.shutdown()`/`mt5.initialize()`
calls in `reconciliation.py` and the `mt5.initialize(login=...)` in
`action_executor.py` / `mt5_direct.py` / `candle_boundary.py` were replaced.
`data/storage/database.py` sets `PRAGMA busy_timeout` (`DB_BUSY_TIMEOUT_SEC`)
and opens every connection with `timeout=`.

**Evidence:** `tests/test_mt5_access_policy.py` (static-analysis ratchet:
only `mt5_session.py` may call `initialize`/`login`/`shutdown`; hot loops
must take `mt5_call()`; a pinned, shrink-only budget on remaining direct
calls) + `TestDatabaseConcurrency` (5 concurrent writers, 50 rows, zero
errors).

### H-02 — Forming-candle look-ahead

**Was:** `get_candles()` could return the currently-forming candle, so
indicators computed on it changed value as the candle continued printing —
different numbers at decision time than at any later inspection.

**Fix:** `data/market/mt5_client.py::get_candles` requests `count + 1` and
drops the last (forming) bar; docstring states completed-candles-only.

**Evidence:** `tests/test_completed_candles.py` (14 tests).

### H-03 — Partial-TP and breakeven state lost on restart

**Was:** `partial_levels_done` lived only in in-memory `TradeRuntimeState`. A
restart while holding a position that had already taken its +2R partial came
back with an empty ladder and could take the same partial again.
`breakeven_done` had the mirror gap: read on startup, never written.

**Fix:** `data/storage/database.py` — new `partial_levels_done` /
`breakeven_done` columns, `update_partial_levels_done()`,
`update_breakeven_done()`, `parse_partial_levels_done()`, all with `get_conn()`
inside the `try` so a DB failure returns `False` instead of raising into the
management loop. `execution/post_entry/post_entry_manager.py::_ensure_state`
restores both on the first management pass after a restart; both are
persisted immediately after broker confirmation, with a loud error if the
write fails after a successful partial close.

**Evidence:** `tests/test_restart_recovery.py` (20 tests) +
`tests/test_integration_execution_path.py::TestRestartRecovery` (real DB
round-trip through the actual write and read paths together).

### H-04 — Entry gates existed but were never called

**Was:** `TradeManagementOrchestrator.check_entry_gate` (0 call sites),
`layer1_intrabar.can_open_new_entry` (0 call sites), and by extension
`RiskGovernor.can_open_new_position` (reachable only through the unwired
gate) — the governor's `MAX_OPEN_TRADES` ceiling never ran. `main.py` checked
`governor.is_halted()` once per *cycle*, and assembled a boolean
(`final_decision_valid`) from several unrelated conditions mixed together —
untestable in isolation and easy to extend wrongly.

**Fix:** `risk/trade_gate.py::validate_trade_request` — the single ALLOW/REJECT
decision point every entry now passes through, in order: signal validity →
numeric finiteness → SL/TP → size → risk engine → ML gate → Risk Governor
(halt + `can_open_new_position`). Fails closed on a governor it cannot reach.
`main.py` replaced its inline boolean with a call to this function.

**Evidence:** `tests/test_trade_gate.py` (57 tests, including
`TestGateOrdering::test_only_allow_can_reach_execution`) +
`tests/test_integration_execution_path.py::TestFullPipelineWiring` (proves
the gate's real output reaches/blocks the real `open_trade`, not a
hand-constructed stand-in).

### M-01 — Duplicate config definitions

**Was:** `ML_EXIT_ENABLED`, `ATR_SL_BASE_MULTIPLIER`, `ATR_TP_BASE_MULTIPLIER`,
`MAX_SL_PIPS` (and `ATR_SL_MULTIPLIER`/`ATR_TP_MULTIPLIER`) were defined in
both `config.py` and `trade_management/tm_config.py`. Nothing read the
`config.py` copies — every consumer resolved through `tm_config` — so the two
could silently disagree, and editing the `config.py` one did nothing.

**Fix:** removed from `config.py`, replaced with a comment naming
`trade_management/tm_config.py` as the single source of truth.
`tests/test_risk_sizing.py` updated to import `MAX_SL_PIPS` from there.

**Evidence:** `tests/test_config_single_source.py` — not just "the duplicate
is gone," but that changing the surviving value actually changes computed
output (`ATR_SL_BASE_MULTIPLIER` → stop distance, `MAX_SL_PIPS` → the
ceiling, `ML_EXIT_ENABLED` → resolved settings, a profile override beating
the module default, a `TM_*` environment variable reaching the constant).

### M-01 (continued) — `MT5_LOGIN` crash on an empty `.env`

**Was:** `int(os.getenv("MT5_LOGIN", "110609311"))` — `getenv`'s default only
applies when the variable is *absent*; a present-but-empty `MT5_LOGIN=` (the
normal result of copying `.env.example`, and the actual state of the `.env`
on disk at the time) produced `int("")` → a bare `ValueError` at import time,
in a module every entry point imports, before any logging existed.
`config.py` also shipped hardcoded fallback credentials — a live DeepSeek
key, a Telegram bot token, an MT5 login/password — as literal defaults.

**Fix:** `_env_str`/`_env_int`/`_env_float` helpers treat empty/whitespace as
absent and raise a message naming the variable on a genuinely malformed
value. Every credential default is now `""`/`0`. `mt5_session.ensure_session`
checks for incomplete credentials before attempting a login and names exactly
which variables are missing, rather than surfacing a generic MT5 login
failure.

**Evidence:** `tests/test_config_single_source.py::TestEmptyEnvironmentVariables`,
`::TestIncompleteCredentialsFailClosed` (12 tests, including one asserting
`mt5.login` is never called when credentials are incomplete).

### M-03 — NaN/Inf could reach order construction

**Was:** `mt5_direct.py`'s SL/TP safety check compared with `>=`/`<=`.
Comparisons against NaN are always `False`, so a NaN stop loss passed every
branch and reached `order_send` — the one input class that could not be
recovered from was the one that sailed through.

**Fix:** `execution/order_validation.py` — pure, broker-free validators
(`validate_order_inputs`, `validate_order_prices`, `validate_market_data`)
using positive assertions on values already proven finite, not negated
comparisons that silently pass on NaN. Checked at two points: `main.py`
before ATR/equity ever reach the sizing path, and `mt5_direct.open_trade`
before any session/symbol/broker call — both reject before touching MT5, not
after.

**Evidence:** `tests/test_order_validation.py` (134 tests, including
`TestTheOriginalDefect::test_nan_defeats_naive_comparisons` documenting *why*
the old shape failed, and `TestOpenTradeRejectsBeforeTouchingTheBroker`
proving the broker is never contacted on bad input).

### M-04 — Exception-handler audit (332 handlers classified)

Full writeup: `M04_EXCEPTION_AUDIT.md`. Two fail-open bugs found and fixed in
`risk/risk_engine.py::can_trade`:

1. **MT5 duplicate-position check** — used to say "allow trading if MT5 query
   fails" and treated `positions_get()` returning `None` (which the MT5 API
   documents as ambiguous between "no results" and "an error occurred") the
   same as "zero positions." Both paths let the one check that stops a second
   position opening on an already-open symbol silently stop working exactly
   when MT5 connectivity was worst. Now rejects on both, and the call moved
   from a raw `import MetaTrader5` to the shared session lock.
2. **Correlation-protection module** — its outer `except Exception: pass`
   turned a broken import or a locked/corrupt database into `return True,
   "OK"` (risk-check approval). Now rejects instead.

`main.py`'s `governor.halt()` swallow during an MT5 outage now logs the
exception instead of discarding it (visibility fix; the entry-blocking effect
was already correct regardless of `halt()`'s outcome).

The remaining ~320 handlers were read in context, not just counted: the
largest file, `execution/reconciliation.py` (60 handlers), was reviewed
handler-by-handler — every silent one is best-effort telemetry running after
the money-moving action already happened, a parsing fallback leaving a value
at a safe `None`, or a documented deliberate design choice
(`event_bus.publish`'s `# fail-safe: do not break management loop`).

**Evidence:** `tests/test_risk_engine_fail_closed.py` (10 tests).

### M-05 — `startup.bat` could not start the bot

**Was:** the script started Docker, brought up the QuantDinger compose stack,
and polled `http://localhost:8888/health` for 60 seconds before launching
`main.py` — exiting with an error if that never answered. QuantDinger no
longer exists, so the check could only ever fail; the step that launches the
bot was unreachable.

**Fix:** rewritten to check what the bot actually needs: `.env` present,
Python on PATH, MT5 terminal running (bounded wait, not an infinite `goto`
loop), and a live session verified via `ensure_session()`/`get_account_info()`
before launching `main.py`.

**Evidence:** `tests/test_startup_script.py` (16 tests: no executable
reference to Docker/QuantDinger/the dead health endpoint survives, nothing
can exit before `python main.py` runs for a reason unrelated to a real
prerequisite, the MT5 wait is bounded, `.env.example` documents every `MT5_*`
variable `config.py` actually reads).

### H-05 — Integration coverage across the full chain

Every layer above had unit coverage in isolation. `tests/test_integration_execution_path.py`
(19 tests) proves they work wired together the way `main.py` actually wires
them, using a broker double (`FakeMT5`) faithful to the real
`MetaTrader5` module's semantics — including a bare `None` from `order_send`
meaning "ambiguous," not "empty":

- **Full pipeline**: an ML-gate rejection, a below-threshold prediction, a
  halted governor, and a NaN in the request all provably never reach
  `order_send`; an ALLOWed request provably does.
- **Duplicate signal**: the same `signal_ts` resolves to the existing
  position via `find_position_for_signal`; a different `signal_ts` is
  genuinely new.
- **Broker-level outcomes**: rejection, timeout-with-no-execution,
  timeout-but-actually-executed (recovered without duplicating), and
  ambiguous-and-unreachable (refuses to retry) — all exercised through
  `open_trade`'s real retry loop, not a stand-in for it.
- **Reconciliation**: an orphan position (broker has it, DB never recorded
  it — the crash-between-`order_send`-and-DB-write case) still resolves by
  magic.
- **Restart**: partial/breakeven state round-trips through the real
  `database.py` write and `post_entry_manager.py` read.
- **Emergency close**: `ActionExecutor.close_position` — success,
  retry-then-succeed on a requote retcode, unknown ticket, non-numeric
  ticket.

---

## 2. What remains deliberately open

Documented in `KNOWN_ISSUES.md`; none of these are silently swallowed.

- **Item 0 (new this pass) — the entry model is currently gated shut.**
  `models/entry/entry_model.json` expects 65 features; neither live inference
  path (`v1`'s `predict_with_v2`, or `v2`'s `predict_with_entry_v2`, which its
  own docstring calls "a placeholder") currently builds more than 10. Verified
  empirically by calling the real production function directly — every call
  returns `ML_GATE_INVALID`, and `trade_gate` correctly rejects every entry as
  a result. **This is the C-01 gate working exactly as designed, not a
  regression from this remediation** — before C-01 the same mismatch existed
  and was silently ignored, producing a meaningless prediction instead of a
  clean refusal. The practical effect: the bot is safe to run (nothing opens
  on an unverified prediction) but will not open any trade until either the
  model is retrained on the 10-feature legacy vector, or the entry_v2 feature
  pipeline is finished to build the real 65 features live. Both are modeling
  work outside this remediation's scope (explicitly not touching entry-signal
  generation without separate sign-off). Pinned by
  `tests/test_entry_model_currently_gated.py` so this state cannot drift
  silently.
- **Item 2 — exit-model feature mismatch (13 vs 12).** Deferred indefinitely
  by explicit user decision; `ML_EXIT_ENABLED` stays `False`. This is the one
  test failure in the suite (`test_predict_proba_parity.py`), present in the
  baseline and unchanged.
- **Item 1 — `MAX_OPEN_TRADES` is unreachable** (a global open-position guard
  in `main.py` makes the bot single-trade regardless of the configured
  limit). Left untouched by explicit instruction — outside the
  trade-management/remediation scope, sits on the entry path.
- **Items 6–12** (duplicate `order_id` rows, two zero-stop-loss historical
  trades, `_feed_risk_governor`'s open-trades-only volume resolution, the RSI
  divide-by-zero on a flat series, `MAX_SL_PIPS` calibrated for forex not
  gold) — all pre-existing, documented with location and fix sketch, none
  reachable in a way that permits an unsafe live trade.

## 3. Other findings from this pass (static analysis + re-audit)

Not in the original audit list; found via `ruff check .` and a manual sweep
for remaining references to removed systems (QuantDinger, the pre-rewrite
trade-management generation, `news_shield`). Each is a real bug or genuine
dead weight, not style noise — the ~400 style-only ruff findings (unused
imports, `E402`, etc.) were left alone as out of scope for a safety
remediation.

- **`analysis/entry_v2/inference.py`** — `contract.model_missing(f"...{artifacts.model_path}")`
  referenced `artifacts`, a name that only exists inside a different function's
  local scope. A missing entry_v2 booster crashed with `NameError` instead of
  returning the clean `ML_MODEL_MISSING` result the ML gate depends on. Fixed
  to read the cached, correctly-scoped value. Covered by
  `tests/test_entry_v2_model_missing.py` (2 tests).
- **`analysis/models/xgboost_exit_model.py`** — `train_exit_model()` called
  `os.makedirs(MODELS_DIR, ...)`; the module defines `_MODELS_DIR`. Would have
  raised `NameError` the moment anyone ran the offline training script.
  One-line fix; not on the live path (`reconciliation.py` only calls
  `predict_exit_probability` from this module, never `train_exit_model`).
- **`analysis/entry_v2/entry_xgboost_trainer.py`** — used `datetime.utcnow()`
  and `timezone` without importing either. Orphaned offline training module
  (zero callers anywhere in the tree); fixed the import since it's a trivial,
  zero-risk correctness fix regardless.
- **`main.py`** — the `expected_payload` dict for `upsert_execution_expected`
  had `"expected_final_score"` as a key twice, with an identical expression
  both times. No behavioural bug (same value either way) but a real footgun —
  editing one copy silently does nothing if the other survives. Removed the
  redundant second occurrence.
- **`SETUP_COMPLETE.py`** (deleted) — an orphaned announcement script from an
  earlier feature (historical training pipeline), imported by nothing,
  referencing a `test_historical_setup.py` that doesn't exist, and starting
  with a stray non-ASCII character glued in front of its own shebang line —
  it would raise `NameError` on the first line if anyone ran it.
- **`core/exceptions.py`** (cleanup) — `QuantDingerError`,
  `QuantDingerAuthError`, `QuantDingerConnectionError` were never raised or
  caught anywhere (confirmed via exhaustive grep, including dynamic-reference
  patterns). Removed.
- **`diagnose_stale_candles.py`** (deleted, this branch only — see branch
  note) — imported `execution.quantdinger_client` and
  `config.QUANTDINGER_URL`, neither of which exist; caught by
  `test_config_integrity.py` after the branch restart below.
- **`execution/reconciliation.py`** — `_smart_profit_protection_step`,
  `check_profit_targets`, `check_news_conflict` (already explicitly commented
  `# UNUSED (legacy, kept for reference only) - not called anywhere in the
  codebase`, with a warning against wiring them in without a heartbeat check)
  contain two further `ruff`-flagged undefined names
  (`_plan_parse_cache`, `notify_alert`). Confirmed zero callers anywhere in
  the tree, including offline scripts. Left as-is: a prior pass already made
  the deliberate call to keep this as labelled dead reference code rather
  than delete it, and it is provably unreachable, so the bugs inside it have
  no runtime effect.

## Branch note

This branch (`claude/system-analysis-review-m2it05`) previously pointed at an
old commit (`58fd713`) that had already been fully merged into `main` (verified:
`git merge-base --is-ancestor origin/claude/system-analysis-review-m2it05
origin/main` → true). Per the standing instruction for that situation, it was
restarted from `origin/main` (`git checkout -B
claude/system-analysis-review-m2it05 origin/main`) and this remediation's work
— committed first on `claude/drop-quantdinger` as a safety checkpoint — was
cherry-picked on top (the README rewrite commit, then the full remediation
commit), plus one small gap between `main` and the prior branch state
(`diagnose_stale_candles.py` and its `.env.example` entry, both QuantDinger-era
and already dead — see above).

## 4. Ten safety properties, proven

1. **No trade opens on an unverified ML prediction.** `entry_feature_contract.py`
   validates before any prediction is trusted; `ML_GATE_INVALID`/`_MISSING`/`_ERROR`
   are all unconditional rejects in `trade_gate.py`. *(`test_entry_ml_contract.py`,
   `test_entry_model_currently_gated.py`)*
2. **No NaN/Inf reaches order construction.** Checked at market-data entry
   (`main.py`) and immediately before any broker call (`open_trade`), with
   positive assertions rather than negated comparisons. *(`test_order_validation.py`)*
3. **No duplicate order for one signal.** Deterministic magic number,
   broker reconciliation before any retry, refusal on an unresolvable
   ambiguous outcome. *(`test_order_idempotency.py`,
   `test_integration_execution_path.py::TestBrokerLevelOutcomes`)*
4. **No position opens without a protective stop.** `validate_order_inputs`
   rejects `sl_distance <= 0`; the SL/TP safety check rejects a stop on the
   wrong side of price using finite-value assertions. *(`test_order_validation.py`)*
5. **A Risk Governor halt stops all new entries.** `trade_gate` checks
   `gov.is_halted()` before every single entry, not once per cycle; a
   governor that cannot be reached is treated as blocking, not passing.
   *(`test_trade_gate.py::TestRiskGovernorGate`)*
6. **A duplicate-position check that cannot be verified rejects, not
   approves.** Both the MT5-duplicate check and the correlation-protection
   check in `risk_engine.can_trade` now fail closed on an internal error.
   *(`test_risk_engine_fail_closed.py`)*
7. **Only one module owns the MT5 session.** Static ratchet: `initialize`/
   `login`/`shutdown` outside `mt5_session.py` fail the test; hot loops must
   take the shared lock. *(`test_mt5_access_policy.py`)*
8. **No look-ahead from an incomplete candle.** `get_candles` always drops
   the forming bar. *(`test_completed_candles.py`)*
9. **Management state survives a restart without double-firing.**
   `partial_levels_done`/`breakeven_done` persist to the DB immediately on
   confirmation and are restored on the first pass after restart.
   *(`test_restart_recovery.py`, `test_integration_execution_path.py::TestRestartRecovery`)*
10. **A blank or malformed `.env` fails loudly and by name, not with a bare
    crash or a hardcoded fallback account.** Empty values are treated as
    absent; a malformed value names the offending variable; incomplete MT5
    credentials are refused before a login attempt. *(`test_config_single_source.py`)*

## 5. Test suite

```
$ pytest tests/ -q
594 passed, 1 failed in ~3.5s   (595 collected)
```

The 1 failure is `test_predict_proba_parity.py::test_predict_proba_matches_booster_predict`
— present in the baseline, unchanged, gating the exit model's 13-vs-12
feature mismatch (`ML_EXIT_ENABLED` stays `False` as a direct consequence,
exactly as intended).

Baseline was `1 failed, 217 passed`. **377 new tests added, all passing.**

`ruff check .` run for a static-analysis pass beyond the audit's original
scope (see section 3); real bugs it surfaced are fixed and tested above. `mypy`
run against every module created or rewritten this pass
(`risk/trade_gate.py`, `risk/risk_engine.py`, `execution/order_validation.py`,
`execution/order_idempotency.py`, `analysis/models/entry_feature_contract.py`)
— zero findings in any of them.

## 6. Verdict

**REMEDIATION COMPLETE.**

Every audit finding (C-01, C-02, H-01 through H-05, M-01 through M-05) is
closed with a targeted fix and a test that fails if the fix regresses. No
fallback in this codebase now permits trading when a subsystem it depends on
cannot be verified — that was checked explicitly, finding-by-finding, in
section 4 above. The exception-handler audit (M-04, 332 handlers) found and
closed the two remaining fail-open paths; everything else was read in
context and is either a correct RECOVER pattern or provably unreachable dead
code, not a silent safety gap.

This is not the same claim as "ready to trade live." It is not, for one
reason unrelated to safety: **the entry model currently rejects every signal**
(KNOWN_ISSUES.md item 0), because its 65-feature contract does not match
either live inference path's 10-feature vector. The gate built to catch
exactly this condition is doing its job — the bot will not size a trade off
an unverified prediction — but as a direct consequence it will not open any
trade until the model is retrained or the entry_v2 feature pipeline is
completed. That is real, necessary modeling work, explicitly out of this
remediation's scope, and the one condition standing between this branch and
functional (not just safe) live trading.
