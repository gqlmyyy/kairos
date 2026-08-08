# Known Issues

Deliberately-deferred defects. Each entry names the exact location, what goes
wrong, and why it was left alone. Nothing here is fixed by the trade-management
rebuild; all of it predates that work.

Planned follow-up work lives in `ROADMAP.md`.

---

## 0. THE ENTRY MODEL CURRENTLY GATES SHUT ON EVERY SIGNAL — the bot cannot open a trade

**Severity: blocks all trading. Read this before anything else in this file.**

**Verified empirically** (2026-08-08, remediation branch), by calling the real
production function directly:

```
$ python3 -c "
from analysis.models.xgboost_v2_inference import predict_with_v2
print(predict_with_v2(rsi=55.0, atr=0.0012, macd=0.0, trend_strength=50.0,
    trend_score=50.0, momentum_score=50.0, volatility_score=50.0,
    market_regime='trending', direction='BUY'))
"
[ML_GATE] ML_GATE_INVALID — entry BLOCKED. reason=feature count mismatch: model expects 65, got 10
{'p_win': None, 'available': False, 'status': 'ML_GATE_INVALID', 'reason': '...'}
```

Every call, for every symbol, every cycle, returns `ML_GATE_INVALID`. In
`risk/trade_gate.py`, `ml_available=False` is an unconditional REJECT. **No
signal can currently pass the ML gate, so no trade can currently open**, via
either entry path:

- `ENTRY_MODEL_VERSION=v1` (the default): `predict_with_v2()` in
  `analysis/models/xgboost_v2_inference.py` sends the 10 legacy scalar
  features (`LIVE_FEATURE_NAMES`), but `models/entry/entry_model.json` on disk
  expects 65 (`booster.num_features() == 65`).
- `ENTRY_MODEL_VERSION=v2`: `predict_with_entry_v2()` in
  `analysis/entry_v2/inference.py` sends the *same* 10 legacy scalars — its own
  docstring calls this "a placeholder until feature_schema v2 is implemented" —
  even though `analysis/entry_v2/feature_schema.py::FEATURE_COLUMNS` defines
  the full 65-feature schema the model file was almost certainly trained on
  (65 matches exactly). **No live code path currently builds that 65-feature
  vector from live market data — the schema exists, the model exists, nothing
  connects them at inference time.**

**Why this is not a regression from this remediation.** Before C-01, this same
mismatch existed and was *not* checked — `predict_with_v2` fed 10 values into a
65-feature model and returned a prediction anyway (`booster.predict()` silently
zero-fills missing feature slots). C-01 added the contract check
(`analysis/models/entry_feature_contract.py`) specifically to stop that: an
unverified prediction from a mismatched model must not size a live trade. The
gate is doing exactly its job. What C-01 does not do, and was never scoped to
do, is fix the mismatch itself — that requires either retraining
`entry_model.json` on the 10-feature legacy vector, or finishing the entry_v2
feature-engineering pipeline so it actually produces the 65 features live and
switching `ENTRY_MODEL_VERSION` to `v2`. Both are real modeling work, not a
remediation-scope code fix, and touch entry-signal generation — explicitly
out of scope for this remediation without separate sign-off.

**Before resuming live trading on this branch**, resolve one of:
1. Retrain `models/entry/entry_model.json` on the current 10-feature
   `LIVE_FEATURE_NAMES` vector (fast; matches what's actually available today), or
2. Finish `analysis/entry_v2/inference.py`'s feature builder to emit the real
   65 features from `feature_schema.FEATURE_COLUMNS` using live market data,
   and switch `ENTRY_MODEL_VERSION=v2`.

Until then, running the bot is safe (nothing opens on a bad prediction) but
non-functional (nothing opens at all).

---

## 1. `MAX_OPEN_TRADES = 3` is unreachable — the bot is effectively single-trade

**Location:** `main.py:425-449` (guard), `config.py:62` (setting)

**What happens:** `run_cycle` scans MT5 and the database for open positions and,
if it finds *any*, skips the entire cycle:

```python
if open_positions_found:
    logger.info("[GUARD] Open positions detected. Skipping all trading this cycle.")
    return
```

So a single open position blocks new entries on every symbol, and the system can
never hold more than one trade at a time.

**Why it matters:** three separate subsystems are built around a 3-trade limit
and are unreachable code as a result:

| Location | Logic |
|---|---|
| `risk/risk_engine.py:103` | `if open_count >= MAX_OPEN_TRADES` |
| `risk/position_sizing.py:148` | `elif open_trades >= MAX_OPEN_TRADES` |
| `risk/risk_governor.py:368` | `can_open_new_position` ceiling |

The correlation logic in `risk_engine.check_correlation` is also dead for the
same reason: it only triggers with two or more positions in a correlated group.

**Status:** left untouched by explicit instruction — it sits on the entry path,
outside the trade-management rebuild.

**If fixing:** decide whether the intent is genuinely one-trade-at-a-time (in
which case set `MAX_OPEN_TRADES = 1` and delete the three dead branches) or
multi-trade (in which case the guard should be per-symbol, not global, since
`risk_engine.can_trade` already blocks duplicate symbols via
`mt5.positions_get(symbol=symbol)`).

---

## 2. Exit model artifact does not match the feature schema

**Location:** `models/exit/exit_model.json` vs `analysis/models/feature_schema.py:9`

**What happens:** the deployed artifact expects **13** features; `FEATURE_ORDER`
defines **12**. The artifact carries `feature_names: []`, so the missing feature
cannot be named from the file. Git history is a single squashed commit and holds
no earlier feature list. Every feature-building path in the repository
(`feature_schema.build_feature_vector`, `train_exit_model.extract_features`)
produces 12.

**Consequence:** the exit model has never been usable. The previous adapter
asserted `feature_vec.shape[1] == bst.num_features()`, which always failed —
which is why `ML_EXIT_ENABLED` has been `False`.

**Status:** `tests/test_predict_proba_parity.py` fails by design as a gate, with
a message stating that `ML_EXIT_ENABLED` must stay `False`. Retraining is
blocked by issue 3.

---

## 3. Exit-model training data is degenerate

**Location:** `execution_dataset` table

**What happens:** of 507 rows, 132 have a realised P&L. Across those 132, **11 of
the 12 schema features have zero variance**:

| Feature | Constant value | Meaning |
|---|---|---|
| `entry_rsi` | 50.0 | `FALLBACK_INDICATORS` placeholder |
| `entry_atr` | 0.0010 | `FALLBACK_ATR` placeholder |
| `market_regime` | 4.0 | "unknown" |
| `session` | 3.0 | "unknown" |
| `spread` | 15.0 | constant |
| `mfe`, `mae` | 0.0 | never populated by the live path |
| `trend_h1`, `trend_h4` | 0.0 | `expected_trend_strength` never written |
| `entry_adx`, `volume` | 0.0 / 1.0 | never collected |

Only `trade_duration` varies. The rows were recorded while the data pipeline was
serving fallback constants, so they are not observations of the market.

The label is also purely temporal: ordered by time it runs 0% "bad exit" in the
first sixth of the data and 100% in the last two sixths, and all 132 rows share
one date (2026-07-17).

**Consequence:** an exit model cannot be trained until the pipeline records real
per-trade features. `scripts/train_exit_model_v2.py` refuses to save a model
that does not clear AUC 0.60 on a time-ordered held-out split.

**If fixing:** populate `expected_trend_strength`, `expected_adx` and a real
`expected_spread` at entry, and let the rebuilt `mfe`/`mae` persistence
accumulate over live trades before retraining.

---

## 4. Exit model probability is excluded, not defaulted — by design

**Location:** `trade_management/layer2_exit_score.py`, `tm_config.py`

**Not a defect**, recorded because the config is easy to misread. With the exit
model disabled, `EXIT_WEIGHT_PROBABILITY = 0.4706` looks like 47% of the score
is frozen at some constant. It is not: the scorer takes a weighted mean over
*available* components, so `probability` drops out entirely and its weight
redistributes.

Effective split today:

| Component | Nominal | Effective |
|---|---|---|
| `probability` | 0.4706 | excluded |
| `trend_reversal` | 0.2941 | **55.55%** |
| `momentum_weakness` | 0.2353 | **44.45%** |

The 0.75 threshold stays meaningful: neutral readings score 0.000, and strong
readings clear it short of the extremes (trend_score 10 with momentum_score 15
scores 0.756). Pinned by `tests/test_exit_score_ml_disabled.py`.

Re-enabling a valid model restores the 47.06% share with no config change.

---

## 5. `config.py` crashes on an empty `MT5_LOGIN` — FIXED

**Location:** `config.py`

Was:

```python
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "110609311"))
```

`os.getenv(name, default)` returns the default only when the variable is
*absent*. A present-but-empty `MT5_LOGIN=` yielded `int("")` and a
`ValueError` at import time, in a module every entry point imports, before any
logging existed. This was a live hazard: the `.env` on disk had `MT5_LOGIN=`
empty, so deploying it — the normal outcome of copying `.env.example` — took
the whole bot down.

**Fix (M-01/M-03 remediation pass):** every `config.py` read now goes through
`_env_str`/`_env_int`/`_env_float`, which treat empty/whitespace-only the same
as absent and fall back to the default; a genuinely malformed value (e.g.
`MT5_LOGIN=abc`) raises with the variable's name in the message instead of a
bare `ValueError`. The hardcoded credential fallbacks (a live DeepSeek key, a
Telegram bot token, an MT5 login/password) that used to ship as defaults in
this file are also gone — every credential now defaults to `""`/`0`, so a
missing `.env` fails loudly (`mt5_session.ensure_session` names exactly which
variables are missing) instead of silently authenticating against whatever
account happened to be hardcoded.

Tests: `tests/test_config_single_source.py` (`TestEmptyEnvironmentVariables`,
`TestIncompleteCredentialsFailClosed`).

---

## 6. Duplicate `order_id` rows in `trades`

**Location:** `data/storage/database.py` — `trades` has no unique constraint on
`order_id` (only `execution_dataset` does).

**What happens:** the same ticket appears multiple times. Observed in the current
database: `9465828541`, `9465545624`, `9465691004` and `9465892054` each have
three rows with differing `stop_loss` and `pnl`.

**Consequence:** `get_open_trades()` and any per-order lookup can return the
wrong row, and P&L aggregation double-counts.

---

## 7. Two trades recorded with `stop_loss = 0`

**Location:** rows `9519062040` and `9472909740` in `trades`

**What happens:** entries were persisted without a stop loss, so
`abs(entry - sl)` is meaningless and `risk_amount_usd` cannot be computed —
those trades contribute nothing to the Risk Governor's R accounting.

**Note:** this is data, not code, but it indicates a missing validation on the
write path.

---

## 8. `_feed_risk_governor` resolves volume from open trades only

**Location:** `execution/reconciliation.py:322-331`

**What happens:** trade size is looked up via `get_open_trades()`. This works
only while the DB row is still `open`; if the close is persisted before the
governor is fed, size resolves to `None` and `risk_amount_usd` silently becomes
`None`.

**Note:** the rebuilt post-entry path does not have this ordering dependency —
it reads volume from the live MT5 position
(`post_entry_manager._risk_amount_usd`). Only the reconciliation path is
affected.

---

## 9. RSI formula divides by zero on a perfectly flat series

**Location:** `data/market/mt5_client.py` `get_indicators()` (inherited verbatim
from the former `data/market/client.py`)

**What happens:** when every close in the window is identical, all 14
differences are zero. The classifier `(gains if diff > 0 else losses)` sends all
of them to `losses`, so `avg_gain` falls back to `0.001` while `avg_loss`
computes as a genuine `0.0`, and `rs = avg_gain / avg_loss` raises
`ZeroDivisionError`. The surrounding `except` catches it and the function
returns `FALLBACK_INDICATORS` — i.e. `rsi=50.0`, `atr=0.0008`.

**Why it matters:** this is exactly the stale-feed condition observed in
production, where ATR was byte-identical for hours. A frozen feed does not
merely repeat the last real reading; past a certain point it silently
substitutes fallback constants, and downstream `p_win` is then computed from
placeholders rather than market data.

**Why it was not fixed here:** the QuantDinger removal deliberately preserved
every indicator formula byte-for-byte so the entry model's input distribution
would not shift (see the module docstring). Changing the RSI is a separate,
deliberate change that needs retraining. Pinned by
`tests/test_mt5_client.py::TestIndicatorFormulas::test_perfectly_flat_series_falls_back`.

**If fixing:** guard the division (`avg_loss = max(avg_loss, 1e-9)`) and only
route a difference to `losses` when it is strictly negative, then retrain.

---

## 10. `MAX_CONSECUTIVE_LOSSES` was dropped from config, silently disabling the Risk Governor

**Location:** `config.py`, imported by `risk/risk_governor.py:30`

**What happened:** the trade-management rewrite removed the "Equity Guard"
config section as part of deleting the dead generation. That section also held
`MAX_CONSECUTIVE_LOSSES`, which the Risk Governor imports. The import therefore
raised `ImportError`, so `get_risk_governor()` failed on every call.

**Consequence:** `main.py` wraps the governor lookup in a bare `except`, so the
failure was silent — no `[RISK_GOVERNOR]` line appeared in any log, and the
cumulative-loss halt never armed. Live logs from 2026-08-06 confirm this: five
hours of cycles with no governor output at all.

**Status:** FIXED on the `claude/drop-quantdinger` branch by restoring the
constant. A guard now exists (`tests` sweep every `from config import ...` and
assert the name resolves) so a missing constant fails loudly instead.

**Lesson recorded:** removing a config section is not safe just because the
modules that defined its behaviour are dead — other modules may import
individual constants from it.

---

## 11. Stop-loss cap crushed stops to ~1% of ATR — FIXED

**Location:** `risk/symbol_info.py` `get_max_sl_distance()`, `risk/position_sizing.py`

**The incident (2026-08-07 18:59:56, live):** a XAUUSD position opened and was
stopped out within the same second.

```
XAUUSD ATR=47.35571  sl_mult=1.50  ->  stop should be 71.03
                                       stop actually  0.497   (1% of one ATR)
entry 4341.55  ->  SL 4341.053     TP 4495.456      R:R = 1:310
```

Every symbol was affected, all showing `capped=True`: EURUSD and GBPUSD both
had their stops cut to a single pip.

**Root cause — two defects that compounded:**

1. `get_max_sl_distance` shrank the stop so that trading `MAX_LOT` would risk
   under 5% of equity. It assumed 0.10 lots for XAUUSD while the actual order
   was 0.01 — ten times tighter than intended (fifty times for EURUSD, where
   `MAX_LOT` is 0.50). More fundamentally it derived stop *placement* from an
   assumed position *size*, reversing the correct order: ATR says where the
   trade is invalidated, and size is then chosen to fit the risk budget.

2. `effective_max = max(min_sl_from_stops, max_sl_from_pips, max_sl_from_atr)`
   combined ceilings with `max()`, so `MAX_SL_PIPS` — a setting whose name
   promises an upper bound — acted as a floor.

**Why it went unnoticed:** no trade had opened since the rebuild, so the stop
calculation had never reached a broker.

**The fix, in two parts.** Fixing only the stop would have been worse: with the
correct 1.5xATR stop, the risk-correct XAUUSD size on a $99.40 account is
0.000035 lots, and `position_sizing` used `max(MIN_LOT, size)` — silently
rounding up to 0.01 lots and risking **$71.03, i.e. 71.5% of the account**, on
one trade.

- `get_max_sl_distance` no longer applies an equity cap and combines ceilings
  with `min()`. Equity now controls size only.
- `calculate_position_size` treats the minimum lot as a rejection threshold,
  not a clamp. When the risk-correct size is below it, the minimum lot is taken
  only if that stays within `MAX_RISK_PER_TRADE_PCT` (new, 2%); otherwise the
  function returns 0.0 and `main.py`'s `position_size > 0` gate blocks entry.

Pinned by `tests/test_risk_sizing.py` (17 tests), including a sweep asserting
no accepted trade ever exceeds the hard ceiling at any account size.

**Consequence to be aware of:** a ~$99 account cannot trade XAUUSD at all with
a correct stop — the broker minimum lot alone risks most of it. The bot now
declines instead of taking the trade. That is the intended behaviour.

---

## 12. `MAX_SL_PIPS` is calibrated for forex, not for gold

**Location:** `config.py` `MAX_SL_PIPS = 100`

**What happens:** the cap is applied as `MAX_SL_PIPS * PIP_VALUES[symbol]`, and
the pip size differs by two orders of magnitude between instruments:

| Symbol | pip | cap in price units | typical ATR | cap as a share of ATR |
|---|---|---|---|---|
| EURUSD | 0.0001 | 0.0100 | 0.00175 | 5.7x ATR — never binds |
| XAUUSD | 0.1 | 10.0 | 47.36 | **0.21x ATR — always binds** |

So on gold the stop is capped at roughly a fifth of one ATR regardless of
market conditions, which still yields a lopsided 1:15 risk/reward and a stop
well inside normal noise.

**Not changed here:** this is a risk limit, and re-calibrating it is a trading
decision rather than a bug fix. Issue 11's rejection logic prevents the unsafe
trade either way.

**If fixing:** make the cap ATR-relative (e.g. `max_sl = k * ATR`) or set it
per-symbol, rather than a single pip count shared across instrument classes.
