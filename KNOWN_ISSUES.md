# Known Issues

Deliberately-deferred defects. Each entry names the exact location, what goes
wrong, and why it was left alone. Nothing here is fixed by the trade-management
rebuild; all of it predates that work.

Planned follow-up work lives in `ROADMAP.md`.

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

## 5. `config.py` crashes on an empty `MT5_LOGIN` — operational risk

**Location:** `config.py:30`

```python
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "110609311"))
```

**What happens:** `os.getenv(name, default)` returns the default only when the
variable is *absent*. A present-but-empty `MT5_LOGIN=` yields `int("")` and a
`ValueError` **at import time**.

**Why this is an operational risk, not a cosmetic bug:** `config.py` is imported
by every entry point in the project — `main.py`, every script, every test. An
empty value does not degrade one feature; it prevents the process from starting
at all, before any logging is configured, with a bare `ValueError` traceback that
does not name the variable.

This is a live hazard: the `.env` currently on disk has `MT5_LOGIN=` empty.
Deploying an empty or partially-filled `.env` — the normal outcome of copying
`.env.example` — takes the whole bot down. It was hit during verification of
this branch and had to be worked around with an environment override.

**The same pattern affects every numeric setting read this way**, so check for
others before fixing just this line.

**If fixing:** `int(os.getenv("MT5_LOGIN") or "110609311")`, or a small helper
that treats empty strings as absent for all numeric config reads.

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
