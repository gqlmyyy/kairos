# Known Issues

Deliberately-deferred defects. Each entry names the exact location, what goes
wrong, and why it was left alone. Nothing here is fixed by the trade-management
rebuild; all of it predates that work.

Planned follow-up work lives in `ROADMAP.md`.

The **research entry models** from the xgbooost project are integrated on a
separate, contract-checked path — see `RESEARCH_MODEL_INTEGRATION.md`. That
work does not fix item 0 below: it neither retrains nor replaces
`models/entry/entry_model.json`, and every research model it imports was
rejected by the research repository's own gates (8 `NO_SIGNAL`, 1 `DROP`).
It is offline validation only; nothing is wired to live trading.

---

## 0. THE ENTRY MODEL STILL GATES SHUT — the bot cannot open a trade

**Severity: blocks all trading. Read this before anything else in this file.**

`models/entry/entry_model.json` expects **65** features; the live path sends
**10**. `risk/trade_gate.py` treats `ml_available=False` as an unconditional
REJECT, so every signal, on every symbol, every cycle, is refused:

```
[ML_GATE] ML_GATE_INVALID — feature count mismatch: model expects 65, got 10
[TRADE_GATE] REJECT — ...
```

This is the C-01 contract check working as designed — before it existed, the
same mismatch was *unchecked* and `booster.predict()` silently zero-filled the
55 absent slots, returning a probability unrelated to the trade (BUY and SELL
scored identically). Blocking is correct; it is not the cure.

**What has been built to fix it** (the model itself is deliberately unchanged):

- `analysis/models/entry_feature_spec.py` — the single source of truth for the
  ten features: names, order, encodings, defaults. Live inference imports its
  vector builder from here, so a second divergent list cannot exist.
- `analysis/features/live_parity_features.py` — recomputes indicators with the
  *live* arithmetic so a training row and a live call agree value-for-value.
- `scripts/fetch_training_candles.py` — captures real OHLC from MT5 (Windows).
- `scripts/train_entry_model.py` — labels both BUY and SELL, picks the horizon
  from the measured resolution curve, validates the dataset, runs walk-forward
  validation, compares against baselines, and refuses to install a model that
  fails any check. It also **refuses to run at all** on candles without a fetch
  manifest, so a synthetic-data model cannot reach production by accident.

**The remaining blocker is data, not code.** Training needs real OHLC for
EURUSD/GBPUSD/XAUUSD across H4 and H1. The existing
`data/entry_v2/labeled_dataset.parquet` was evaluated and **rejected**: it has
no `direction` column, so all 24,851 rows were labelled BUY — a model trained on
it could not distinguish BUY from SELL, reproducing the exact defect being
fixed. Its indicators also use standard Wilder/EMA formulas while live uses
simple-average RSI and SMA-based MACD (measured skew: RSI 6.6 points mean,
MACD 3.5x scale).

**To finish, on the Windows machine with MT5 running:**

```
python scripts/fetch_training_candles.py
python scripts/train_entry_model.py --dry-run    # validate only
python scripts/train_entry_model.py              # train and install
```

The trainer backs up the current model first and installs only after the
feature-contract, walk-forward and live-inference checks all pass.

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

## 9. RSI divided by zero on a flat series — FIXED

**Location:** `analysis/features/live_parity_features.py::live_rsi`, now the
single implementation, imported by `data/market/mt5_client.get_indicators`.

**What happened:** when every close in the window is identical, all 14
differences are zero. `(gains if diff > 0 else losses)` sends an exact `0.0` to
`losses`, so the list was **non-empty but summed to zero**. The guard read
`if losses` — "is the list empty?" — when the question was whether its average
was zero. `avg_loss` became `0.0` and `rs = avg_gain / avg_loss` raised
`ZeroDivisionError`.

**Why it mattered more than it looked.** In live the exception was caught by
`get_indicators`' `except` and the function returned `FALLBACK_INDICATORS`:
`rsi=50.0`, `atr=0.0008`. That is the origin of the constant `entry_rsi = 50.0`
and `entry_atr = 0.0010` values that make the recorded `execution_dataset`
degenerate (issue 3) — the pipeline was not recording the market, it was
recording its own fallback table. Flat stretches are ordinary in real data:
weekends, holidays, thin sessions, frozen feeds.

It surfaced as a hard crash only when the training pipeline called the same
formula without a catch-all around it, on real MT5 H1 candles.

**Fix:** apply the epsilon the original author intended to a zero *average*
rather than an empty list:

```python
avg_gain = (sum(gains) / 14) or 0.001
avg_loss = (sum(losses) / 14) or 0.001
```

A flat window now scores RSI 50 (rs = 1 — price did not move), an all-up window
~100 and an all-down window ~0. Inputs that did not previously raise are
bit-identical, so the feature distribution does not shift.

**Also fixed here:** `mt5_client` no longer keeps its own copy of the RSI, ATR,
MACD, ma_trend and volatility arithmetic. It imports them from
`live_parity_features`, the same module the training pipeline uses, so the two
cannot drift apart — the failure mode behind the original 65-vs-10 mismatch.

Tests: `tests/test_entry_feature_parity.py::TestDegenerateRealWorldInput` (11),
`::TestSingleIndicatorImplementation`, and
`tests/test_mt5_client.py::TestIndicatorFormulas::test_perfectly_flat_series_is_computed_not_faked`.

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

---

## 13. Three of the entry model's ten features were constants — FIXED

Found while rebuilding the entry-model training pipeline. Each was verified by
calling the live functions directly, before and after.

| Feature | Was | Cause | Fix |
|---|---|---|---|
| `trend_strength` | always `0.0` | `main.py` passed `mtf.strength if isinstance(mtf.strength, (int, float)) else 0.0`, but `MultiTimeframeData.strength` is a *string* (`"weak"`/`"moderate"`/`"strong"`), so the guard never passed | `entry_feature_spec.encode_trend_strength` maps the string via `config.TREND_STRENGTH_VALUES` (25/60/100). Unknown maps to `0.0`, kept distinct from a genuine "weak" |
| `volatility_score` | always `55.0` | `get_volatility_score_from_snapshot` reads a `volatility` key that `get_indicators` never emitted, so every lookup fell to the final `else` | `get_indicators` now emits it, bucketed from **current ATR / this symbol's median ATR over the window** |
| `market_regime` | always `TRENDING` (`1.0`) | `ma_trend` returned `"sideways"` only when `price == ma20` *exactly* (0 hits in 200,000 draws), so the H4 trend direction was never `"neutral"` | price within `config.MA_TREND_FLAT_ATR_MULT` (0.25) ATRs of MA20 is now flat, making RANGING reachable; with volatility live, HIGH/LOW_VOLATILITY are reachable too |

**Why the volatility measure is self-relative, not an ATR percentage.** The first
attempt bucketed `atr / price` against fixed cuts. Measured on live data, EURUSD
H4 runs at ~0.14% of price and XAUUSD at ~0.96% — an order of magnitude apart —
so any fixed cut pins each symbol to one bucket permanently, which is the same
frozen-feature bug in a new place. The dataset validator caught it (reported
`volatility_score` as an unexpected constant) before it reached a model. The
measure is now `ATR_now / median(ATR over the 100-candle window)`, which is
scale-free; verified to give the same bucket for identical relative volatility
at prices 1.10, 1.27 and 4330.

**Side effects of the regime fix, beyond the model:** Layer 6 can now select the
`mean_reversion` and `range` profiles, and Layer 1 no longer applies the trending
SL/TP factors unconditionally. Layer 6's `classify_entry` also received the same
string-vs-number fix — it compares `trend_strength` against `TRAILING_TREND_HIGH`
and was being passed `None`, so that branch never fired.

Training reproduces all three through the same code
(`analysis/features/live_parity_features.py` mirrors `get_indicators`, and both
sides call the spec's encoders), so there is one calibration rather than two.
The two `_atr_ratio` implementations are asserted equal to within 1e-12.

Pinned by `tests/test_entry_feature_parity.py::TestPreviouslyConstantFeaturesAreNowDynamic`
(20 tests).

---

## 14. A failed model load poisoned the inference cache until restart — FIXED

**Location:** `analysis/models/xgboost_v2_inference.py::load_v2_model`

```python
_model = xgb.Booster()      # global assigned first
_model.load_model(MODEL_PATH)   # ...and this is what raises
```

**What happened:** on any load failure — truncated file, partial write, a read
racing a replacement — the function correctly returned `None`, but the global
was already holding an empty `Booster`. Every later call took the
`if _model is not None` fast path and returned that empty object, whose
`num_features()` raises. The gate then reported

```
ML_GATE_INVALID — model feature count could not be determined
```

for the rest of the process: a *transient* file problem became permanent, the
error blamed the feature contract rather than the load, and the process never
recovered even after a good model was put in place. Only a restart cleared it.

**Fix:** load into a local and publish to the cache only on success.

Tests: `tests/test_entry_model_loader.py` (8 tests, including recovery after a
failed load without a restart).
