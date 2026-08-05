# Roadmap

Work that is deliberately **out of scope** for the trade-management rebuild and
must be planned as its own effort. Nothing here blocks that merge.

---

## 1. Re-enable the ML exit model

**Status: deferred indefinitely.** `ML_EXIT_ENABLED = False` and
`ML_EXIT_SHADOW_MODE = False` are a documented architectural constraint, not a
temporary defect. The Exit Score works without the model — see
`tests/test_exit_score_ml_disabled.py` — so there is no pressure to rush this.

Re-enabling depends on three things, in order. Skipping any of them produces a
model that memorises noise while appearing to work.

### Step 1 — Fix the entry-side data pipeline

The exit model's 12-feature schema is only as good as what the live path
records. Today 11 of the 12 features are constant across every usable row,
because the pipeline was writing fallback placeholders (see KNOWN_ISSUES.md 3).

Fields that must be populated at entry, in `main.py` around the
`upsert_execution_expected` call:

| Field | Current state | What is needed |
|---|---|---|
| `expected_trend_strength` | never written → `trend_h1`/`trend_h4` are always 0.0 | write the real MTF strength (it is already computed as `mtf.strength`) |
| `expected_adx` | column does not exist → `entry_adx` always 0.0 | add the column and populate it from the regime detector, which already computes ADX |
| `expected_spread` | constant 15.0 | capture the real spread at fill time from `mt5.symbol_info_tick` |
| `expected_rsi` | constant 50.0 | this is the `FALLBACK_INDICATORS` placeholder leaking through; requires the QuantDinger/MT5 data path to be healthy, not a code change |
| `expected_atr` | constant 0.0010 | same root cause — `FALLBACK_ATR` |

The last two are the important ones: they are not a schema gap but evidence that
the market-data path was degraded for the entire recorded period. Fix data
availability first, or every future row is as unusable as the current ones.

### Step 2 — Accumulate real excursion data

`mfe`/`mae` were never populated by the live path. The rebuilt post-entry
manager now persists them (in **dollars**, matching the historical column
contract — see KNOWN_ISSUES.md and `_update_excursions`).

These need to accumulate over genuinely live trades before they carry signal.
Do not backfill them from `actual_pnl`: the previous trainer's
`mfe = actual_pnl * 1.5` estimate manufactures a feature that is a linear
function of the label, which is the most direct way to build a model that scores
well and predicts nothing.

**Suggested minimum before retraining:** 300+ closed trades with non-zero,
varying `mfe`/`mae`, spanning more than one market regime and more than one
calendar week. The current 132 rows all share a single date.

### Step 3 — Retrain and validate

`scripts/train_exit_model_v2.py` is ready and trains against the canonical
12-feature schema through `feature_schema.build_feature_vector`, so the training
vector is by construction the vector inference builds.

It enforces its own gates and refuses to save otherwise:

- at least 100 samples, minority class at least 15
- a **time-ordered** split (never random — this is a time series)
- held-out AUC at least 0.60
- reports zero-variance features so a degenerate feature set is visible

Run: `python3 scripts/train_exit_model_v2.py`

### Step 4 — Shadow, then decide

Only once a model clears its gates:

1. Set `ML_EXIT_SHADOW_MODE=True` (leaving `ML_EXIT_ENABLED=False`). The
   probability is computed and logged with `[TM_L2_PROB][SHADOW]` and returns
   `influences_decision=False`, so it cannot affect a single decision.
2. Review the first batch of shadow output against realised outcomes.
3. `ML_EXIT_ENABLED=True` only on an explicit decision, never as a default.

When enabled, `probability` re-enters Exit Score at its designed 47.06% with no
config change — the weighted mean redistributes automatically.

### Step 5 — Resolve the artifact mismatch

`models/exit/exit_model.json` (13 features) is incompatible with the schema (12)
and cannot be reconciled — see KNOWN_ISSUES.md 2. A model produced by step 3
supersedes it. `tests/test_predict_proba_parity.py` will keep failing as a gate
until a schema-matching artifact exists; that is intentional.

---

## 2. Supply `exit_features` to the post-entry loop

`execution/post_entry/post_entry_manager.py` passes `exit_features=None` to the
orchestrator, so the probability provider is never called even in shadow mode.
Wiring it up is deferred until item 1 step 3 produces a usable model — building
the feature assembly now would only be exercised against a model that cannot
load.

---

## 3. Performance work identified during the initial review

Not started, listed so it is not lost. Ordered by measured impact.

| # | Item | Measured effect |
|---|---|---|
| 1 | Global lock around all MT5 calls | 726 watchdog disconnects and 198 `No IPC connection` errors from four threads sharing one IPC handle |
| 2 | Move migrations out of `get_conn`, pool connections | `get_conn()` costs 0.689 ms vs 0.012 ms reused — 57x |
| 3 | Drop `mt5.login()` from the per-fetch path | a broker round trip on every candle fetch |
| 4 | Read ATR from the cycle snapshot in `main.py` | three redundant HTTP calls per cycle |
| 5 | Negative caching + circuit breaker for QuantDinger | 1,629 repeated failures at a 10s timeout each |
| 6 | `requests.Session` + parallel RSS/DeepSeek/snapshot | cycle median 22s, p99 80s |

---

## 4. Train/serve consistency in indicators

Two defects found during the review, both affecting model input quality:

- **MACD**: the live path computes `SMA12 - SMA26` (`data/market/client.py:182`
  and `hybrid_client.py:182`), while the training pipeline computes a proper
  EMA-based MACD with signal and histogram
  (`analysis/entry_v2/feature_engineering.py:203`). The entry model is trained
  on one definition and served another.
- **RSI**: the primary source uses a simple 14-period average
  (`client.py:161`), the MT5 fallback uses Wilder smoothing over 100 candles
  (`hybrid_client.py:141`). The value jumps whenever the source switches, which
  the logs show happening 637 times.
