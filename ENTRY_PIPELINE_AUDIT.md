# Entry Pipeline — Forensic Audit (Phase 1 & 2)

Reproduce every number here with `python scripts/audit_entry_pipeline.py`.

## The question

The deployed model `models/entry/entry_model.json` expects **65** features. The
live path sends **10**. The ML gate therefore refuses every signal and the bot
cannot open a trade.

The brief was explicit that "65 wrong, 10 right" must not be assumed. It is not
what the evidence shows. The 65 are not a superset of the 10 — they do not
overlap by even one name — and the pipeline that produced them reads the future.

## Root cause in one sentence

**Four independent trainers write to the same filename `models/entry/entry_model.json`,
and the one that wrote the deployed artifact was trained on a dataset whose H4
features contain up to three hours of future price.**

---

# Phase 1 — Repository audit

## 1.1 Who writes `models/entry/entry_model.json`

Not one pipeline with a bug. Four separate pipelines, each believing it owns the
file:

| # | Writer | Features | Label | Reachable from `main.py` |
|---|---|---|---|---|
| 1 | `analysis/models/xgboost_trainer.py` (`DEFAULT_MODEL_PATH`) | 12 | `actual_pnl > 0` from the `execution_dataset` DB table | yes — imported, dormant (see 1.4) |
| 2 | `analysis/entry_v2/entry_xgboost_trainer.py` (`DEFAULT_OUTPUT_MODEL_PATH`) | **65** | TP/SL barrier on the entry_v2 dataset | no |
| 3 | `scripts/verify_entry_model_v2.py` | 65 | retrains + overwrites the production pointer | no |
| 4 | `scripts/train_entry_model.py` | 10 | TP/SL barrier on real MT5 candles | no — gated, never auto-installs |

Writer 2 is the origin of the deployed artifact. Its last line copies its own
output over the shared production pointer:

```python
DEFAULT_OUTPUT_MODEL_PATH = "models/entry/entry_model.json"
...
with open(entry_model_json_path, "rb") as src:
    with open(model_out_path, "wb") as dst:
        dst.write(src.read())
```

No schema check, no backup, no version guard. Nothing in the repository prevents
any of the four from clobbering the others' work.

## 1.2 Who reads it

| Reader | Features sent | Status |
|---|---|---|
| `analysis/models/xgboost_v2_inference.py` | 10, via `entry_feature_spec` | **the live path** (`ENTRY_MODEL_VERSION=v1`, the default) |
| `analysis/models/xgboost_inference.py` → `model_manager.py` | unchecked passthrough | imported, dormant |
| `analysis/entry_v2/inference.py` | 10 placeholder scalars | live when `ENTRY_MODEL_VERSION=v2`; reads `models/entry_v2/`, not this file |
| `analysis/entry_v2/entry_calibration.py` | 65 | not reachable |

The live reader and the writer of the deployed artifact have never agreed. The
gate is the only thing that noticed.

## 1.3 Model artifacts on disk

| Path | `num_feature` | Origin |
|---|---|---|
| `models/entry/entry_model.json` | **65** | writer 2 |
| `models/entry_v2/train_run_20260713_210843/entry_model.json` | 65 | writer 2 |
| `models_backup/entry_backup_20260717_041826/*.json` (9 files) | **12** | writer 1 |
| `models/tmp_xgb.json`, `models/test_xgb.json` | 12 | writer 1 scratch |
| `models/exit/exit_model.json` | 13 | exit pipeline (out of scope) |

None carries `feature_names` — all have an empty list — so XGBoost itself cannot
reject a wrong-schema vector. It silently treats absent columns as missing and
follows default branch directions. The count check in
`analysis/models/entry_feature_contract.py` is the only thing standing between a
10-vector and a fabricated probability.

A 10-feature entry model has never existed on disk. The history is 12 → 65.

## 1.4 The dormant auto-retrain path

`analysis/models/system_orchestrator.py` runs a daily thread that calls
`train_model_from_db(strict_mode=True)` and overwrites the production pointer
with a 12-feature model, then hot-reloads it. Its guard:

```python
def should_retrain(new_rows_count, last_train_ts, max_elapsed_hours=6.0, row_threshold=50):
    if new_rows_count >= row_threshold: return True
    if last_train_ts is None: return True
    return (time.time() - last_train_ts) / 3600.0 >= max_elapsed_hours
```

`new_rows_count` is set to the *total* row count, not a delta, so this returns
`True` on essentially every call.

**It does not fire today.** `start_daily_orchestrator_thread` is imported at
`main.py:323` and never called. But it is one line from firing, and if it fires
it silently replaces the entry model with a 12-feature one trained on
`actual_pnl` — bypassing the promotion gate entirely.

## 1.5 Duplicate and dead code

Six modules build an entry feature vector:

| Module | Features | Status |
|---|---|---|
| `analysis/models/entry_feature_spec.py` | 10 | **ACTIVE** — the live contract |
| `analysis/features/live_parity_features.py` | indicator primitives | **ACTIVE** — shared by live and training |
| `analysis/features/ml_dataset_builder.py` | 12 | LEGACY — feeds writer 1 only |
| `analysis/entry_v2/feature_engineering.py` | 65 | LEGACY — produced the deployed artifact |
| `analysis/features/entry_features.py` | 36 | research — this investigation, never deployed |
| `analysis/features/feature_builder.py` | trade-logging features | ACTIVE, unrelated to the entry model |

`analysis/entry_v2/` is 4,845 lines across 15 modules. `dry_run_pipeline.py`
raises `ImportError` on import — it imports four names (`TP_SL_Config`,
`Holding_Config`, `generate_labels_for_symbol_tf`, `compute_label_stats`) that
do not exist in `entry_labels.py`. It is dead code that has never run.

## 1.6 System map — the path that produced the deployed model

```
data.market.client.get_candles
  └─ entry_v2/candle_loader.py        dedupe + sort, per (symbol, timeframe)
     └─ entry_v2/dataset_builder.py   ✗ grid = H1 timestamps
        │                             ✗ attaches "latest H4 candle at or before t"
        │                                = the candle that OPENS at t, still forming
        └─ entry_v2/feature_engineering.py
           │                          ✗ H4 indicators computed over the H1 grid
           │                          ✗ entry_price = h4_ema_50, not the close
           │                          → 65 columns, no direction
           └─ entry_v2/entry_labels.py
              │                       ✗ no direction column → every row is a BUY
              │                       ✗ horizon 24 rows = 24h, not 24 H4 candles
              └─ entry_v2/entry_xgboost_trainer.py
                 │                    random-free chronological 80/10/10, Optuna 25 trials
                 │                    objective = validation logloss only
                 └─ models/entry/entry_model.json   ← overwrites the live pointer
```

And the live path it collides with:

```
mt5_client.get_indicators  →  live_parity_features  →  entry_feature_spec (10)
   →  xgboost_v2_inference.predict_with_v2
      →  entry_feature_contract.validate_features   →  ML_GATE_INVALID
```

---

# Phase 2 — Root cause

Four defects, each verified against the artifacts the pipeline itself wrote
(24,851 labelled rows; EURUSD 8,466 / GBPUSD 8,464 / XAUUSD 7,921).

## D1 — Look-ahead leakage (decisive)

`dataset_builder` builds its grid from H1 timestamps and attaches the *latest H4
candle at or before t*. Candle timestamps are **open** times, so the H4 candle
"at or before" an hourly timestamp is the one that opened at t and closes three
hours later.

| Measurement | EURUSD | GBPUSD | XAUUSD |
|---|---|---|---|
| H4 open == H1 open at the candle's first appearance | 100.00% | 100.00% | 100.00% |
| H4 high already covers hours t..t+3 | 98.40% | 98.54% | 90.14% |
| H4 low already covers hours t..t+3 | 98.12% | 97.99% | 91.96% |
| H4 close == H1 close at **t+3h** | 96.61% | 96.44% | 80.50% |

`h4_close` at decision time is, 96% of the time, literally the price three hours
in the future. Every one of the 17 `h4_*` features is contaminated, and so is
`entry_price`, which is derived from `h4_ema_50`.

The same convention applies to H1: the candle stamped `t` closes at `t+1h`, so
the 17 `h1_*` features carry an hour of future price.

This invalidates **every metric ever measured on this dataset**, including the
model's own reported test logloss.

## D2 — The barriers are built from an EMA, not a price

`feature_engineering.py:676`:

```python
"entry_price": float(h4["close"][i]) if "close" in h4 else float(h4["ema_50"][i]) ...
```

`h4` is the dict returned by `_compute_indicator_series_for_tf`, whose keys are
indicator names. There is no `"close"` key, so the guard is always false and
`entry_price` falls through to **`h4_ema_50`**.

Verified: `entry_price == h4_ema_50` in **100.0000%** of 24,851 rows.

Measured against the real close:

| | value |
|---|---|
| mean \|entry_price − h4_close\| | 11.27 |
| median | 0.0038 |
| **in units of ATR — mean** | **0.921** |
| **in units of ATR — p95** | **2.241** |

The barriers are SL 1.0 ATR and TP 1.5 ATR. The entry point is displaced by
roughly one ATR on average and more than two at p95 — the same size as, and
often larger than, the barriers themselves. The labels describe trades nobody
could have placed.

## D3 — The "H4" indicators are not H4 indicators

`_compute_tf("h4")` selects rows where `has_h4 == 1`. Because the builder
forward-fills, that is *every* row — so the indicators are computed over the H1
grid, where each H4 candle appears four times.

| Symbol | h4_close changes on | shipped `h4_rsi_14` vs RSI over the H1 grid | vs RSI over true H4 candles |
|---|---|---|---|
| EURUSD | 24.19% of rows | mean err **0.0000** | mean err **12.53** |
| GBPUSD | 25.14% | **0.0000** | **12.76** |
| XAUUSD | 26.92% | **0.0000** | **11.75** |

An exact match against the H1-grid series. `h4_rsi_14` has a 14-*hour* lookback,
not 14 H4 candles; `h4_sma_200` spans 50 H4 candles, not 200.

## D4 — The model cannot represent a SELL

There is no `direction` column in the engineered dataset, so
`entry_labels._direction_from_row` returns BUY for all 24,851 rows. The
65-feature schema contains no direction feature either.

The deployed model answers "will price rise?" — never "is this trade good?" —
while the live path asks it to score both BUY and SELL signals.

Consequences visible in the label distribution: 12,869 `sl_first`, 9,620
`tp_first`, 105 `tp_first_same_bar`, and **2,257 `fallback_neither`** — 9% of
rows where neither barrier was touched and the label was invented by "which
level is closer to the final close". Horizon is a further defect: `max_h4_candles
= 24` walks 24 rows of the H1 grid, i.e. **24 hours, not 96**.

## Feature comparison

| | live path | old trainer | deployed artifact |
|---|---|---|---|
| count | 10 | 12 | 65 |
| source | `entry_feature_spec` | `ml_dataset_builder` | `entry_v2/feature_schema` |
| naming | `rsi`, `atr`, … | `rsi`, `atr`, … | `h4_rsi_14`, `h1_atr_14`, … |
| overlap with live | — | 8 of 10 | **0 of 10** |
| has `direction` | yes | no | no |
| has `trend_score` | yes | no | no |
| extra | — | `spread`, `ai_score`, `sentiment_score`, `news_impact_score` | lags, deltas, interactions, `symbol_encoded` |
| available at decision time | yes | yes | **no — reads up to 3h ahead** |
| label | TP/SL barrier | `actual_pnl > 0` | TP/SL barrier on a leaked EMA |

Three schemas, three label definitions, one filename.

Beyond the leakage, several of the 65 are pathological on their own terms:
`atr_x_trend = h1_atr × h4_ema_50` multiplies a volatility by a raw price level;
`trend_x_session = h1_ema_50 × session_code` multiplies a raw price by a
categorical code 1/2/3; `symbol_encoded` lets a tree branch on which instrument
it is looking at; and the raw `ema_*`/`sma_*` levels pool a 1.10 instrument with
a 4330 one.

---

## What this changes

The previous investigation (`ENTRY_MODEL_INVESTIGATION.md`) concluded no edge
exists in H4/H1 indicator features on a *correctly built* dataset. That still
stands — it was measured with completed candles only, with leakage tests, and
its verdict was to reject.

This audit is about the artifact that is actually deployed, and it is worse than
"trained on different features". It was trained on data that reads the future,
labelled against a price that was never traded, in one direction only. Its
reported test score is meaningless. **It must never be promoted, and the ML gate
refusing it is the correct outcome — not an obstacle to work around.**

The 65-vs-10 mismatch is a symptom. The disease is that four trainers share one
filename with no schema contract, no provenance, and no gate between them.

## Status against the 12-phase plan

Phase 1 and Phase 2 are complete. Phase 3 (repair) has not started — no
production code has been modified by this audit. The brief's rule that a phase
must not proceed on an invalid predecessor applies directly here: the entry_v2
dataset is invalid, so no amount of Optuna tuning on it is meaningful.

Open decisions for Phase 3, in the order they matter:

1. **Close the shared-filename hole.** A model file needs `feature_names`
   embedded and a metadata sidecar (schema, provenance, git commit); the loader
   should fail closed on mismatch — which it now does by count, but not by name.
2. **Decide the fate of `analysis/entry_v2/`.** 4,845 lines, one dead module,
   and a dataset builder with proven look-ahead. Repair or remove — leaving it
   importable is what let it overwrite production once already.
3. **Neutralise the dormant auto-retrain.** It is one uncommented line from
   overwriting the entry model with a 12-feature artifact.
4. Only then: whether any labelling/feature framing is worth optimising, which
   the previous investigation suggests is doubtful for this timeframe.
