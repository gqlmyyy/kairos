# Phase 3 — Data, Model Supply Chain and Safety Repair

No model was trained. No Optuna. No AUC was optimised. The goal was to make the
system capable of learning something trustworthy, and to make it impossible to
repeat the failure that produced the deployed artifact.

`models/entry/entry_model.json` is byte-identical throughout: sha256
`ecbfc94bfb7f625b310f176ffe2de5dfd29dd19aeb0b9f671e090ea4daa7d8cc`.

---

## 1. Root causes fixed

| # | Root cause | Repair |
|---|---|---|
| 1 | Four trainers wrote to one filename with three schemas and no arbitration | `production_model_guard` is the only sanctioned writer; the other three now write research artifacts and are blocked from the production path |
| 2 | A model file declared nothing about itself (`feature_names` empty in every artifact) | Required metadata sidecar; the loader refuses anything that cannot prove its identity |
| 3 | Validation was by feature *count* — ten features are not necessarily the same ten | Validation by name **and order**, plus feature/label schema versions |
| 4 | A daily thread could retrain and overwrite production, with a trigger that was constantly true | Off unless `KAIROS_ENABLE_AUTO_RETRAIN=1`; never reloads production; `should_retrain` is a decision again |
| 5 | Look-ahead: the H4 candle attached at decision time was still forming | `timeframe_alignment` — a candle is knowable only from `open + duration`; every slice goes through `closed_slice` |
| 6 | Decision timestamp treated as the bar's *open* | Decision time is the bar's **close**, the first moment its indicators exist |
| 7 | Entry price fell through an always-false guard to `h4_ema_50` (100% of rows) | Entry is the **open of the next bar** — the first obtainable price |
| 8 | "H4" indicators computed over a forward-filled H1 grid | Indicators computed on real H4 candles; regression test compares against a true-H4 computation |
| 9 | No direction column; every row labelled BUY | Each decision bar yields a BUY row and a SELL row, labelled independently, with `direction` in the vector |
| 10 | Three of four market regimes encoded to the same number | Four regimes, four values, and a distinct code for "unknown" |
| 11 | **The leakage detector was itself broken** | Rewritten tie-corrected — see below |

### The leakage detector was broken

Promoting the single-feature-AUC probe to a blocker exposed a bug in the probe.
It ranked with `sorted(zip(col, y))`, which breaks ties on the second element,
so within every group of equal feature values the `y=1` rows sorted last and
took the highest ranks. Measured on a clean dataset:

| feature | distinct values | true AUC | probe's AUC |
|---|---|---|---|
| `volatility_score` | 4 | 0.5088 | **0.9068** |
| `market_regime` | 2 | 0.5014 | **0.9083** |
| `direction` | 2 | 0.5067 | 0.7432 |
| `session` | 3 | 0.5035 | 0.6702 |
| `rsi` | 3,132 | 0.5084 | 0.5085 |

It lies only on tied columns — that is, on exactly the encoded categoricals,
and on all of them. As a report nobody acted on it was harmless; as a blocker
it would have rejected every honest dataset while catching nothing. Now uses
the same tie-corrected Mann-Whitney routine as the metrics, verified against
constant → 0.5 and perfect → 1.0.

---

## 2. Legacy systems disabled or archived

Nothing was deleted. The dependency graph was mapped first: `main.py` imported
only `analysis/entry_v2/inference.py`, via `ENTRY_MODEL_VERSION=v2`; no
scheduler, cron or service file referenced the package at all.

| Component | Status | Action |
|---|---|---|
| `analysis/entry_v2/{dataset_builder,feature_engineering,entry_labels,entry_xgboost_trainer}` | LEGACY / INVALIDATED | Entrypoints refuse to run; override needs `KAIROS_ALLOW_INVALIDATED_ENTRY_V2=1` |
| `analysis/entry_v2/inference.py` | Removed from the execution path | `ENTRY_MODEL_VERSION=v2` rejected at config load |
| `analysis/entry_v2/dry_run_pipeline.py` | DEAD | Raises `ImportError` — imports four names that do not exist. Left in place, documented |
| `analysis/models/xgboost_trainer.py` (12-feature) | LEGACY | Writes to `models/entry/research/legacy_execution_dataset/` |
| `scripts/verify_entry_model_v2.py` | LEGACY | Writes to `models/entry/research/legacy_verify_v2/` |
| `analysis/models/system_orchestrator.py` | DORMANT → explicitly disabled | Opt-in only; cannot touch production |

The package stays importable on purpose: quarantine must not mean "cannot be
audited", and `scripts/audit_entry_pipeline.py` still reads its output.

---

## 3. Production safety — who can write the model now

Exactly one path:

```
scripts/train_entry_model.py
  → gates: provenance manifest, dataset validation, walk-forward ≥ --min-auc
  → production_model_guard.install(candidate, metadata)
       requires KAIROS_ALLOW_MODEL_INSTALL=1
       re-validates candidate vs its metadata and vs the live contract
       backs up what it replaces, verifies the copy by checksum
  → models/entry/entry_model.json + .metadata.json
```

Everything else calls `assert_not_production()` and raises. The opt-in is an
environment variable rather than a function argument because an argument can be
passed by a scheduler or a retry loop; an environment variable has to be set by
a person.

---

## 4. Feature contract

`analysis/models/entry_feature_spec.FEATURE_CONTRACT` — one machine-readable
entry per feature carrying dtype, source, timeframe, required history,
calculation, availability, missing-value behaviour, encoding, valid range, and
both training and live usage. `validate_contract()` runs at import and fails if
the contract and the deployed vector disagree in membership or order.

| # | feature | tf | encoding | classification |
|---|---|---|---|---|
| 1 | `rsi` | H1 | none | VALID |
| 2 | `atr` | H4 | none | **DEGRADED** — price-scale dependent (EURUSD ~0.0017 vs XAUUSD ~41.7) |
| 3 | `macd` | H1 | none | **DEGRADED** — same scale problem |
| 4 | `trend_strength` | H4+H1 | 3 values | **PARITY GAP** — live uses H4/H1/M15, training substitutes H1 for M15 |
| 5 | `trend_score` | H4 | none | VALID |
| 6 | `momentum_score` | H1 | none | **REDUNDANT** — a three-bucket re-encoding of `rsi` |
| 7 | `volatility_score` | H1 | none | VALID (self-relative since the earlier fix) |
| 8 | `market_regime` | H4+H1 | 4 values | VALID (was collapsing 3 of 4 states) |
| 9 | `session` | clock | 3 values | VALID |
| 10 | `direction` | n/a | 2 values | VALID |

None are LEAKED or UNAVAILABLE_LIVE. Every entry declares availability at the
decision timestamp, and a test asserts it.

The four flagged entries are recorded, not silently fixed. Scale-free
replacements for `atr`/`macd` already exist in
`analysis/features/entry_features.py` and were measured in the previous
investigation; adopting them is a feature decision that belongs in Phase 4 with
evidence, not a Phase 3 repair. `market_regime` was fixed because collapsing
three distinct states into one number is provable information destruction
independent of any score.

---

## 5. Dataset

Rebuilt as code, **not yet run on real candles** — `data/historical/` is not
present in this environment and requires a live MT5 terminal on Windows.

Verified end-to-end on synthetic candles (3 symbols, 1,600 H4 + 6,400 H1 bars
each): 8,010 rows, BUY 4,044 / SELL 3,966, win rate 34.4%, zero non-finite
values, zero duplicates, zero constant features, all gates passing, and the
promotion gate correctly refusing at walk-forward ROC-AUC 0.4525 — which is the
right answer, since the synthetic series contains no signal to find.

- **Time range / symbols / timeframes**: whatever `fetch_training_candles.py`
  fetches; scope is recorded in the model metadata rather than assumed. The
  standing configuration is EURUSD, GBPUSD, XAUUSD on H4 (decision) + H1.
- **Fingerprint**: `dataset_fingerprint()` hashes each row individually and
  sorts the digests, so the same data in a different order fingerprints
  identically while a single changed value does not.

### Entry price

The **open of the bar following the decision bar**. The decision bar's close is
what told us to trade and is history by the time we know it; the next open is
the first price a market order could have received.

Spread and commission are **not** modelled — no historical spread series
exists. Both make the target harder, never easier, so the labelling is
optimistic by a known sign. This is a remaining risk, not a silent assumption.

### Label

```
decision at close of bar i
  → fill at open of bar i+1
  → walk bars i+1 .. i+horizon
  → first touch of TP (1.5×ATR) → 1
    first touch of SL (1.0×ATR) → 0
    both inside one bar         → 0   (order unknowable without ticks)
    neither                     → row dropped
```

Unresolved trades are **dropped, not guessed**. entry_v2 invented a label for
9% of its rows by asking whether the final close sat nearer TP or SL, which is
a different question from the one the model is asked.

### Direction

Design **A** — `features + direction → P(success)` — one model, each decision
bar producing an independently labelled BUY row and SELL row.

Chosen over **B** (separate BUY/SELL models) and **C** (direction prediction +
outcome prediction) for reasons that hold now and are testable later:

- **B** halves the data per model on a dataset already short of independent
  observations (adjacent H4 bars share ~96 of their 100-bar indicator window),
  and forbids the model from learning anything symmetric across directions.
- **C** answers a different question. The entry model is a *filter* on a signal
  the rule-based layer has already produced with a direction; predicting the
  direction duplicates that layer instead of gating it.
- **A** keeps every row and lets the model discover direction-specific
  behaviour if it exists, since `direction` is a real input.

A test asserts that paired BUY/SELL rows are not forced opposites — both can
lose, because TP and SL are asymmetric. If they were always opposite, the label
would be a direction predictor rather than a trade-quality one.

This is the reasoned default, not a settled result. The A-vs-B experiment needs
valid data first, which is exactly the sequencing the brief specifies.

---

## 6. Leakage test results

| Test | Result |
|---|---|
| Mutating every candle after the decision (×7) | **0 of 836** feature vectors changed |
| Mutating the past (non-vacuity control) | **835 of 835** changed |
| Candle knowable at its own open | Refused, all of M15/M30/H1/H4/D1 |
| Candle knowable exactly at its close | Accepted, all timeframes |
| H4 decision sees four closed H1 bars | Confirmed |
| D1 bar visible mid-day | Refused |
| Cross-timeframe pairs (M15/H1, M30/H4, H1/H4, H4/D1) | No leak in any pairing |
| Shipped H4 indicator vs true-H4 computation | Identical to 1e-12 |
| Oversampled-grid control | Differs by >1 RSI point, so the check is not vacuous |

The alignment rule is enforced for every timeframe, not just H4. The audit found
the bug in H4, but nothing about it was H4-specific — the same forward-fill
would leak a day from D1.

---

## 7. Tests

**877 total: 876 passed, 1 failed.**

The failure is `test_predict_proba_parity.py::test_predict_proba_matches_booster_predict`
— the exit model expects 13 features while its schema defines 12. Pre-existing,
unrelated to the entry pipeline, documented as KNOWN_ISSUES item 2, and left
alone deliberately: it is the exit model's own contract mismatch and fixing it
inside a phase about the entry pipeline would hide it.

New coverage, mapped to the brief's required areas:

| Area | File |
|---|---|
| H4/H1/D1/M15/M30 closed-candle alignment | `test_timeframe_alignment.py` (48) |
| Indicator timeframe correctness | `test_entry_dataset_semantics.py` |
| Entry price | `test_entry_dataset_semantics.py` |
| Label generation | `test_entry_dataset_semantics.py` (21 total) |
| No-lookahead | `test_entry_dataset_semantics.py`, `test_training_no_leakage.py` |
| Feature schema and order | `test_feature_contract.py` (15) |
| Model metadata / loading failure | `test_entry_model_loader.py` (15) |
| Production overwrite protection | `test_production_model_safety.py` (26) |
| Dormant retraining disabled | `test_production_model_safety.py` |
| Legacy trainer isolation | `test_production_model_safety.py` |
| Dataset construction gate | `test_dataset_validation_gate.py` (15) |

---

## 8. Files

**Added**

```
analysis/models/entry_model_metadata.py       provenance contract
analysis/models/production_model_guard.py     the only sanctioned writer
analysis/features/timeframe_alignment.py      knowability + alignment
scripts/audit_entry_pipeline.py               reproduces the Phase 1/2 evidence
ENTRY_PIPELINE_AUDIT.md                       Phase 1/2 findings
PHASE3_REPORT.md                              this file
tests/test_timeframe_alignment.py
tests/test_entry_dataset_semantics.py
tests/test_dataset_validation_gate.py
tests/test_feature_contract.py
tests/test_production_model_safety.py
```

**Changed**

```
analysis/models/xgboost_v2_inference.py       provenance gate before serving
analysis/models/entry_feature_spec.py         FEATURE_CONTRACT; 4-value regime
analysis/models/xgboost_trainer.py            research output; should_retrain fixed
analysis/models/system_orchestrator.py        auto-retrain opt-in; real delta
analysis/entry_v2/__init__.py                 quarantine notice + refusal helper
analysis/entry_v2/{dataset_builder,feature_engineering,entry_labels,entry_xgboost_trainer}.py
config.py                                     ENTRY_MODEL_VERSION=v2 rejected
scripts/train_entry_model.py                  alignment, entry price, metadata, gates
scripts/verify_entry_model_v2.py              research output
scripts/{sweep_label_config,evaluate_feature_groups}.py   same alignment rule
tests/{test_entry_feature_parity,test_entry_ml_contract,test_entry_model_currently_gated,test_entry_model_loader}.py
```

**Removed** — nothing.

---

## 9. Remaining problems

1. **No real data has been through the rebuilt pipeline.** Everything above is
   verified on synthetic candles. The real run needs a Windows machine with a
   live MT5 terminal.
2. **Spread and commission are unmodelled.** Labels are optimistic by a known
   sign.
3. **`atr` and `macd` are still price-scale dependent.** Recorded in the
   contract; scale-free replacements exist but adopting them is a Phase 4
   decision.
4. **`momentum_score` is redundant** with `rsi`.
5. **`trend_strength` parity gap** — live uses three timeframes, training two.
6. **Effective sample size is much smaller than the row count.** Adjacent H4
   bars share ~96 of their 100-bar window and label horizons overlap; the
   earlier investigation estimated roughly n/20.
7. **The previous investigation's negative verdict still stands.** On a
   correctly built dataset, H4/H1 indicator features showed no stable edge, with
   ~20σ fold-to-fold instability. Phase 3 removes the reasons to distrust the
   measurement; it does not create signal.
8. **Exit model 13-vs-12 mismatch** — out of scope, still open.

---

## 10. Stop condition

| Gate | Verdict |
|---|---|
| DATA INTEGRITY | PASS (synthetic; real data pending) |
| FEATURE INTEGRITY | PASS with four recorded degradations |
| LABEL INTEGRITY | PASS |
| TIMESTAMP INTEGRITY | PASS — all timeframes |
| MODEL CONTRACT | PASS |
| LIVE COMPATIBILITY | PASS — training and live share one contract, verified by test |
| PRODUCTION MODEL SAFETY | PASS |

## Recommendation

**Ready for Phase 4 — conditional on one step.**

The pipeline is sound and the supply chain is closed. The condition is that
Phase 4 must begin by running the rebuilt pipeline on **real MT5 candles**, on
the Windows machine, and confirming the dataset gate passes there. Every
integrity result above was obtained on synthetic data; synthetic data proves the
code is correct, not that the real series is clean. Grid alignment in particular
is worth watching — a broker timezone offset would show up as
`misaligned_to_grid` and would silently shift every alignment if the check were
not there.

Once that passes: baselines, walk-forward, then feature experiments. Optuna
stays out until there is evidence that a signal exists to tune for. Tuning
hyper-parameters against ~20σ fold instability is fitting noise with more
decimal places.
