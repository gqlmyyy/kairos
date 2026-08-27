# Research Model Integration

How KAIROS consumes the entry models produced by the **xgbooost** research
repository, what contract they speak, and what has and has not been shown
about them.

This document describes code that exists in this repository. Where something
is not done, it says so.

---

## 1. Status

| | |
|---|---|
| Scope | Entry models only. Exit models are untouched. |
| Execution | Offline replay on Linux from stored candles. No MT5, no broker, no account, no orders. |
| Models integrated | 18 — two generations × `{EURUSD, GBPUSD, XAUUSD}` × `{M15, H1, H4}` |
| research_v2 verdicts | 8 × `NO_SIGNAL`, 1 × `DROP` |
| research_v3 verdicts | 7 × `NO_SIGNAL`, 1 × `DROP`, 1 × `CANDIDATE` (GBPUSD M15) |
| `PRODUCTION_ELIGIBLE` | **None.** No model passed the gate. |
| Production activation | None. `ML_ENTRY_ENABLED` and every other switch are unchanged. |
| Golden parity | Bit-exact (0.000e+00) against the research engine on all 18 models |

**The one model with a real signal still fails.** research_v3's GBPUSD M15
(`return_0.5atr_h24`, elastic-net) reaches validation AUC 0.5469 against a
block-permutation null of 0.4994 ± 0.0109 — p = 0.0010, BH q = 0.0090 across
nine datasets. It is `CANDIDATE`, not `VALIDATED`, because the best gross
expectancy it can isolate is **+0.0073 R** against a median trading cost of
**0.1269 R** — costs are ~17× the edge. That gap is a property of the
instrument and timeframe, not of the model, and no threshold or retraining
closes it.

**Nothing here is evidence that any model is profitable or ready for
production.** The research repository's own gates rejected every one of these
artifacts. They are integrated so the *plumbing* can be verified and so a
future model that does pass its gates has a contract to arrive through.

---

## 2. The problem this replaced

KAIROS carried several incompatible entry-model contracts at once:

| Path | Features | State |
|---|---|---|
| `analysis/models/entry_feature_spec.py` | 10 | live contract |
| `models/entry/entry_model.json` | 65 | deployed artifact |
| `analysis/features/ml_dataset_builder.py` | 12 | a third training vocabulary |
| `analysis/entry_v2/` | 65 | abandoned experiment |

Live inference sent 10 values to a 65-feature artifact. XGBoost accepted the
short vector, treated the 55 absent columns as `missing`, followed default
branch directions and returned a plausible number — so every entry decision
was gated on a probability unrelated to the trade, and BUY and SELL received
identical values. That is documented in
`analysis/models/entry_feature_contract.py` and is now blocked by a hard
contract check.

Three further defects were known and are now addressed by the research
contract rather than papered over:

- **Price-scale features.** `atr` and `macd` are in price units. XAUUSD traded
  1619–2789 in the research TRAIN window and 2845–3784 in VALIDATION — zero
  overlap — so every tree split on such a column is a constant in validation.
  This is a domain shift, not overfitting, and retraining does not fix it.
- **Training/live parity gap.** Legacy `trend_strength` used H4+H1 in training
  and H4+H1+M15 live.
- **Redundant capacity.** Legacy `momentum_score` is a three-bucket re-encoding
  of `rsi`, which is already a feature.

---

## 3. The canonical feature contract

`analysis/research/contract.py`. One contract; there is no live variant, no
legacy fallback and no implicit default set.

Every feature declares all of:

```
name  source  formula  timeframe  lookback  minimum_history
dtype  unit  normalization  availability  missing_policy  stationarity  requires
```

`stationarity` is the mechanical form of the research finding. Features are
classified `SCALE_FREE` / `LEVEL` / `PRICE_UNIT` / `BROKER_UNIT`, and
`assert_scale_free()` — called by the loader on every model — refuses a
contract containing anything but the first. Requesting `atr`, `macd_line`,
`ema_20`, `bb_upper`, `range`, `spread_points` or any other price-scale column
raises `ContractError` naming why it is excluded.

The scale-free replacements the research models actually use:

| Instead of | The contract uses |
|---|---|
| `atr` | `atr_pct` = ATR/close, `volatility_score` = ATR / rolling-mean ATR |
| `macd_line` | `trend_strength` = (EMA20−EMA50)/close |
| raw distances | `distance_close_ema{20,50}_atr`, `close_to_sma20_atr` |
| `range`, `body_size`, wicks | `range_atr`, `body_atr`, `upper_wick_atr`, `lower_wick_atr` |
| `spread_points` | `spread_relative` = spread / trailing median |

### Same name, different arithmetic

The legacy and research contracts share several names and **none of the shared
names mean the same thing**:

| Name | Legacy | Research |
|---|---|---|
| `trend_score` | bucket from {40, 65, 70, 75, 85} | normalised OLS slope of close |
| `momentum_score` | RSI bucket {40, 65, 85} | `close.pct_change(10)` |
| `market_regime` | 4 encoded states, −1 for unknown | binary: ADX ≥ 25 |
| `trend_strength` | encoded MTF agreement string | (EMA20−EMA50)/close |

They must never be mapped onto one another. The two vocabularies live in
separate packages, the research package is forbidden from importing the legacy
one, and `test_research_feature_contract.py` asserts the formulas differ.

---

## 4. Multi-timeframe semantics

Each timeframe's frame carries `close_time = timestamp + timeframe_minutes` —
the instant that candle's information becomes knowable. Context timeframes are
joined with `merge_asof(direction="backward", allow_exact_matches=True)` on
`close_time`, so only a **fully closed** context candle is ever visible. A
candle still forming has a later `close_time` and is structurally unreachable.

Stacks are taken from the models, not assumed: M15 entry → H1 + H4 context;
H1 entry → H4 context; H4 entry → no context. **No shipped model uses M30, so
M30 is not computed or added.**

Verified in `tests/test_research_causality.py` by future mutation (rewrite
every bar after a cutoff, assert nothing at or before it moves), MTF boundary
checks, history truncation, and a weekend-gap test.

---

## 5. Availability: VALID / MISSING / INVALID / UNAVAILABLE

`analysis/research/availability.py`. The four states are kept apart because a
model cannot distinguish a substituted value from an observed one — so
substituting does not degrade a prediction, it invalidates it while attaching
a plausible number.

| State | Meaning |
|---|---|
| `VALID` | a real, finite, observed value. **Zero is VALID.** |
| `MISSING` | not yet defined — warm-up incomplete, per feature and per timeframe |
| `INVALID` | NaN after warm-up, an infinity, or a non-numeric type |
| `UNAVAILABLE` | the source cannot produce the column at all |

Any state other than `VALID` on a required feature makes the **prediction**
invalid. It never makes the feature `0.0`.

Contrast the legacy path's `float(value or default)`, which turns a genuine
`0.0` into the default, and `MISSING_DEFAULTS`, which substitutes `50.0` for
an unmeasured score.

---

## 6. Model artifact contract and loader

Each artifact under `models/research/<SYMBOL>/<TF>/` carries `model.joblib`,
the original `research_manifest.json`, and a generated `model_card.json`
declaring `model_id, symbol, timeframe, model_version, feature_schema_version,
feature_list, target, horizon_bars, tp_atr_multiple, sl_atr_multiple,
training_dataset_hash, feature_manifest_hash, target_spec_hash, model_hash,
probability_semantics, research_verdict, calibration, context_timeframes,
decision_threshold` and the build environment.

`load_model()` runs, in order: registry lookup → artifact present → card valid
→ **model hash** → **symbol** → **timeframe** → **feature schema version** →
contract resolvable → **scale-free** → artifact feature count → **feature
order, element by element**. Any failure raises `MODEL_NOT_COMPATIBLE`. There
is no fallback, no truncation, no zero-padding and no positional guessing.

Every one of those checks has a test that breaks it deliberately
(`tests/test_research_model_compatibility.py`).

### Registry

`models/research/registry.json`. Five statuses, weakest first:

| Status | Meaning |
|---|---|
| `RESEARCH` | imported and contract-checked; research found no demonstrated edge |
| `CANDIDATE` | statistically established in research, but calibration or economics failed |
| `VALIDATED` | passed everything through the cost-aware validation backtest |
| `PRODUCTION_ELIGIBLE` | the above **plus** a passing one-shot out-of-sample, verified hashes, verified KAIROS parity, and a countersigned human approval |
| `RETIRED` | superseded or withdrawn; kept for comparison, never served |

An import can never write `PRODUCTION_ELIGIBLE` — not even for a model the
research repository classified that way. A status copied across a repository
boundary is a claim, not a check, so the importer maps such a model down to
`VALIDATED` and KAIROS re-derives the last rung itself.

### The production-eligibility gate

`analysis/research/production_gate.py` is the only route to
`PRODUCTION_ELIGIBLE`. It requires all seven, each from evidence on disk:

```
model_card         loads and validates
artifact_verified  the bytes hash to what the card declares
contract_verified  features resolve, are scale-free, fingerprint matches
research_verdict   research classified this artifact PRODUCTION_ELIGIBLE
final_oos          a recorded one-shot out-of-sample pass exists
economics          the economic gate passed AND survived the cost stress
feature_parity     KAIROS reproduced the research vectors within tolerance
explicit_approval  a signed record naming a person, a date and the model hash
```

`promote()` refuses and changes nothing if any check fails, naming which.

**Eligible is not enabled.** Nothing in KAIROS reads this status to switch
trading on; there is no auto-enable and the gate deliberately provides none.
A test (`test_eligibility_is_not_activation`) asserts the constant is
referenced nowhere outside the registry, the gate and the importer.

The approval record is a file rather than a flag because a flag can be
flipped by anyone and carries no account of why. An approval names who, when,
and against which artifact hash — change the model by one byte and the
approval stops applying to it.

```bash
python scripts/record_research_parity.py     # measures parity, records it
python -c "from analysis.research import production_gate as g;            print(g.evaluate('models/research/XAUUSD/H1').describe())"
```

---

## 7. `p_win` semantics

There are now **two kinds** of research entry model, and they emit different
probabilities. The kind travels with the artifact (`target_kind` on the card)
and the loader refuses any card whose declared semantics do not match it.

| kind | `p_win` | 1R is |
|---|---|---|
| `barrier` | `P(TP before SL \| entry_direction)` | the SL distance |
| `return` | `P(forward move beyond threshold \| entry_direction)` | 1 ATR at entry |

A barrier model's 0.55 and a return model's 0.55 do not mean the same thing,
and their R-multiples are not comparable. research_v2 shipped barrier models
only; research_v3 selects the target per dataset and produces both.

For the barrier kind: TP at 2.5×ATR(14) and SL at 1.5×ATR(14), both fixed from
the ATR at the entry bar, within a bounded horizon (M15 96 bars, H1 72, H4 60).

It is **not** P(price goes up), **not** an expected return, and **not** a
confidence score. Both sides of a bar are scored independently and their
probabilities do not sum to 1. The card declares the semantics and the loader
refuses any artifact declaring different ones.

### The shipped calibrators cannot reach 0.5

Every model's isotonic calibrator is a coarse step function (5–23 distinct
levels) whose output **never reaches 0.5** anywhere in [0, 1]. Each model's
research-selected threshold is ≈0.38.

A downstream gate written against the legacy path's hard-coded `p_win >= 0.60`
would therefore never fire on any of these models. This is a property of
`NO_SIGNAL` artifacts, not a defect in the integration, and it is pinned by
`tests/test_research_probability_semantics.py` so a future artifact that
changes it is noticed.

---

## 8. Offline replay

```bash
# against a source that carries every canonical column
python scripts/research_replay.py --symbol XAUUSD --tf H1 \
    --source-kind json --source-root tests/fixtures/research/candles --tail 6

# against KAIROS's stored history
python scripts/research_replay.py --symbol XAUUSD --tf H1 --tail 5
```

`--tail` scores the most recent N rows; `--limit` scores the first N (which on
a short history are still inside their lookback windows and correctly produce
nothing).

### Known blocker: the committed candle snapshot predates the spread fix

`scripts/fetch_training_candles.py` **does** record MT5's per-bar `spread`
(added in commit `30afc30`). The snapshot committed under `data/historical/`
was fetched a few hours earlier the same day and holds only
`{t, open, high, low, close, volume}`.

`KairosHistoricalSource` therefore **probes the files** rather than assuming an
answer: a refreshed snapshot satisfies the contract and is served; the current
one does not, and the pipeline reports:

```
0 scored, N refused, statuses={'FEATURE_UNAVAILABLE': N}
  source is missing columns ['spread'], which makes 2 contract features
  UNAVAILABLE: ['spread_relative', 'H4_spread_relative']
```

81 of 83 features compute correctly on the real data; the two spread features
cannot, and the prediction is refused rather than computed from an invented
spread of zero.

**The fix is to re-run the existing fetcher** on a machine with MT5, which
already captures the field — not to fabricate the column. No code change is
needed for that; the detection picks it up automatically.

The detection is whole-source and conservative: if any candle file in the
directory lacks the column, the source declares spread absent. A stack where
H1 carried spread and H4 did not would assemble an entry vector whose context
features came from a different contract than its own.

KAIROS also stores no M15 candles at all, so the six M15 models have no
real-data replay source here. Their golden fixtures use a clearly-labelled
`SYNTH_` stack, which is why they still prove implementation parity but cannot
be replayed against real candles.

---

## 9. Golden parity

`tests/fixtures/research/golden/` holds reference vectors and predictions
produced by the **xgbooost repository's own engine and its own shipped
models**, over the shared candle fixture in
`tests/fixtures/research/candles/`. KAIROS recomputes from the identical file
and must match.

Result: **all 9 models, 0.000e+00 maximum difference** on every feature value
and on both the raw and calibrated probabilities.

Regenerate with:

```bash
# in the xgbooost repo
python scripts/export_kairos_golden.py \
    --candles ../kairos/tests/fixtures/research/candles \
    --out ../kairos/tests/fixtures/research/golden
```

**What the fixtures are not.** The fixture `spread` column is synthetic — a
documented deterministic function of the bar, since KAIROS's real candles have
no spread — and the M15 stack is a random walk. These files prove that two
implementations agree on identical input. They are not evidence about markets.
Each golden file records its own provenance and the model's verdict.

---

## 10. Paths: canonical, legacy, deprecated

| Path | Status | Note |
|---|---|---|
| `analysis/research/` | **canonical** | the only research-model contract |
| `scripts/import_research_model.py` | **canonical** | the only writer into `models/research/` |
| xgbooost `src/research/` | **canonical (training)** | KAIROS does not train research models |
| `scripts/train_entry_model.py` | legacy | trains the 10-feature legacy model; writes `models/entry/` only |
| `analysis/models/entry_feature_spec.py` | legacy | the 10-feature live contract |
| `analysis/entry_v2/` | abandoned | 65-feature experiment, kept for comparison |
| `analysis/features/ml_dataset_builder.py` | legacy | 12-feature DB path behind `train_pipeline.py` |
| `train_from_historical.py` | deprecated | prints a pointer; imports nothing |

Legacy artifacts are **preserved, not replaced**: `models/entry/entry_model.json`
is still the 65-feature file it was, and a test asserts it. The two stores
cannot reach each other — enforced by `tests/test_research_training_paths.py`.

---

## 11. Reproducing everything

```bash
# import both research generations (artifacts are version-scoped and coexist)
python scripts/import_research_model.py --source ../xgbooost --version research_v2
python scripts/import_research_model.py --source ../xgbooost --version research_v3

python scripts/build_research_fixtures.py            # rebuild candle fixtures
python scripts/record_research_parity.py             # measure + record v2 parity
python scripts/record_research_parity.py --prefix golden_v3   # ...and v3

python -m pytest tests/ -q                           # full suite (1363 tests)
python -m pytest tests/ -q -k research               # this integration only
```

Both generations stay separately runnable. With two registered for the same
symbol/timeframe, a load must name one — the registry refuses to choose,
because picking by recency or status would be exactly the implicit second
source of truth this design exists to prevent:

```bash
python scripts/research_replay.py --symbol GBPUSD --tf H1 --version research_v3 \
    --source-kind json --source-root tests/fixtures/research/candles --tail 5
```

Tests: contract schema/order/formulas, MTF causality and future mutation,
session and weekend boundaries, model metadata, compatibility, hashes, symbol
and timeframe mismatch, missing and zero values, offline replay, golden
parity, determinism, probability semantics, and path separation.

---

## 12. What is not done

- **No live inference path.** Nothing calls this from `main.py`. Wiring it in
  is a separate decision, and on `NO_SIGNAL` models there is no reason to.
- **No production activation.** No switch was changed.
- **No exit-model work.** `models/exit/` and its dataset are out of scope and
  untouched.
- **The committed candle snapshot predates the fetcher's spread fix**, so no
  research model can be served from KAIROS's own history until it is
  re-fetched (§8). The fetcher already captures the field; no code change is
  required.
- **No M15 candles**, so the three M15 models have no real-data replay source.
- **No claim of predictive value.** Every integrated model was rejected by the
  research repository's own gates.
