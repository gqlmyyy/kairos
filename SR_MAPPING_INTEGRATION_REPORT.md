# SR_Mapping_NN → KAIROS: Audit and Compatibility Report

Reference: https://github.com/Mrizalfahlepi/SR_Mapping_NN, commit
`c0115aa83bf30538030b80c69b4b0f0ce0553150`.

Audited before any KAIROS code was modified, per the brief's step 33.

---

## 1. What the reference model actually is

An XGBoost binary classifier over 26 features, trained on GC=F H1 candles from
yfinance, labelled TP-before-SL at 2.0/1.6 × ATR with a 48-bar timeout, split
70/15/15 chronologically (2828 / 606 / 606 rows).

Its headline claim is 72.4% precision. **That claim does not survive
inspection**, and neither do two other properties needed for it to be usable.

### 1a. Seven of its 26 features read two bars into the future

`scripts/02_feature_engineering.py`:

```python
if all(highs[i] > highs[i-j] and highs[i] > highs[i+j]
       for j in range(1, lookback+1)):
    fup[i] = highs[i]
```

`highs[i+j]` for j∈{1,2}. A Williams Fractal at bar i is decided by bars i+1
and i+2, then carried forward as the active support/resistance level.

Reproduced on the project's own `data/xauusd_h1.csv` — mutating **only** bars
after index 3000, leaving bar 3000 itself untouched:

```
fractal_down[3000]:  before = NaN   →   after = 3115.20
fractal_up[2999]  :  also changed
```

Affected features: `dist_res_norm`, `dist_sup_norm`, `sr_position`,
`near_support`, `near_resistance`, `bars_since_frac_up`,
`bars_since_frac_down`.

### 1b. The threshold was chosen on the test set, and its precision reported from the same set

`scripts/04_train_model.py` scans 55 thresholds against `y_test`, picks the
best, and writes that precision into `feature_config.json` as
`test_precision`. Computed from their shipped `data/test_predictions.csv`:

| | |
|---|---|
| test base rate | 0.5132 |
| signals at t=0.51 | 29 |
| wins | 21 (precision 0.7241) |
| one-sided binomial p, ignoring selection | **0.0173** |
| same p, Bonferroni-corrected for 55 thresholds tried | **0.9491** |

Not significant.

### 1c. The model barely separates anything

From the same file:

```
p_win min = 0.4901   max = 0.5116   std = 0.0046
100% of predictions fall inside [0.49, 0.52]
```

It never expresses more than 51.16% confidence about any trade. Threshold 0.51
slices the top ~5% of a noise distribution.

### 1d. Its own equity report shows the filter losing money

From `docs/VALIDATION_REPORT.md`:

| Strategy | Final | Trades | Win rate |
|---|---|---|---|
| EA, no filter | $11,946 | 606 | 51.3% |
| EA + NN filter | **$10,504** | 29 | 72.4% |

Profit fell from +$1,946 to +$504. The report describes this as a significant
improvement because it reads the win-rate column only.

**Conclusion: the model is discarded. Only the feature ideas are carried
across** — and only after the look-ahead is fixed.

---

## 2. Compatibility map

| Dimension | SR_Mapping_NN | KAIROS | Resolution |
|---|---|---|---|
| Data source | GC=F yfinance | MT5 broker feed | KAIROS (brief §6) |
| Entry timeframe | H1 | H4 decision + H1 context | **H4 first** (agreed) |
| SL / TP | 1.6 / 2.0 ×ATR (R:R 1.25) | **1.5 / 2.5 ×ATR (R:R 1.667)** | KAIROS, from `tm_config` (brief §9) |
| Horizon | 48 H1 bars | 24 H4 bars | KAIROS |
| Trade direction | `daily_direction` | signal engine | **direction-free** (agreed) |
| TP+SL in one bar | counted WIN | counted LOSS | KAIROS (conservative) |
| Timeout policy | floating P/L decides | row dropped | KAIROS — the floating-P/L rule is the same defect that invalidated `entry_v2` |
| Feature count | 26 (7 leaking) | 10 | 19 adapted, leak fixed |
| Return units | percent | ATR-normalised | KAIROS — percent is scale-dependent across instruments |
| Normalisation | stored mean/std | none | Not needed; trees are scale-invariant per feature |
| Threshold | 0.51, picked on test | to be picked on validation | KAIROS (brief §12) |
| Model format | pickle + ONNX | Booster JSON + metadata sidecar | KAIROS |

**`ML_MODEL_MISSING` root cause confirmed:** `models/entry/` contains
`entry_model.json` with no `entry_model.json.metadata.json`. That is the
fail-closed provenance gate built in Phase 3 working exactly as designed — an
artifact that cannot prove its schema is refused rather than served.

---

## 3. What was built

`analysis/features/sr_structure_features.py` — 19 features carrying the
reference's ideas with its defects removed:

* **Look-ahead fixed.** `confirmed_fractals()` indexes a fractal at the bar
  where it becomes *knowable* (centre + 2), not at its centre. This also
  matches what a live chart shows: a fractal that has not printed yet is not
  visible to a trader.
* **Scale-free.** Returns are in ATR units, not percent; distances and candle
  anatomy are ATR-normalised. Tested: identical relative dynamics at price
  1.10 and 2330.0 produce identical features to 1e-6.
* **Conservative on missing data.** Returns `None` on short history or zero
  ATR rather than dividing.

`scripts/evaluate_sr_features.py` — runs these features through the same gate
that produced the RED verdict: stratified AUC (base-rate effects removed),
block-permutation null, decorrelation-aware effective sample size,
direction-free walk-forward, XAUUSD only.

`tests/test_sr_structure_features.py` — 11 tests, including a deterministic
reproduction of the reference's leak proving the no-look-ahead check can
actually fire.

---

## 4. Files

**Added**

```
analysis/features/sr_structure_features.py
scripts/evaluate_sr_features.py
tests/test_sr_structure_features.py
SR_MAPPING_INTEGRATION_REPORT.md
```

**Changed** — none. No production file, no model, no loader, no risk logic.
`models/entry/entry_model.json` sha256 `ecbfc94b…` unchanged.

**Deliberately NOT yet built:** the model registry, `xauusd_entry_model.json`,
its metadata sidecar, the training script, and shadow mode. Those are the
right things to build *after* the features show signal, not before — the
brief's own §15 and §21 require the feasibility gate to pass first. Building a
registry and a shadow-mode harness for features that may return FAILED would
repeat the pattern this whole engagement has been correcting.

---

## 5. Commands

```bash
# 1. verify holdout isolation on real data (refuses to proceed if it fails)
python scripts/evaluate_sr_features.py --verify-holdout-isolation

# 2. the actual feasibility question, XAUUSD only
python scripts/evaluate_sr_features.py --permutations 200
```

Verified end-to-end on synthetic noise: holdout isolation passes, every
feature correctly fails its permutation test, verdict **FAILED**. The
machinery behaves as designed before being pointed at real data.

---

## 6. Verdict scale

The script emits exactly one of:

| Verdict | Meaning |
|---|---|
| `PRODUCTION CANDIDATE` | beats the null, stable across folds, improves on the old features, and reaches AUC ≥ 0.55 |
| `NOT PRODUCTION READY` | beats the null but is unstable or adds nothing on top of existing features |
| `NO EDGE` | near chance |
| `FAILED` | inside the noise distribution |

---

## 7. Blockers to production activation

Every one of these is currently unmet, and none can be waived:

1. **Feasibility gate has not been run on real data.** No result exists yet.
2. **No model trained** — deliberately, pending (1).
3. **No registry / metadata sidecar** — pending (1).
4. **No shadow-mode evidence** — pending (2).
5. **Account size.** Equity $98.90 against a 0.01 minimum lot means the risk
   engine rejects all three symbols on hard-ceiling grounds. This is
   unrelated to ML and must not be "fixed" by loosening the ceiling, per
   brief §22–23. Even a perfect entry filter cannot trade this account.

Blocker 5 stands regardless of what the model does.
