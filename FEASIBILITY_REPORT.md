# Pre-Training Feasibility Gate — Final Verdict

Real MT5 data: EURUSD/GBPUSD/XAUUSD, H4+H1, 2023-08-02 to 2025-09-11 (research
split; 7,870 rows of holdout never read). `python scripts/feasibility_gate.py
--permutations 200`. No model trained, no Optuna, no production file touched.

## Verdict: RED — DO NOT TRAIN

**Score: 10/100.**

## What the first run said, and why it was wrong

The first pass scored 63/100 YELLOW off `mean AUC 0.5446`. Reading its own
ablation table showed why that was an artifact, not evidence:

```
direction alone        0.5468
all ten features        0.5446   <- nine more features made it WORSE
every other group      0.4973–0.5040   (exactly chance)
```

Over 2023–2025 all three instruments trended up. Longs reached the 2.5-ATR
target before the 1.5-ATR stop more often than shorts (BUY 41.1% vs SELL
33.2%) — a base rate, not discrimination. `direction` is not a feature that
predicts trade quality; it is the side already chosen by the rule layer before
this model runs. The scoring counted "prefer BUY over this sample period" as
signal. Fixed: every model-level test now runs **direction-free**.

## Corrected numbers

| test | value | reading |
|---|---|---|
| Direction-free logistic, walk-forward | **0.5010** | 68th percentile of its own block-permutation null — indistinguishable from noise |
| Best stratified univariate feature | momentum_score, 0.0060 | 38th percentile of null |
| Feature groups (direction excluded) | 0.4973 – 0.5040 | all at chance |
| Horizon sweep (direction-free) | 0.4997 – 0.5213 | flat, no rising/falling structure |
| Cross-symbol (direction-free) | 0.4959 – 0.5046 | flat, none stable above 0.54 |
| BUY vs SELL, base rate held fixed | 0.5116 (2/4 folds) / 0.5179 (3/4 folds) | inside noise, unstable |
| Effective sample size | ~574 of 18,361 nominal rows | label decorrelation length 32 rows |

## Section-by-section

**Target Health** — weak. 37.15% class balance. No symbol/regime/session slice
exceeds its own noise band (SYMBOL ±0.068, REGIME ±0.042–0.30, SESSION ±0.068).

**Feature Information** — near-zero. Mutual information is exactly 0.0 for
seven of ten features. Best stratified deviation across all ten: 0.0060.

**Feature Groups** — none carry signal. Every group 0.4973–0.5040.

**Horizon Analysis** — flat once direction-free: 0.4997–0.5213 across
4/8/12/16/20/24 bars. This is the "structural absence" pattern named in the
original brief, not the rising-then-falling curve that would justify further
horizon search.

**Timeframe Analysis** — H4/H1 not obviously mismatched; only 16.1% of trades
resolve within 2 bars, so the resolution isn't the binding constraint. Return
autocorrelation is ±0.02–0.03 at every lag on all three instruments — no
exploitable momentum in the raw price series either.

**Symbol Analysis** — direction-free, flat: EURUSD 0.5021, XAUUSD 0.5046,
GBPUSD 0.4959. The earlier XAUUSD 0.5908 was direction leaking through the
pooled model, not an instrument-specific edge.

**Direction Analysis** — is the entire earlier "signal". Holding the base rate
fixed removes it.

**Regime Analysis** — none. TRENDING holds 87.6% of rows at Δ=−0.0007 from
base rate.

**Simple Baselines** — logistic 0.5010, tree ~0.50, forest ~0.50, all
direction-free. None beat the permutation null.

**Target Design Audit** — the symmetric barrier (removes the 1.5:2.5
asymmetry) drops to 0.0031 stratified deviation, *below* the current
asymmetric target's 0.0060. This is not promise for an alternative target — it
confirms the asymmetric barrier's small apparent edge was itself skew, not
signal. `B_forward_sign` scored 0.0115, best of the alternatives and not
volatility-tautological, but still deep inside the noise band measured in
section 3 (±0.054 at this effective sample size).

**Predictive Signal Evidence: NONE.**

**Training Feasibility Score: 10/100.**

## Root Cause Ranking

1. **I — No real predictive signal exists** in this feature set against this
   target. Every direction-free test — univariate, model, group, horizon,
   symbol — lands inside its own permutation null. This is not "weak", it is
   absent by every measure applied.
2. **F — Direction was masquerading as signal.** The entire apparent edge in
   the first run traced to one column encoding which side already trended,
   not to any discriminative feature.
3. **B — Target design contributes, but not favorably in the tested
   alternative.** The asymmetric TP/SL barrier's small stratified deviation
   (0.0060) is itself partly sample-period skew — the symmetric barrier scored
   *lower* (0.0031), not higher.
4. **A — Feature information is exhausted.** Seven of ten features carry zero
   measured mutual information with the target. Not "poorly combined" —
   individually uninformative.
5. **C, D — Timeframe and horizon are not implicated.** Both were tested
   directly and neither shows structure that better resolution or a different
   horizon would recover.
6. **E — Symbol pooling is not hiding anything.** Direction-free per-symbol
   AUC is flat across all three instruments.

## Decision: RED

No evidence of predictive information survives a noise floor built to respect
this data's autocorrelation. Retraining XGBoost on these ten features against
this target, however tuned, will not manufacture information that direction-
free testing could not find. Optuna redistributes a score across
hyperparameters; it does not create signal absent from the inputs.

## Next Action

The framework — these ten features, this asymmetric barrier target, H4/H1,
pooled symbols — is exhausted as tested. Before any further training:

1. **New feature inputs, not new combinations of the existing ten.** Seven of
   ten already carry zero mutual information. Candidates not yet available in
   this pipeline: spread/liquidity data, an economic calendar, positioning
   data — none of which the current H4/H1 OHLC-derived indicator set can
   proxy.
2. **Reconsider whether an ML entry filter is the right layer at all.** The
   bot has six trade-management layers already operating. A filter with no
   demonstrable signal adds gating risk (blocking real trades on noise)
   without a compensating edge.
3. If new inputs are pursued, they go through this same gate — diagnose and
   prove signal direction-free before any training — not straight to Optuna.

`models/entry/entry_model.json` is untouched. No production behavior changed.
