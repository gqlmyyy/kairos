# Entry Model — Investigation Result

Three years of real MT5 candles for EURUSD, GBPUSD and XAUUSD. Nine labelling
configurations, six cumulative feature groups, walk-forward validation
throughout. The conclusion is negative and the promotion gate correctly refuses.

## Verdict

**No exploitable edge was found. The current model is not replaced.**

The bottleneck is not the labelling, the horizon, the TP/SL ratio, or the
feature set. On this data, at this timeframe, **H4/H1 indicator features do not
predict whether a fixed-R trade reaches its target before its stop** — and the
relationship is not even stable across time.

## Evidence

### 1. Labelling is not the constraint

Nine (SL, TP, horizon) configurations, walk-forward:

| SL:TP | H | rows | win% | AUC | fold spread | folds>0.5 |
|---|---|---|---|---|---|---|
| 1.0:1.5 | 8 | 24,550 | 38.5% | 0.5088 | 0.0459 | 1/3 |
| 1.0:1.5 | 16 | 27,449 | 39.7% | 0.5055 | 0.0657 | 2/3 |
| 1.0:1.5 | 24 | 27,894 | 39.9% | 0.5090 | 0.0555 | 2/3 |
| 1.5:2.5 | 8 | 16,599 | 31.1% | 0.5173 | 0.0487 | 3/3 |
| 1.5:2.5 | 16 | 23,532 | 35.8% | 0.5149 | 0.0554 | 2/3 |
| 1.5:2.5 | 24 | 26,212 | 37.1% | 0.5183 | 0.0541 | 2/3 |
| 1.5:3.0 | 8 | 14,989 | 23.1% | 0.5103 | 0.0572 | 2/3 |
| 1.5:3.0 | 16 | 22,005 | 29.7% | 0.5129 | 0.0641 | 2/3 |
| 1.5:3.0 | 24 | 25,203 | 32.2% | 0.5227 | 0.0392 | 3/3 |

Tripling the horizon moves AUC by **<0.02**. Changing the reward:risk ratio
moves it by **<0.02**. Every configuration lands in 0.505–0.523, and in every
one the fold-to-fold spread is **2–12x** the distance from 0.5.

### 2. Feature engineering did not help — it hurt

36 scale-free features in six groups, added cumulatively, label fixed at
SL 1.5 / TP 2.5 / horizon 24, 26,242 rows, BUY 13,125 / SELL 13,117:

| step | features | AUC | ΔAUC | fold spread | folds>0.5 | PR lift | Brier skill |
|---|---|---|---|---|---|---|---|
| baseline | 9 | **0.5273** | — | 0.1874 | 4/5 | +0.0256 | **−0.0184** |
| +trend | 18 | 0.5121 | −0.0152 | 0.1572 | 4/5 | +0.0100 | −0.0261 |
| +volatility | 22 | 0.5020 | −0.0101 | 0.1735 | 3/5 | +0.0075 | −0.0302 |
| +momentum | 27 | 0.5044 | +0.0024 | 0.1889 | 4/5 | +0.0083 | −0.0280 |
| +structure | 32 | 0.5020 | −0.0024 | 0.1936 | 3/5 | +0.0069 | −0.0288 |
| +mtf | 36 | 0.5100 | +0.0080 | 0.1914 | 4/5 | +0.0139 | −0.0224 |

**Total gain from all feature engineering: +0.0000.** The best step is the
9-feature baseline. Every group either degraded out-of-sample performance or
moved it within noise — the signature of fitting noise when there is no signal
to fit.

**Brier skill is negative at every step.** The model's probabilities are worse
than simply assigning every trade the base rate. That is a stronger and cleaner
statement than AUC: the output is not merely uninformative, it is misleading.

### 3. The relationship is not stable across time

This is the finding that closes the question. Test folds hold ~4,373 rows, where
the sampling standard error of AUC is about **0.0090**. Observed fold spreads:

| step | spread | in standard errors |
|---|---|---|
| baseline | 0.1874 | **20.7σ** |
| +trend | 0.1572 | 17.4σ |
| +volatility | 0.1735 | 19.2σ |
| +momentum | 0.1889 | 20.9σ |
| +structure | 0.1936 | 21.4σ |
| +mtf | 0.1914 | 21.2σ |

A ~20σ spread is not sampling noise. The folds genuinely disagree: what holds in
one period reverses in another. Even if an edge existed in the training window,
it would not survive into the next one.

### 4. Per-symbol: two of three are at or below chance

| step | EURUSD | GBPUSD | XAUUSD |
|---|---|---|---|
| baseline | 0.5160 | 0.5150 | 0.5522 |
| +trend | 0.4997 | 0.4851 | 0.5420 |
| +volatility | 0.4922 | 0.4692 | 0.5343 |
| +momentum | 0.4976 | 0.4711 | 0.5350 |
| +structure | 0.4887 | 0.4729 | 0.5321 |
| +mtf | 0.4984 | 0.4758 | 0.5425 |

EURUSD and GBPUSD sit at or **below** 0.5 in nearly every step. Only XAUUSD is
consistently above (0.532–0.552). Given the ~20σ fold instability above, this is
a lead to test rather than a result to trade — see Remaining risks.

## What the investigation did fix

These were real defects found and corrected along the way, independent of the
modelling outcome:

1. **ROC-AUC was computed wrongly for tied scores.** `argsort().argsort()`
   assigns ordinal ranks and broke ties by array position, with positives
   concatenated first. A constant predictor scored 0.0 instead of 0.5 — which is
   why the `train_prior_constant` baseline reported `roc_auc: 0.0174`. Every
   baseline comparison before this fix was unreadable. Now Mann-Whitney with
   average ranks; verified constant→0.5, perfect→1.0, inverted→0.0.

2. **`market_regime` collapsed three regimes into one value.** RANGING,
   HIGH_VOLATILITY and LOW_VOLATILITY all encoded to 0.0.

3. **`atr` and `macd` were raw and pooled across symbols** with a 25,000x scale
   gap, so a tree splitting on them was splitting on which instrument it was
   looking at.

4. **Features were redundant** — `momentum_score` was a three-bucket re-encoding
   of `rsi`, itself feature #1.

Fixes 2–4 raised the baseline from 0.5078 to 0.5273. Real, but still inside the
noise band, and they do not change the verdict.

## Promotion decision

```
REJECT — KEEP CURRENT MODEL
```

`--min-auc` was not lowered, the promotion gate was not modified or bypassed,
and `models/entry/entry_model.json` is unchanged (sha256 verified).

Note what this means operationally: the current model expects 65 features while
the live path sends 10, so the ML gate refuses every signal and **the bot still
cannot open a trade**. Nothing here changes that. The choice is between a bot
that does not trade and a bot that trades on a model with no demonstrated edge
and negative Brier skill; refusing to promote is the correct half of that
choice, but it is not a working system.

## Remaining risks and honest unknowns

- **This tested one hypothesis, not "ML for entries".** The hypothesis was:
  *classical H4/H1 indicators predict a fixed-R barrier outcome*. It is refuted
  on this data. Other framings — shorter timeframes, order-flow or spread data,
  predicting a continuous return instead of a barrier hit, or a regime filter
  rather than a per-trade probability — were not tested and are not ruled out.

- **XAUUSD may carry something the others do not**, but with 20σ fold
  instability it cannot be separated from noise at this sample size. Testing it
  properly means a gold-only model with its own walk-forward, not reading the
  per-symbol column of a pooled model.

- **Three years may be too short.** ~4,800 H4 bars per symbol is 26k rows but
  far fewer independent observations, because adjacent bars share ~96 of their
  100-bar indicator window and overlapping horizons make consecutive labels
  dependent. The effective sample is closer to n/20.

- **The `sig` column in the label sweep was optimistic.** It assumed independent
  samples; correcting for that overlap inflates its z-values ~4.5x, so the 2–4
  "significant" buckets were most likely not significant.

- **Labels ignore spread and commission.** A real 1.5:2.5 trade pays the spread
  twice. Including it would lower every win rate and make the target harder, not
  easier — so this does not rescue the result.

- **Not tested:** classifier hyper-parameters were held fixed. Given negative
  Brier skill and 20σ fold instability, tuning them would be fitting noise, but
  it is a stone left unturned.
