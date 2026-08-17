# Information Discovery — Repository Audit and Research Pipeline

## 1. Executive Summary

`FEASIBILITY_REPORT.md` established RED: the existing ten features carry no
predictive signal against the barrier target, direction-free, on real MT5
data (logistic AUC 0.5010, 68th percentile of its own permutation null,
feasibility score 10/100).

This phase does not retest those ten features. It audits the repository for
genuinely new information sources, builds a leakage-safe pipeline to test
them, and reports what it finds — with a programmatic proof, not a claim,
that the research pipeline never reads the holdout set.

**One source was discovered that is already in hand and was simply never
captured**: MT5's rate structure returns `spread` and `real_volume` alongside
every OHLC bar; both `data/market/mt5_client.py::get_candles` and
`scripts/fetch_training_candles.py` discarded them. Capturing them needed one
line each, no new dependency, and adds no look-ahead risk beyond what OHLC
capture already carries — spread is a property of the closed bar, known at
the same instant as its open/high/low/close.

**Real-data results, run 2026-08-17 (§7–13): RED.** Holdout isolation
verified against real data first (18,353 research rows, 7,877 holdout,
byte-identical after full holdout corruption). Neither microstructure
(spread/volume) nor cross-asset context (DXY, US10Y, US13W, silver, oil —
all five fetched successfully) produced a single feature beating its own
block-permutation null. `OLD + NEW` scored worse than `OLD` alone in both
runs. See §13b for the verdict and §15 for what remains genuinely untested.

## 2. Current RED Baseline (unchanged, carried forward)

```
direction-free logistic AUC = 0.5010   (68th pct of block-permutation null)
EURUSD = 0.5021   XAUUSD = 0.5046   GBPUSD = 0.4959
best stratified univariate |AUC-0.5| = 0.0060   (38th pct of null)
feasibility score = 10/100
verdict = RED
```

`models/entry/entry_model.json` unchanged throughout (sha256
`ecbfc94b...`). Not retested here — this phase is additive.

## 3. Data Sources Discovered

| Source | Status | Basis |
|---|---|---|
| Spread (per-bar) | **AVAILABLE, not previously captured** | MT5 rate struct field, discarded by every prior export |
| Tick volume | Available, unused as a feature | Already in every candle dict as `volume` |
| Real volume | Present in MT5's struct; frequently constant-zero on FX/CFD | Reported per-instrument by `field_availability`, never assumed |
| DXY / US 10Y / US 13W / silver / oil | Code path exists via yfinance, orphaned before this work | `analysis/market/historical_fetcher.py::AlternativeDataFetchers`, never imported anywhere |

## 4. Data Sources Confirmed Unavailable (checked directly, not assumed)

| Source | Why |
|---|---|
| Historical economic calendar | `data/news/calendar.py` fetches today/tomorrow only via Finnhub; `FINNHUB_API_KEY` is unset by default; no archive mechanism exists |
| News archive | `data/news/fetcher.py` is RSS-live-only; the `news` DB table exists but held **0 rows** at audit time |
| Sentiment | Derived from news; same historical gap |
| Order book / liquidity | Not exposed by the MT5 API calls this repo uses |
| Bid/ask history | `mt5_client.py` only exposes the current live tick (`symbol_info_tick`), no tick archive |

## 5. New Features Built

**Microstructure** (`analysis/features/microstructure_features.py`, 6 features):
`spread_atr`, `spread_percentile`, `spread_zscore`, `real_volume_zscore`,
`real_volume_percentile`, `volume_zscore`. Degrades explicitly — returns
`None` for a row when spread is absent from the source file, and reports
`AVAILABLE (constant — no information)` rather than a fabricated reading when
a field exists but never varies.

**Cross-asset** (`analysis/features/cross_asset_features.py`, up to 10 features
depending on relevance per symbol): daily return and z-score for each of
DXY/US10Y/US13W(/silver/oil for XAUUSD). Availability is what `fetch_all`
actually returns from yfinance — never assumed present.

Neither module touches `analysis.models.entry_feature_spec`. The production
ten-feature contract is unmodified.

## 6. Timestamp / Leakage Audit

- **Microstructure**: spread/volume are read only from `closed_slice` output,
  identical mechanism to every OHLC feature since Phase 3. `TestNoLookAhead`
  in `tests/test_microstructure_features.py` proves mutating post-decision
  candles leaves the feature vector unchanged (and mutating pre-decision
  candles does change it — the non-vacuity control).
- **Cross-asset**: a daily bar dated D becomes available at `(D+1) 00:00 UTC`
  — deliberately pessimistic (real closes are often known same evening) but
  never optimistic, which is the property that matters. Proven in
  `tests/test_cross_asset_features.py`.
- **Holdout isolation**: proven programmatically, not asserted. See §14 —
  two real bugs were found and fixed while building this proof, both
  documented in `tests/test_holdout_isolation.py`'s docstring.

## 7–13. Experimental Results — Real Data

Run 2026-08-17, EURUSD/GBPUSD/XAUUSD, H4+H1, same real MT5 fetch as
`FEASIBILITY_REPORT.md` plus freshly captured spread. Holdout isolation
**VERIFIED against this real data** first (18,353 research rows, 7,877
holdout, byte-identical after full holdout corruption) — everything below is
reported only because that check passed.

### §7–8 Microstructure only (spread, volume; real_volume constant on this broker)

| feature | stratified \|AUC-0.5\| | MI | note |
|---|---|---|---|
| volume_zscore | 0.0064 | 0.000000 | best of the six, still noise |
| spread_percentile | — | 0.003495 | |
| spread_atr / spread_zscore | — | 0.000000 | |
| real_volume_* | n/a | n/a | constant on this broker — confirmed, not assumed |

Best stratified deviation (volume_zscore, 0.0064) sits at the **64.5th
percentile** of its block-permutation null (median 0.0054, p95 0.0108) —
**NO EVIDENCE**, needs ≥95th to count.

```
OLD FEATURES ONLY (direction-free)   9 features   AUC 0.5015
NEW INFORMATION ONLY (spread/vol)    6 features   AUC 0.5002
OLD + NEW                           15 features   AUC 0.4990   (worse than OLD alone)
```

Symbol-specific (new-info-only): EURUSD 0.5012, XAUUSD 0.5043, GBPUSD 0.4985
— flat, none above ~0.505.

### §7–8 Microstructure + cross-asset (DXY, US10Y, US13W, silver, oil)

All five yfinance sources **fetched successfully** — full coverage,
18,353/18,353 research rows.

| feature | stratified \|AUC-0.5\| | MI |
|---|---|---|
| oil_zscore_20d | **0.0128** (highest of all 16 new features) | 0.006824 |
| dxy_return_1d | — | **0.011983** (highest MI) |
| us10y_return_1d | — | 0.008758 |
| oil_return_1d | — | 0.008000 |
| silver_return_1d | — | 0.003853 |
| real_volume_percentile | — | 0.005980 |
| us13w_* | — | 0.000000 |

`real_volume_percentile`'s 0.005980 MI is flagged, not reported as a lead:
this column is constant on this broker (confirmed in the microstructure-only
run, `std=0.0000`), and sklearn's k-NN mutual-information estimator can
return small positive noise on a near-constant column. A genuinely
zero-information feature does not get a second reading just because a
different feature set was evaluated alongside it.

Best stratified deviation overall (oil_zscore_20d, 0.0128) sits at the
**55.0th percentile** of its null (median 0.0118, p95 0.0215) — **NO
EVIDENCE**.

```
OLD FEATURES ONLY (direction-free)    9 features   AUC 0.5015
NEW INFORMATION ONLY (16 features)   16 features   AUC 0.4969
OLD + NEW                            25 features   AUC 0.4967   (worse than OLD alone)
```

Symbol-specific (new-info-only): EURUSD 0.4935, XAUUSD 0.5035, GBPUSD 0.5014
— flat, consistent with the microstructure-only run.

### §9 Target / §10 Horizon

Not re-run — this phase tests new *features* against the same target and
horizon `FEASIBILITY_REPORT.md` already audited. Nothing here motivates
revisiting them: no new feature comes close enough to the noise floor to
suggest the target framing is hiding a real relationship.

### §11–12 Stability

Microstructure-only: fold-stable (`stable across folds: True`) but at chance
— stability without an effect is not evidence.

Microstructure + cross-asset: **not** fold-stable (`spread 0.0330`,
folds inconsistently above 0.5) on top of also not beating the noise floor —
two independent reasons to reject, not one borderline reason.

### §13 Decision gate: **RED**, both runs

```
new-information beats permutation null : False  (both runs)
OLD+NEW improves over OLD alone         : False  (both runs; OLD+NEW is
                                                   always slightly WORSE than
                                                   OLD alone — added columns,
                                                   added noise, no signal to
                                                   offset it)
```

## 13b. Final Verdict

**RED.** Two genuinely independent information-source batches — one native
to the broker feed (spread, volume), one external market context (DXY, US
rates, silver, oil) — were tested with full leakage and holdout-isolation
proof, and neither produced a single feature that beat its own
block-permutation null. `OLD + NEW` was worse than `OLD` alone in both runs.
This is not "weak", it is the same absence the original feasibility gate
found, now also checked in the two most plausible places new information
could have been hiding.

## 14. Holdout Isolation — What Was Actually Verified

`scripts/information_discovery.py --verify-holdout-isolation` corrupts the
holdout region (`×97 +13` on every field) and rebuilds the research dataset,
asserting byte-identical rows, features, and labels. Three real bugs surfaced
and were fixed while building this check, not merely documented as risks:

1. **Corrupting a research row's own decision time is wrong.** A barrier
   label reads up to `horizon` bars *forward* of its own decision — that is
   normal labelling, not leakage. The first version corrupted those bars and
   falsely reported a violation on every run.
2. **Deriving the boundary from resolved rows misses unresolved ones.** A
   decision near the cutoff that had not yet touched TP or SL in the real
   data can flip to "resolved" when its forward window is corrupted,
   manufacturing a new row inside the research region. Fixed by computing the
   boundary from every *possible* decision under the cutoff, not only the
   ones that happened to produce a row.
3. **Splitting by resolved-row count is itself holdout-dependent.**
   `cut = int(len(y_all) * frac)` — what `feasibility_gate.py` uses — makes
   the split point a function of how many holdout decisions happen to
   resolve. Corrupt the holdout, change that count, and a handful of
   borderline rows change sides. Fixed here by splitting on a **timestamp**
   computed from raw candle span before any labelling occurs
   (`research_cutoff_time`), which cannot move regardless of what happens to
   holdout candle values.

Current status: **VERIFIED** on synthetic data (`tests/test_holdout_isolation.py`,
6 tests, including a non-vacuity control proving the check can fire on a
genuine research-region corruption). Must be re-verified against real data
before trusting any experimental result — the script does this automatically
before running anything else, and refuses to report a verdict if it fails.

## 15. What Remains Untested

Both real-data runs completed 2026-08-17 and both returned RED (§13b).
Everything below this line was true before those runs and is unchanged by
them — no source tested here was expected to fix the original gate's
finding, and none did.

What is left, honestly ranked by how likely it is to matter:

1. **Historical economic calendar / news / sentiment** — still not available
   (§4). This is the one category of information genuinely untested, because
   it does not exist in this repository in archived form. Acquiring it means
   a paid historical calendar feed or a news archive built forward from now,
   not a code change.
2. **A different timeframe (M15/M30)** — not fetched or tested anywhere in
   this arc. `FEASIBILITY_REPORT.md` §8 found holding times mostly longer
   than 2 bars, which argued against an obvious resolution mismatch, but
   that argument was made on H4/H1 only.
3. **A gold-only model with its own full pipeline** — XAUUSD was the one
   instrument to score above 0.50 in earlier investigation phases, though
   never survived direction-removal or noise-floor checks on its own. Testing
   it properly means its own walk-forward validation, not reading the
   per-symbol column of a pooled model.
4. **Accepting the negative result.** Two independent, leakage-proven
   information searches found nothing. The rule-based layer plus the six
   trade-management layers already in production do not need an ML entry
   filter to operate; a filter with no demonstrated edge only adds the risk
   of blocking real trades on noise.

Per the governing rule: no Optuna, no full XGBoost, no production model
change on this framework. If a future session pursues #1–#3, it goes through
this same holdout-isolation-proven pipeline before any training step.
