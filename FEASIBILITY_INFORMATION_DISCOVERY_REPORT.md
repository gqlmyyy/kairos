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

**Experimental results below are PENDING.** The candle files under
`data/historical/` in the environment that produced `FEASIBILITY_REPORT.md`
were fetched before `spread`/`real_volume` capture existed. Re-run
`scripts/fetch_training_candles.py` (now updated) to regenerate them, then run
`scripts/information_discovery.py` — see §15.

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

## 7–13. Experimental Results

**PENDING** — require candle files with `spread`/`real_volume` populated,
i.e. a fresh run of `scripts/fetch_training_candles.py` on the Windows
machine. `scripts/information_discovery.py` reports, once run:

- §7 Direction-free results (old-only / new-only / old+new, walk-forward)
- §8 Permutation results per new feature source
- §9 Symbol-specific results (EURUSD / GBPUSD / XAUUSD independently)
- §10 Target audit (unchanged from Phase 4 — no new target is introduced here)
- §11 Walk-forward results
- §12 Stability results
- §13 Decision gate (GREEN / YELLOW / RED)

Verified end-to-end on synthetic data with injected spread/volume (pure
noise, uncorrelated with outcomes by construction): the pipeline runs
without error, correctly reports every new feature as failing its
permutation test, and correctly returns RED — confirming the machinery
behaves as designed before being pointed at real data.

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

## 15. Recommended Next Experiment

```
# Windows machine, MT5 running and logged in
python scripts/fetch_training_candles.py            # now captures spread/real_volume
python scripts/information_discovery.py --verify-holdout-isolation
python scripts/information_discovery.py --permutations 200 --json research/information_discovery/information_audit.json
python scripts/information_discovery.py --cross-asset --permutations 200
```

The first `information_discovery.py` call re-verifies holdout isolation
against real data before anything else runs and refuses to proceed if it
fails. If it passes, the microstructure-only run answers whether spread/volume
carry anything; the `--cross-asset` run additionally attempts DXY/yields/
silver/oil (network-dependent, best-effort, reports what actually fetched).

Per the governing rule: no Optuna, no full XGBoost, no production model
change until one of these runs returns GREEN or a defensible YELLOW — and even
then, the next step is one fixed-hyperparameter XGBoost fit compared against
the logistic baseline already computed here, not a hyperparameter search.
