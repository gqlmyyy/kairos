# PHASE 1 — DATA FOUNDATION REPORT

Scope: data only. No training, no labels, no model changes, no live-behaviour
changes. Everything below is reproducible:

```text
fetch:     python scripts/fetch_training_candles.py --symbols EURUSD GBPUSD XAUUSD \
               --timeframes H4 H1 M15 --start 2021-09-05
validate:  python scripts/phase1_validate.py
tests:     python -m pytest tests/test_phase1_data_integrity.py
```

## 1. What was inspected before anything was written

* `data/market/mt5_session.py` — the single owner of the MT5 session (all
  fetches go through `ensure_session` / `mt5_call`).
* `scripts/fetch_training_candles.py` — the existing raw-candle exporter
  (completed candles only, spread + real_volume captured, manifest per file).
* `analysis/features/timeframe_alignment.py` — the existing, tested gap
  engine (`validate_series`, `classify_gaps`, `diagnose_grid`).
* `scripts/validate_m30_candles.py` / `scripts/validate_real_dataset.py` —
  the existing Gate-1 integrity precedent (unexplained gaps BLOCK).
* `analysis/research/candles.py` — the readers of `data/historical/`
  (`KairosHistoricalSource`), which pinned the storage format.
* `analysis/baseline/vendor/config/*` — the vendored sources/timeframes/
  session-rules philosophy (evidence-based classification, no invented hours).

## 2. Decisions (and why)

1. **Extend, don't fork.** The existing exporter, gap engine, readers and
   manifest convention were kept and extended additively. Nothing that
   consumed `data/historical/` needed to change.
2. **Storage layout kept flat**: `data/historical/<SYMBOL>_<TF>.json` +
   `manifest.json`. The SYMBOL → TIMEFRAME separation the phase requires is
   the file-naming contract every existing reader already uses; a parallel
   `SYMBOL/TF/` tree would have forked the loaders and duplicated ~40 MB.
3. **Direct MT5 source for every timeframe — no resampling.** H1/H4 come
   straight from the terminal (`copy_rates_range`), which the phase prefers
   over deriving H1/H4 from M15. No synthetic candle exists anywhere in the
   store; no forward-fill, no interpolation.
4. **The row schema is unchanged** (`t, open, high, low, close, volume,
   spread, real_volume`; `t` = UTC epoch seconds, bar OPEN; `volume` =
   MT5 tick_volume). The phase's canonical names map onto it via
   `analysis.data.phase1.CANONICAL_FIELD_MAP`; symbol/timeframe identity
   lives in the filename + manifest, not repeated per row.
5. **Gap taxonomy** maps the existing engine's categories onto the phase's:
   `weekly_close`->WEEKEND, `known_holiday`/`broker_maintenance`->MARKET_CLOSED,
   SUSPICIOUS_GAP->UNKNOWN (small, honest ignorance, non-blocking),
   DATA_ERROR->DATA_GAP (large, unexplained — BLOCKS PASS).
6. **Terminal depth is measured, not assumed.** `fetch_range` binary-searches
   the earliest start the terminal answers for each timeframe and records
   `depth_limited` + `requested_start` vs `actual_start` per file. Shortfalls
   are reported INCOMPLETE, never filled.

## 3. Verified data-source facts (probed directly against the terminal)

* H1/H4: 5 full years available (2021-09-06 -> fetch date) for all 3 symbols.
* M15: the terminal's history depth ends at **2023-10-30** (~2.85 years).
  `copy_rates_from_pos` also rejects large counts ("Invalid params"), which
  is why the range-based fetch path exists.
* The broker's own history has holes (see §5) — same dates across symbols,
  verified absent on a fresh terminal round-trip, i.e. broker-side, not a
  fetch bug.

## 4. The nine datasets (validated 2026-09-05)

| dataset | rows | start (UTC) | end (UTC) | coverage | W | MC | DG | U | max gap | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| EURUSD M15 | 70,727 | 2023-10-30 04:45 | 2026-09-04 23:45 | 2.85y INCOMPLETE | 147 | 9 | **6** | 6 | 288 bars (NY 2024) | FAIL |
| EURUSD H1 | 31,110 | 2021-09-06 00:00 | 2026-09-04 23:00 | 5.00y COMPLETE | 259 | 7 | **3** | 2 | 72 bars (NY 2024) | FAIL |
| EURUSD H4 | 7,789 | 2021-09-06 00:00 | 2026-09-04 20:00 | 5.00y COMPLETE | 259 | 6 | 0 | 2 | 18 bars | PASS |
| GBPUSD M15 | 70,726 | 2023-10-30 04:45 | 2026-09-04 23:45 | 2.85y INCOMPLETE | 147 | 9 | **7** | 5 | 288 bars (NY 2024) | FAIL |
| GBPUSD H1 | 31,109 | 2021-09-06 00:00 | 2026-09-04 23:00 | 5.00y COMPLETE | 259 | 7 | **3** | 3 | 72 bars (NY 2024) | FAIL |
| GBPUSD H4 | 7,789 | 2021-09-06 00:00 | 2026-09-04 20:00 | 5.00y COMPLETE | 259 | 6 | 0 | 2 | 18 bars | PASS |
| XAUUSD M15 | 66,859 | 2023-10-30 04:45 | 2026-09-04 22:45 | 2.85y INCOMPLETE | 143 | 543 | **57** | 11 | 296 bars (Good Fri 2024) | FAIL |
| XAUUSD H1 | 29,461 | 2021-09-06 01:00 | 2026-09-04 22:00 | 5.00y COMPLETE | 250 | 1040 | **2** | 3 | 74 bars (Good Fri 2024) | FAIL |
| XAUUSD H4 | 7,741 | 2021-09-06 00:00 | 2026-09-04 20:00 | 5.00y COMPLETE | 250 | 16 | 0 | 1 | 18 bars | PASS |

W = WEEKEND, MC = MARKET_CLOSED, DG = DATA_GAP, U = UNKNOWN.
Zero duplicate timestamps, zero out-of-order, zero off-grid, zero invalid or
non-positive OHLC, zero missing fields — on all nine datasets.

## 5. Why six datasets FAIL — the honest findings

1. **Broker feed holes (real DATA_GAP).** A handful of multi-hour windows are
   absent from the broker's own history on WEEKDAYS, on the same dates across
   symbols: 2024-07-02/03, 2025-01-07, 2025-07-03, 2025-10-23,
   2026-02-26/27, 2026-03-12. Verified against a fresh terminal round-trip:
   e.g. EURUSD M15 holds 1 bar where 48 belong on 2024-07-02 12:00->00:00,
   while XAUUSD traded normally through the same window (48/48 bars) — so it
   is an FX-feed hole at the broker, not a market closure and not a fetch
   bug. They cannot be filled from this source. Full list with per-gap rule
   audits: `reports/phase1/data_integrity_report.json`.
2. **XAUUSD daily-maintenance size drift (57 M15 gaps).** Gold's daily break
   follows the US session; in DST-transition weeks its UTC size is 4 bars
   while the series-wide median at that hour is 8 (MAD 0 -> tight band). The
   era-blind band rejects them as unexplained. The data is real; the
   classifier's per-hour band is era-blind. Left as-is deliberately —
   widening the band would be tuning the classifier to pass, which is what
   Phase 1 must not do. Phase 2 can make the band DST-era-aware WITH this
   report as evidence.
3. **M15 coverage.** ~2.85y (2023-10-30 ->) vs the 5-year target: the
   terminal simply does not hold deeper M15. Marked INCOMPLETE, unfilled.
   Deeper M15 needs a second source (the vendored config documents a
   Dukascopy path) — explicitly out of scope here per the single-source rule.

## 6. Tests

* New: `tests/test_phase1_data_integrity.py` — 32 tests: schema, OHLC
  validity (incl. the non-positive check the engine lacks), ordering,
  duplicates, grid alignment, the full gap taxonomy with synthetic
  weekend/holiday/thin/data-gap fixtures, dataset isolation,
  manifest-merge reproducibility, reader compatibility, and the fetch-script
  extension contract.
* Modified (with reasons, per the phase rules):
  * `test_diagnose_entry_gate.py` — the "MT5 absent here" guard now SKIPS on
    machines where MT5 is installed (this data machine) instead of failing;
    the headless behaviour it guards is still exercised.
  * `test_research_missing_policy.py` / `test_research_offline_replay.py` —
    these rode on the old data/historical snapshot being pre-spread; Phase 1
    re-fetched every file WITH spread (a fix, not a regression), so the
    no-spread condition is now built explicitly from the golden fixtures.
    The policy under test is unchanged.
* Full suite: **1459 passed, 3 skipped, 0 failed.**

## 7. Files

Created: `analysis/data/__init__.py`, `analysis/data/phase1.py`,
`scripts/phase1_validate.py`, `tests/test_phase1_data_integrity.py`,
`reports/phase1/data_integrity_report.json`,
`reports/phase1/metadata/<SYMBOL>_<TF>.json` (9), this report.
Modified: `scripts/fetch_training_candles.py` (additive: `--start` range
fetch + terminal-depth binary search, `--timeframes`, manifest merge,
M15 support), the three test files above, `data/historical/*` (6 files
re-fetched to 5 years, 3 new M15 files) and `data/historical/manifest.json`
(merged, enriched with source/schema/timezone/coverage/validation).

## 8. Phase 1 status

**FAIL — deliberately not papered over.** 6 of 9 datasets carry unexplained
DATA_GAP windows (broker-side holes + XAUUSD DST-era pause drift), and all
three M15 datasets are INCOMPLETE (~2.85y) against the 5-year target. The
foundation itself (fetch -> validate -> report, metadata, tests, reproducible
re-runs) is complete and green; the failures are data-quality facts that
Phase 2 must either accept (gap-aware labelling already exists in the
vendored target config) or solve at the source. No dataset is claimed
complete that is not.
