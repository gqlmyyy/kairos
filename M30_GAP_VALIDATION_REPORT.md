# XAUUSD M30 gap validation — investigation and fix

Scope: Gate 1 (Data Integrity) only. No model, no training, no labels, no
thresholds, no trading logic touched. `data/historical/XAUUSD_M30.json`
lives only on the user's Windows machine — nothing here was run against the
real file; it was verified on synthetic fixtures built to reproduce the
same weekly-close + daily-rollover-pause pattern a real MT5 XAUUSD M30 feed
shows, and the user still needs to run the new script against the real file
(command at the end).

## 1. The "downstream xgbooost validator" does not exist in this repository

Searched the whole tree for the literal strings `SUSPICIOUS_GAP`,
`DATA_ERROR`, `EXPECTED_MARKET_GAP`, `gap_type`, `classify_gap` — zero
matches. The only two files with "validate" or "xgboost" in the name
(`scripts/validate_real_dataset.py`, `analysis/entry_v2/entry_deployment_validator.py`)
were read in full: neither classifies gaps at all.
`validate_real_dataset.py`'s only gap-related logic
(`analysis/features/timeframe_alignment.py::validate_series`, before this
fix) counted total gaps and the single largest gap — no per-gap
classification, no category names, and it hard-codes `TIMEFRAMES = ("H4",
"H1")`, so it does not even look at M30 files.

This is consistent with the two discrepancies already flagged in the
previous turn (no `main.py --tf` CLI, no matching experiment-ID format):
the "downstream xgbooost validator" is very likely a separate tool the user
has outside this repository, not something built in this session. If it
does live in this repo somewhere outside GitHub-tracked history, its source
was not found — worth confirming.

## 2. What was almost certainly happening

An M30 XAUUSD series over ~3.25 years has, structurally, two recurring gap
patterns that are entirely normal, not errors:

* **The weekly close.** FX/CFD brokers close Friday evening and reopen
  Sunday evening — one gap per week, ~169 times over 3.25 years.
* **A daily rollover pause.** Many brokers pause briefly once a day
  (commonly around end-of-day server time) to roll the daily bar and
  compute swap. On a 30-minute grid that is a 1-2 bar gap, once per trading
  day — up to ~800 times over the same period.

A validator that reports only **2** `EXPECTED_MARKET_GAP` over 3.25 years
cannot be recognizing the weekly close at all (there should be roughly
169), which means its "expected" bucket is almost certainly matched against
a fixed list of calendar dates (e.g. two hard-coded holidays) rather than
against the weekly pattern. Everything the weekly-close and daily-pause
patterns would otherwise explain — hundreds of routine gaps — falls through
into `SUSPICIOUS_GAP` (664) and `DATA_ERROR` (184) instead. The `664 + 184
= 848` total is the right order of magnitude for "every weekly close plus
every daily pause, none of them recognized as expected."

None of this can be confirmed without the external tool's source or the
real gap list, so it is presented as the most likely explanation, not a
proven one. Section 5 below is how to get a confirmed answer.

## 3. The actual gap in KAIROS's own tooling

Independent of what the external validator does, `timeframe_alignment.py`
had no gap *classification* at all — only a count — and
`validate_real_dataset.py` never validates M30. That is a real deficiency
for the M30 track's own Gate 1, fixed here regardless of the external
tool's behavior.

## 4. The fix

**`analysis/features/timeframe_alignment.py`** — two additions:

* `diagnose_grid()` — moved here from `validate_real_dataset.py` unchanged,
  so the calendar logic below can share it (a fixed-UTC-hour classifier
  would silently mis-classify a broker whose clock is not on UTC).
* `classify_gaps(candles, timeframe)` — classifies every gap as
  `EXPECTED_MARKET_GAP`, `SUSPICIOUS_GAP`, or `DATA_ERROR`, calendar-first
  rather than duration-first:
  * **Weekly close**: starts Friday afternoon/evening (>= 12:00 UTC, so a
    Friday-morning outage that happens to still be down at the weekend is
    not confused with the market actually closing), ends Saturday, Sunday
    or Monday (a holiday long weekend included, without a separate
    per-holiday rule).
  * **Named holiday**: overlaps Christmas Day or New Year's Day, for the
    case where the holiday itself is not a Friday.
  * **Recurring daily pause**: detected from the series' own data, not a
    hard-coded hour — if a UTC hour accounts for a large share (>= 40%) of
    the remaining unclassified weekday gaps, *and* it recurs at least 3
    times (a share threshold alone is vacuous on one occurrence), every gap
    at that hour is one explained, recurring event.
  * Everything else small (<= 2 missing bars) is `SUSPICIOUS_GAP`, not
    blocking. Everything larger is `DATA_ERROR` — **fails closed**,
    unchanged from the existing philosophy in this codebase.

  Matching by day-of-week rather than by an exact expected duration is
  deliberate: broker close/open minutes vary, and an MT5 server clock that
  follows exchange DST shifts every session-boundary hour by an hour twice
  a year. A duration-matching classifier would mis-judge real weekend
  closes for roughly half the year; this one does not depend on the hour at
  all for the weekly-close and holiday rules, and *learns* the relevant
  hour from the data itself for the daily-pause rule.

**`scripts/validate_real_dataset.py`** — the local `diagnose_grid` was
replaced with an import of the moved function. No other line changed; H4/H1
behavior is identical to before.

**`scripts/validate_m30_candles.py`** (new) — Gate 1 only, for M30,
deliberately independent of `validate_real_dataset.py`'s dataset-build path
(that path is wired to `train_entry_model.build_dataset`, which is an
H4-decision + H1-context construction that does not apply to M30 considered
on its own, per the M30 track's own design — M30 does not feed or get fed
by H4/H1). Checks, in order: provenance, timestamp grid integrity
(`validate_series` + `diagnose_grid`), completed-candle policy, field
usability (`spread`/`real_volume`/`tick_volume`, reusing
`microstructure_features.field_availability` — an all-zero `real_volume` is
reported as `AVAILABLE (constant — no information)` and explicitly called
out as **not a defect**, never as an error), and gap classification. Exits
1 on any `DATA_ERROR` gap, or on structural problems (misaligned grid,
duplicate timestamps, bad OHLC, a forming candle in the export). Trains,
labels and installs nothing.

## 5. Tests

`tests/test_timeframe_alignment.py` — 12 new tests (60 total, up from 48;
all pass), built on real 2024 calendar dates since the classifier reasons
about actual weekdays:

* the weekly close is accepted, including one that runs through a Monday
  holiday
* a named holiday landing on a weekday (Christmas 2024, a Wednesday) is
  accepted
* a recurring daily pause (one bar dropped at the same UTC hour on ten
  separate weekdays) is accepted as one explained pattern, not ten
* a **single** small weekday gap is `SUSPICIOUS_GAP`, not silently accepted
  as "recurring" from one data point
* a large, non-recurring, mid-week gap with no calendar excuse still fails
  closed as `DATA_ERROR` — the actual regression guard for "don't whitelist
  everything"
* a Friday-*morning* outage that happens to end on a Saturday is **not**
  confused with the weekly close (the afternoon/evening-start requirement)
* classification reads only timestamps — mutating OHLC values changes
  nothing (no dependency on price data, no lookahead surface)
* `diagnose_grid` correctly reports a constant broker offset, and correctly
  does *not* claim a whole-hour offset is invisible on a 30-minute grid
  (it isn't — a hidden bug the naive version of this test would have missed)

## 6. Verified on a synthetic reproduction of the real pattern

Built a 39,780-bar synthetic M30 series over the same span as the real
file (2023-05 → 2026-08), with a Friday-21:00-UTC-to-Sunday-22:00-UTC
weekly close and a daily 21:00 UTC rollover pause on every weekday —
i.e. the two patterns section 2 describes, nothing else:

```
5. GAP CLASSIFICATION
  XAUUSD_M30: 850 total gaps over 1190 days (39780 bars)
    EXPECTED_MARKET_GAP  850
    SUSPICIOUS_GAP       0
    DATA_ERROR           0
  ok    XAUUSD_M30: 0 DATA_ERROR gaps
GATE 1 PASSED
```

Then deliberately deleted 3.5 hours of bars on an unrelated midweek
Wednesday, to check the fix still fails closed on a real problem:

```
5. GAP CLASSIFICATION
  XAUUSD_M30: 851 total gaps over 1190 days (39774 bars)
    EXPECTED_MARKET_GAP  850
    SUSPICIOUS_GAP       0
    DATA_ERROR           1
  FAIL  XAUUSD_M30: 1 DATA_ERROR gaps — genuinely unexplained, listing every one below for audit:
    2024-06-12T14:30:00+00:00 (Wed) -> 2024-06-12T18:00:00+00:00 (Wed)  missing=6 bars  3.5h  reason: not the weekly close, not a known holiday, not a recurring daily pause, and too large to be thin liquidity
GATE 1 FAILED
```

Both outcomes are what Gate 1 is supposed to do: pass clean, ordinary
market structure, and fail loudly — with the exact timestamp — on a genuine
hole.

## 7. What is still needed from the real file

This was verified on a synthetic reconstruction of the pattern, not the
real `data/historical/XAUUSD_M30.json` — that file only exists on the
user's machine. Run:

```bash
git pull
python scripts/validate_m30_candles.py --symbols XAUUSD
```

This reports the real `EXPECTED_MARKET_GAP` / `SUSPICIOUS_GAP` /
`DATA_ERROR` counts and, for every `DATA_ERROR` gap, its exact UTC start
and end timestamp, weekday, size and the reason it could not be explained.

If it comes back with 0 `DATA_ERROR` gaps: Gate 1 passes for M30, and the
184-gap failure was very likely the external validator's classification
gap described in section 2 — the underlying data was fine.

If some `DATA_ERROR` gaps remain: they are printed individually, which is
enough to tell whether they cluster on a specific date (a broker outage,
worth checking against that broker's status page), a specific hour not
covered by the recurring-pause detection (the threshold or hour-matching
here can be adjusted once real examples are visible), or scattered with no
pattern (which would point at something in the fetch itself worth a second
look). None of that can be diagnosed further without seeing the actual
list.

## 8. Boundary respected

Not touched: `models/entry/`, `analysis/models/entry_feature_spec.py`,
`scripts/train_entry_model.py`'s labelling/dataset logic, any threshold,
any risk/trading code, H4/H1 behavior in `validate_real_dataset.py`. This
is Gate 1 only, and Gate 1 is not yet confirmed to pass on the real file.

---

## Round 2 — the real gap list came back, and it found real bugs in this fix

Section 7 asked for the real output. It came back: 38,400 real bars,
850 raw gaps, `EXPECTED_MARKET_GAP: 696 / SUSPICIOUS_GAP: 48 / DATA_ERROR:
106`, a forming-candle failure, and — critically — the actual `DATA_ERROR`
examples. Those examples are what made this round possible: three genuine
bugs, all in round 1's own code, none of them the external validator's
fault this time.

### Root cause 1 — the forming-candle export was real, and round 1's own
`check_completed` diagnosis was correct about the symptom

The exported newest M30 bar (22:00 → 22:30 UTC) had not closed by the time
`fetch_training_candles.py`'s manifest claimed the data was pulled. The
manifest's `fetched_at` was captured **once**, before the whole
multi-symbol, multi-timeframe loop started — so a run that took long enough
compared a file against the wrong clock reading, and `rates[:-1]` assumed
MT5 always hands back exactly one still-forming bar in a known array
position.

**Fix**: `fetch()` in `scripts/fetch_training_candles.py` now filters by
`open_time + span <= real UTC now` (`_closed_by`) instead of dropping array
position `-1`. Over-requests 3 extra bars and keeps whichever are actually
closed, then trims to the requested count. Every manifest file entry now
also carries its own `fetched_at` and `forming_candles_dropped`, and
`check_completed` in both `validate_m30_candles.py` and
`validate_real_dataset.py` compares a file against *its own* pull time
first, falling back to the batch timestamp only if a file entry lacks one
(files written before this fix). H4/H1 behavior is unaffected except that
this same false-positive class is now closed for them too.

### Root cause 2 — the daily-maintenance-pause rule had a hidden size ceiling

Round 1's `classify_gaps` accepted a recurring daily pause only up to
`suspicious_max_missing_bars * 2` = **4 bars**. The real broker's daily
pause is **5 bars** (22:30→01:00, 2026-era) or **8 bars** (21:00→01:00,
2023-era) — both larger than that ceiling, so every single instance of the
routine, recurring, evidenced daily pause was falling through to
`DATA_ERROR`. This one bug alone accounts for most of the 106.

**Fix**: the ceiling is gone. `broker_maintenance` now accepts any size
within tolerance of that hour's own **median** size in this series
(`_typical_missing_bars`, ±50% + 1 bar slack) — consistency of hour and
size across many days is the evidence a pause is routine, not an arbitrary
absolute bar count. Verified with two direct regression tests using the
report's own numbers: a 5-bar pause at 22:30 UTC and an 8-bar pause at
21:00 UTC are both now accepted (`test_the_recurring_2026_style_daily_pause_is_accepted_at_its_real_size`,
`test_a_larger_historical_daily_pause_is_also_accepted`).

### Root cause 3 — only one daily-close hour was ever recognized, across a
series that has (at least) two

Round 1 picked a single "most common hour" globally. A multi-year MT5
series whose server clock follows exchange DST — or whose broker simply
changed its maintenance schedule at some point, as this one evidently did
between the 2023 (21:00 UTC) and 2026 (22:30 UTC) portions of the real
report — has **two** legitimate recurring hours, not one. Whichever season
lost the "most common" contest got none of its gaps recognized as routine
at all, forcing them into `SUSPICIOUS_GAP` or `DATA_ERROR` regardless of
how consistently they recurred.

**Fix**: `_daily_close_hours` now returns every hour that clears a
recurrence floor (default 20 occurrences), not just the single most common
one — `daily_close_hours_utc` in the result can (and, on this data,
does) list more than one hour. Verified with
`test_dst_produces_two_recognized_daily_close_hours_not_one`, which builds
a fixture with two separate hour-clusters and confirms both are recognized
independently.

### A fourth thing that was imprecise, not wrong: gap-boundary semantics

Round 1's `gap_start_t` was the *last present* candle's own timestamp; the
real report describes gaps as "22:30 → 01:00" — the *missing* window,
i.e. the last present candle's **close**. `gap_start_t` is now
`prev_t + span` (first missing bar's open), and `duration_hours` is now
exactly `missing_bars * span`, which reproduces the report's own numbers
bar-for-bar. This wasn't a classification bug, but it made every printed
gap boundary off by one bar from how a human (or the external tool) reads
them, which matters for a field this document asks to be auditable.

### Evidenced holiday calendar, not a textbook one

The 2023 examples (Memorial Day, Juneteenth, Independence Day, Labor Day,
Thanksgiving) and the 2024/2025/2026 Good-Friday-adjacent long weekends are
now recognized by `us_market_holidays(year)` — computed per year (Good
Friday via the Meeus/Jones/Butcher Easter algorithm, Memorial
Day/Labor Day/Thanksgiving via nth-weekday-of-month arithmetic), not
hard-coded to any single year, so it holds across the full 3+ year span
and beyond. Deliberately **not** the full textbook NYSE holiday list (no
MLK Day, no Presidents' Day) — every entry is backed by an actual gap in
the real report; an unlisted holiday still gets a fair chance at the
weekly-close or daily-maintenance rules before falling through to
`DATA_ERROR`, which is the intended fail-closed behavior for something
genuinely never seen before. Verified against the report's own three
Good-Friday examples by exact timestamp, parametrized across all three
years (`test_good_friday_is_recognized_in_every_year_of_the_dataset`).

### What is still enforced, unchanged from round 1

A recognized daily-close hour is not a blanket excuse — a gap at that hour
whose size falls outside the tolerance band around the typical size, and
that does not overlap a known holiday, still fails closed as `DATA_ERROR`
(`test_an_outlier_at_the_recognized_hour_is_not_silently_absorbed`). No
gap is whitelisted by proximity to an explained one; every classification
still has to individually pass one of the three explained rules.

### Verified on a reconstruction of the real report's exact shape

Built a synthetic 2023-05-11 → 2026-08-17 M30 series combining everything
the real report showed: a 4-hour (8-bar) daily pause at 21:00 UTC through
2024, a 2.5-hour (5-bar) pause at 22:30 UTC from 2025 on, ordinary weekly
closes, and the evidenced US holidays.

```
5. GAP CLASSIFICATION
  XAUUSD_M30: 826 total gaps over 1194 days (35934 bars)
  recognized daily-close hour(s) UTC: ['21:00', '22:00']
    EXPECTED_MARKET_GAP  826
    SUSPICIOUS_GAP       0
    DATA_ERROR           0

  by reason:
    EXPECTED_MARKET_GAP  reason=broker_maintenance               644
    EXPECTED_MARKET_GAP  reason=known_holiday                    26
    EXPECTED_MARKET_GAP  reason=weekly_close                     156
  ok    XAUUSD_M30: 0 DATA_ERROR gaps
GATE 1 PASSED
```

Then removed 8 real hours of bars on an unrelated Wednesday to re-confirm
fail-closed behavior wasn't lost in the process:

```
  FAIL  XAUUSD_M30: 1 DATA_ERROR gaps — genuinely unexplained, listing every one below for audit:
    2025-08-13T15:00:00+00:00 (Wed) -> 2025-08-13T19:00:00+00:00 (Wed)  missing=8 bars  4.0h  reason=unexplained_missing_candles
GATE 1 FAILED
```

### Files changed, round 2

* `scripts/fetch_training_candles.py` — `_closed_by`, rewritten `fetch()`,
  per-file manifest provenance.
* `analysis/features/timeframe_alignment.py` — `classify_gaps` rewritten:
  fixed gap-boundary semantics, `_daily_close_hours` (multi-hour),
  `_typical_missing_bars` (no hard cap), `us_market_holidays` +
  `_easter_sunday`/`_nth_weekday`/`_last_weekday` (evidenced, per-year
  calendar), machine-readable `reason` codes matching the brief exactly
  (`weekly_close`, `broker_maintenance`, `known_holiday`,
  `thin_liquidity_or_irregular_session`, `unexplained_missing_candles`).
* `scripts/validate_m30_candles.py` — per-file `fetched_at` preference,
  prints `daily_close_hours_utc` and a reason breakdown.
* `scripts/validate_real_dataset.py` — same per-file `fetched_at`
  preference for H4/H1 (same false-positive class, same fix, no behavior
  change for correctly-timed data).
* `tests/test_fetch_training_candles.py` (new) — `_closed_by` unit tests,
  MT5-free.
* `tests/test_timeframe_alignment.py` — `TestClassifyGaps` rewritten:
  18 tests total including the two size-regression tests, the DST
  dual-hour test, the three parametrized real-date Good-Friday tests, and
  the outlier-is-not-absorbed guard.

### Test results, round 2

`pytest -q`: **945 passed**, 1 pre-existing unrelated failure
(`test_predict_proba_parity.py` — exit-model artifact/schema feature-count
mismatch, nothing to do with M30 or this fix, present before this work
started).

### Still needed from the real file

This round was verified on a synthetic reconstruction of the real report's
own numbers, not the real file itself. Re-run:

```bash
git pull
python scripts/fetch_training_candles.py --years 3 --symbols XAUUSD
python scripts/validate_m30_candles.py --symbols XAUUSD
```

Expected, if the diagnosis above is complete: the forming-candle failure
is gone (the fetch itself no longer exports one), and the `DATA_ERROR`
count drops from 106 toward 0 — remaining ones, if any, are printed
individually with their exact UTC timestamp, weekday and size, which is
what determines whether anything past this point is a new, previously
unseen broker behavior worth a fourth root-cause rule, or genuine data
corruption.

Boundary unchanged: no model, training, label, threshold or trading-logic
code touched in round 2 either.
