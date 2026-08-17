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

---

## Round 3 — the real gap list came back again: 132 DATA_ERROR, forming
candle still present, and a genuine midnight-boundary bug in round 2's own fix

Real output this time: 38,400 bars, `EXPECTED_MARKET_GAP: 716 / SUSPICIOUS_GAP:
2 / DATA_ERROR: 132`, plus the forming-candle check still failing at
22:00→22:30 UTC. The user flagged two recurring `DATA_ERROR` examples that
were obviously the weekly close misclassified — `Saturday 00:00 UTC ->
Monday 01:00 UTC` (49h) and its 48h DST variant — plus five scattered,
irregularly-sized gaps to investigate and explicitly NOT auto-whitelist.

### Root cause 4 — the weekly-close rule breaks when the close lands exactly
on a midnight boundary

Round 2's `gap_start_t` is `prev_t + span` — the first MISSING bar's open.
When the last present candle is Friday 23:30, that first missing bar opens
at **Saturday 00:00**, so `gap_start_t`'s own `.weekday()` is Saturday, not
Friday. Pass 1 (`weekly_close`) checked `start_weekday == 4` (Friday) — so
every single instance of this broker's actual weekly close, which happens
to land exactly at the day boundary, silently failed the rule and fell
through to `DATA_ERROR`. This is almost certainly the largest single
contributor to the 132: two clean, real, recurring examples were handed to
us, and both were exactly this.

**Fix**: classification is now keyed on a new `stop_weekday`/`stop_hour_utc`
pair, computed from `prev_t` — the **last present** candle — instead of the
first missing one. Friday 23:30 is unambiguously Friday regardless of which
side of midnight the next missing bar's timestamp falls on. `start_weekday`
stays first-missing-bar-based for display (that's how the report itself
reads a gap), but every classification decision (`_daily_close_hours`, the
weekly-close check, the broker-maintenance weekday filter) now reads
`stop_weekday`. Verified against both of the report's own examples by exact
timestamp (`test_a_weekly_close_landing_exactly_on_the_midnight_boundary_is_accepted`,
`test_a_dst_shifted_48_hour_variant_of_the_same_close_is_also_accepted`) —
both now classify as `weekly_close`.

The Friday-afternoon/evening hour guard (`_FRIDAY_CLOSE_EARLIEST_HOUR_UTC`)
also moved to `stop_hour_utc` for the same reason, and — since it no longer
needs `daily_close_hours` to already exist as a prerequisite — pass 1 is
simpler than round 2's version: weekday structure (last trade Friday,
resume weekend/Monday) is sufficient on its own, once it isn't reading the
wrong day's weekday.

### The daily-maintenance tolerance band was loosened where it should have
been tightened

Round 2's `broker_maintenance` acceptance band was `median ± 50% + 1 bar` —
generous enough that a genuinely irregular gap landing on a recognized hour
could plausibly be waved through if its size happened to fall inside that
wide a band. The user's explicit instruction ("do NOT automatically
classify these as expected") for five scattered real examples made this
worth tightening rather than leaving as-is.

**Fix**: the band is now `median ± max(2 × MAD, 1)` bars — driven by how
much this broker's pause size *actually varies* at that hour
(`_pause_size_stats`, median absolute deviation), not a fixed percentage. A
broker whose pause is always exactly N bars gets a tight band; a genuinely
jittery one gets a band sized to the jitter actually measured. Verified with
`test_reported_irregular_gaps_are_not_auto_whitelisted_by_a_nearby_recognized_hour`,
which reproduces the report's `2026-02-26 22:00 -> 2026-02-27 03:00` example
against a backdrop where 22:00 UTC is a real, tight, 2-bar daily pause — the
outlier (10 bars) is far outside tolerance and correctly stays `DATA_ERROR`.

All five of the report's flagged irregular examples were checked in
isolation against this fix (no established pattern to lean on) and all five
correctly still classify as `DATA_ERROR` — none of the fixes made in this
round are a general loosening, only a fix for the specific structural bug
(root cause 4) and a tightening (this one). None of the five match any
evidenced holiday either. They remain visible, unexplained, and worth a
human look — the audit output (see below) is built specifically to make
that possible without re-deriving anything.

### Forming candle: excluded, not fatal

The brief was explicit: exclude a forming candle from the validated view
rather than failing Gate 1 outright, and never modify the historical file
to satisfy validation. `check_completed()` in `validate_m30_candles.py` is
rewritten: it now trims trailing candles that either (a) have not closed
yet by real UTC "now", or (b) — per the file's own manifest — were still
forming *at the moment they were exported*, so their OHLC may be
provisional even though real time has since passed their close. Both cases
`WARN`, neither `FAIL`s Gate 1 by itself; the trimmed, usable view is what
field-usability and gap classification actually run against. Only if
*nothing* remains usable does this fail. The file on disk is never touched.
7 new tests in `tests/test_validate_m30_candles.py` (no MT5 needed) cover:
a fully closed final candle passing whole, a forming one being excluded (not
fatal), multiple trailing forming candles all being trimmed, an
all-forming file failing closed with nothing to validate, a
provisional-at-export-time candle being excluded even though it has since
closed, the input list never being mutated, and two runs against identical
input agreeing (no lookahead into anything not yet computed).

### Audit output (task 12)

Every `DATA_ERROR` gap printed by `validate_m30_candles.py` now shows, per
candidate rule, why it did not match: `weekly_close` (last-trade weekday/
hour and resume weekday), `known_holiday` (evidenced-holiday overlap or
not), and `broker_maintenance` (recognized hour or not; if recognized, the
measured typical size, MAD, tolerance band, and this gap's own size against
it). A `DATA_ERROR` verdict is now auditable line by line without
re-running anything.

### Verified on a reconstruction of this round's real numbers

Rebuilt the synthetic XAUUSD M30 series with the weekly close landing
exactly at midnight (root cause 4's exact shape), a DST-shifted daily pause,
evidenced holidays, all five of the report's flagged irregular examples
injected verbatim, and a genuinely forming final candle:

```
5. GAP CLASSIFICATION
  XAUUSD_M30: 829 total gaps ...
  recognized daily-close hour(s) UTC: ['22:00', '23:00']
    EXPECTED_MARKET_GAP  824
    SUSPICIOUS_GAP       0
    DATA_ERROR           5
  FAIL  XAUUSD_M30: 5 DATA_ERROR gaps — genuinely unexplained, full audit below...
    [all five of the injected irregular examples, each with its per-rule reasoning]

  WARN  XAUUSD_M30: excluded 1 trailing candle(s) still forming as of validation time
  GATE 1 FAILED — 1 blocking problem(s)
```

Exactly the intended outcome: the forming candle no longer blocks Gate 1 by
itself, the structural weekly-close bug is gone, and the five genuinely
unexplained gaps are neither hidden nor force-passed — Gate 1 still fails,
honestly, on what remains actually unexplained. Per the brief: "If genuine
DATA_ERROR gaps remain after correcting session classification, STOP and
report them instead of forcing a pass" — this synthetic run does exactly
that, and the real run is expected to behave the same way modulo whatever
the real five (or fewer, if some turn out to have a defensible explanation
once looked at directly) actually are.

### Files changed, round 3

* `analysis/features/timeframe_alignment.py` — `stop_weekday`/`stop_hour_utc`
  fields; `_daily_close_hours`, pass 1, and the broker-maintenance weekday
  filter now read `stop_weekday`; `_pause_size_stats` (median + MAD) replaces
  `_typical_missing_bars`; MAD-based tolerance band; `rule_checks` audit
  trail attached to every gap.
* `scripts/validate_m30_candles.py` — `check_completed` rewritten to trim
  and return the usable view instead of failing on a forming/provisional
  candle; `check_gaps`'s `DATA_ERROR` printout now includes the full
  per-rule audit trail.
* `tests/test_timeframe_alignment.py` — 3 new tests: the exact 49h and 48h
  midnight-boundary examples from the report, and the irregular-gap
  not-auto-whitelisted regression (69 total, up from 66).
* `tests/test_validate_m30_candles.py` (new) — 7 tests for the
  exclude-not-fail forming-candle behavior.
* This report.

### Test results, round 3

`pytest -q`: **955 passed**, 1 pre-existing unrelated failure (unchanged
from rounds 1-2).

### Still needed from the real file

Re-run on the real data after pulling this fix:

```bash
git pull
python scripts/validate_m30_candles.py --symbols XAUUSD
```

If the diagnosis is complete, `DATA_ERROR` should drop from 132 toward a
small number close to the five flagged examples (possibly fewer, if any of
those turn out to have a defensible explanation on closer look; possibly
the same five, confirming they are genuinely unexplained). The forming-
candle line should show as a `WARN` (excluded) rather than a `FAIL`. Any
remaining `DATA_ERROR` gaps print with their full per-rule audit trail —
report them back rather than force a pass, per the brief's own success
criteria.

Boundary unchanged: no model, training, label, threshold or trading-logic
code touched in round 3 either.

---

## Round 4 — forensic audit of the remaining 5 gaps: a tool, not a verdict

Round 3's fix brought the real result to `EXPECTED_MARKET_GAP: 798 /
SUSPICIOUS_GAP: 47 / DATA_ERROR: 5` — the same five gaps flagged as
"do not auto-classify" all along. This round is explicitly forensic: find
the *actual* root cause of each of the five, using real evidence, not
another calendar heuristic.

**This sandbox has no MT5 terminal and no broker connection.** Every
finding below that requires live MT5 data — Steps 2, 3, 4, 6, 7's "against
real data" — cannot be produced from here. What follows is (a) a code-level
audit of the fetch pipeline (Step 1, fully doable from source), and (b) a
diagnostic tool built to Step 5's exact spec for the user to run on the
Windows machine with MT5 running. **No calendar rule was changed in this
round.** Per the brief's own Step 8, that only happens after real evidence
exists — none does yet.

### Step 1 — pipeline audit: a real, code-level candidate cause

Traced the whole path: `fetch_training_candles.py::fetch()` → a single call
to `mt5.copy_rates_from_pos(symbol, timeframe, 0, count + 3)` requesting
**38,403 bars in one request** for M30 (3 years × 12,800/year) → `_closed_by`
filters forming bars → sort → truncate to `count` → write JSON. No other
module in this codebase (`data/market/mt5_client.py`,
`data/market/mt5_session.py`, `data/market/historical_fetcher.py`) is on
this path, and none of them implement chunking, pagination, or a
completeness check either — `historical_fetcher.py` is a pre-M30 module
capped at `limit=5000` and unrelated to this pipeline.

**The finding**: there is no pagination, no retry-with-backoff, and no
verification that the returned range is actually contiguous or complete —
`fetch()` trusts a single `copy_rates_from_pos` call for the entire 3-year,
38k-bar request and only checks `rates is None or len(rates) == 0`, never
"did I get everything I asked for." This is a well-documented MT5 behavior
class: the terminal can return a **partial, silently truncated** result for
a deep-history request when part of that history has not yet been
downloaded into its local cache from the broker server — with no error, no
exception, nothing but fewer bars than expected somewhere in the middle of
the range. Five *scattered, differently-sized, non-recurring* gaps (6, 5,
17, 3, 10 bars) at unrelated calendar dates is a plausible signature of
exactly this — an actual broker closure recurs on a predictable schedule;
a local-cache gap can land anywhere the terminal happened not to have
history for yet. This is a real, code-evidenced hypothesis
(`FETCH_PIPELINE_FAILURE`), not a guess, but it is still a hypothesis —
confirming it requires asking MT5 directly, which is exactly what the tool
below does.

### Step 5 — `scripts/audit_xauusd_m30_gaps.py` (new)

Read-only, MT5-only (loads without it, like every other fetch script here,
but every real check needs the Windows terminal). For each of the five
reported gaps, independently:

* queries MT5 `copy_rates_range` for **M30** over `[gap_start - 24h,
  gap_end + 24h]` and counts bars actually inside the gap window,
* the same for **M1** — M1 presence during an M30 hole would mean the
  market was NOT closed, which no calendar explanation could survive,
* queries `mt5.symbol_info_session_trade` for every day the gap touches, if
  the broker publishes one (correctly converts Python's Monday=0 weekday
  to MT5's Sunday=0 `day_of_week` — a silent day-off-by-one here would
  query the wrong day's session entirely),
* loads the exported historical file over the same window for comparison,

then prints the exact report format specified in the brief, and one of
five conclusions: `BROKER_SESSION_GAP`, `FETCH_PIPELINE_FAILURE`,
`HISTORICAL_FILE_CORRUPTION`, `CALENDAR_RULE_MISMATCH`, `UNKNOWN` — plus a
summary table across all five gaps at the end.

**Decision logic**, in the order checked:

1. MT5 has M30 bars in the gap window that the historical file is missing
   → `FETCH_PIPELINE_FAILURE` (unambiguous: if MT5 has the data, the market
   was not closed, and the exporter is what's responsible).
2. The historical file has bars in the gap window that MT5 no longer
   serves → `HISTORICAL_FILE_CORRUPTION`.
3. MT5 has zero M30 and zero M1 bars, and a published session says the
   market was closed → `BROKER_SESSION_GAP`.
4. MT5 has zero M30 and zero M1 bars, but no session data confirms a
   closure (either none published, or the broker doesn't expose the call)
   → `CALENDAR_RULE_MISMATCH` — genuinely absent, but not yet backed by any
   rule this codebase has.
5. Anything else (e.g. M30 empty but M1 has bars — a real contradiction,
   not an explanation) → `UNKNOWN`, printed with full evidence for a human
   to read rather than silently resolved either way.

### Step 7 — tests for the decision logic itself

`tests/test_audit_xauusd_m30_gaps.py` (new, 8 tests, no MT5 needed) — a
`FakeMT5` stand-in exercises all five conclusions directly: a pipeline
failure, a broker session gap, a calendar mismatch, the M30-empty-but-M1-
has-bars contradiction correctly landing as `UNKNOWN` rather than being
silently resolved, file-has-bars-MT5-doesn't as corruption, plus the
window-membership helper and the UTC-naive datetime conversion MT5's API
needs. This proves the tool's *logic* is sound; it says nothing yet about
which category the real five gaps actually fall into — only a real MT5 run
can answer that.

### What this round deliberately did NOT do

Per the brief's Step 8 ("only then modify the validator... do NOT introduce
a broad rule"): `classify_gaps()` was not touched, no calendar rule was
added or removed, `[00:00, 21:00, 23:00]` (or whatever the real
`daily_close_hours_utc` turns out to be) was not assumed correct or
incorrect, and the five `DATA_ERROR` gaps were not reclassified. Gate 1
remains failed on the real data, honestly, until real MT5 evidence exists.

### Files changed, round 4

* `scripts/audit_xauusd_m30_gaps.py` (new) — the forensic audit tool.
* `tests/test_audit_xauusd_m30_gaps.py` (new) — 8 tests for its decision logic.
* This report.

### Test results, round 4

`pytest -q`: **963 passed**, 1 pre-existing unrelated failure (unchanged).

### What is needed to actually close this out

Run on the Windows machine, with MT5 open and logged in:

```bash
git pull
python scripts/audit_xauusd_m30_gaps.py --symbol XAUUSD
```

This produces, per gap, the exact evidence Steps 2/3/4/6 asked for
(MT5 M30/M1 bar counts, published session data, historical-file comparison)
and one of the five conclusions above, plus a summary table. Share that
output back — it is what turns "5 unexplained gaps" into either a narrow,
documented, evidence-backed calendar rule (Step 8, only for whichever
gaps prove to be `BROKER_SESSION_GAP` or a defensible
`CALENDAR_RULE_MISMATCH`), a fetch-pipeline fix and re-fetch (Step 9, for
whichever prove `FETCH_PIPELINE_FAILURE`), or a confirmed genuine data
problem that stays `DATA_ERROR` and keeps Gate 1 failed, per the brief's
own explicit success criteria: "Every missing interval has a defensible,
evidence-based explanation," not "Gate 1 passes at any cost."

Boundary unchanged: no model, training, label, threshold or trading-logic
code touched in round 4. Nothing was re-fetched (no MT5 access here to do
it with), and no historical file was modified.

---

## Round 5 — the audit tool was actually run, and it had a real bug: fixed

The user ran round 4's `scripts/audit_xauusd_m30_gaps.py` against the real
MetaTrader5 Python package (**5.0.5735**, terminal build **6116**). Two
things came back:

1. `mt5.symbol_info_session_trade` — **does not exist in this package
   version**. The round-4 script called it, caught the resulting
   `AttributeError`, and (a real bug) treated that catch as if it were
   informative, weakly reporting `CALENDAR_RULE_MISMATCH` for all five gaps
   on that basis alone. That conclusion was never actually earned by
   evidence.
2. With that call removed, the already-established facts stand:
   `M30 bars in gap = 0` and `M1 bars = 0` for all five gaps, and the
   historical file also has zero bars in each — so there is still no
   evidence of a fetch-pipeline failure, but there was also no legitimate
   basis yet for calling these closures either.

### The fix — remove the invented API, add real evidence sources instead

`scripts/audit_xauusd_m30_gaps.py` was rewritten (this is now the only file
changed, per this round's explicit instruction — `validate_m30_candles.py`,
`classify_gaps()`, the historical file, and training code are all
untouched):

* **No hard-coded session API.** `detect_session_api(mt5)` inspects
  `dir(mt5)` at runtime for anything with "session" in its name — on
  5.0.5735 that is an empty list, reported honestly as
  `SESSION_METADATA_UNAVAILABLE`, never silently promoted to "market was
  closed." If a future package version exposes something under a different
  name, this picks it up automatically and calls it defensively, but does
  **not** auto-interpret its return value as proof of anything — no named,
  reviewed interpreter exists for an API this script has never seen
  documented behavior for, so `confirms_closed` stays `None` even when a
  session-named attribute is found and called successfully. A regression
  test (`test_broker_session_gap_is_unreachable_without_an_interpreted_confirmation`)
  proves `BROKER_SESSION_GAP` cannot be manufactured from an unknown API's
  raw output.
* **Explicit, self-contained MT5 connection.** `connect_mt5()` calls
  `mt5.initialize()` → `terminal_info()` → `symbol_info()` (with
  `symbol_select()` fallback) directly — matching exactly how the user's
  own successful manual diagnostic connected (bare `initialize()`, no
  `login()`, attaching to the already-running, already-authenticated
  terminal) rather than going through `mt5_session.ensure_session()`'s
  heavier `.env`-credentialed login path. `mt5.shutdown()` runs in a
  `finally`, so it happens even when a gap audit raises.
* **Ticks, not just M30/M1.** `copy_ticks_range` is queried for every gap
  (`hasattr` — checked, not assumed; a failed or unsupported query reports
  `ERROR`/`UNSUPPORTED` explicitly and is never read as "no ticks exist").
  Tick activity immediately before AND after a gap, combined with zero
  ticks/M1/M30 *inside* it, is now the actual corroborating evidence for
  `CALENDAR_RULE_MISMATCH` — proof that MT5's history coverage isn't
  broken in that region generally, which a bare M30/M1 zero count alone
  never established.
* **Exact timestamp SETS, not counts.** `mt5_only` / `historical_only` are
  now computed by diffing the actual timestamp sets MT5 and the historical
  file return over both the full 48h window and the gap itself — the
  brief's explicit requirement ("do not compare only counts... do not
  classify based only on counts").
* **Conservative classification**, in order: (1) MT5 has M30 timestamps
  inside the gap the file lacks → `FETCH_PIPELINE_FAILURE`, HIGH; (2) the
  file has timestamps MT5 no longer serves → `HISTORICAL_FILE_CORRUPTION`,
  MEDIUM (explicitly caveated: MT5's local cache can itself be limited);
  (3) both empty, and an interpreted session source confirms closure →
  `BROKER_SESSION_GAP`, HIGH (currently unreachable on this package, by
  design); (4) both empty, no session confirmation, but ticks corroborate
  genuine coverage around the gap → `CALENDAR_RULE_MISMATCH`, MEDIUM; (5)
  anything else, including a tick-query failure or missing before/after
  tick corroboration → `UNKNOWN`, LOW. Every path is evidence-gated; none
  of them can be reached by absence of evidence alone.
* **Full per-gap report** in the exact format requested (MT5 connection,
  M30, M1, TICKS, HISTORICAL FILE, SESSION EVIDENCE, EVIDENCE bullets,
  CONCLUSION, CONFIDENCE) plus a summary table and conclusion counts, and
  the required `READ-ONLY AUDIT` banner up front.

### Tests

`tests/test_audit_xauusd_m30_gaps.py` rewritten, 22 tests (up from 8), all
mocked — no live MT5 needed, none fabricate a live conclusion. Covers
every case the brief listed: initialization failure, symbol missing (both
recoverable via `symbol_select` and not), the old hard-coded session call
being structurally absent from the source (`mt5.symbol_info_session_trade`
never appears as an attribute access), a detected-but-uninterpreted session
API, MT5-has-bars-file-doesn't, file-has-bars-MT5-doesn't, MT5-and-
historical-both-empty-with-no-tick-corroboration landing as `UNKNOWN` (the
literal real-world case this round investigates), tick-query failure and
tick-query exception both reported as `ERROR` rather than "closed", the
tick-corroborated `CALENDAR_RULE_MISMATCH` path, the M30/M1-empty-but-
ticks-present contradiction staying `UNKNOWN`, exact timestamp-set
matching, and that neither `audit_one_gap` nor `connect_mt5` contain a
file-write call anywhere in their source.

### Test results, round 5

`pytest -q`: **977 passed**, 1 pre-existing unrelated failure (unchanged).

### What this round found — and did not find

With the invented API removed, the honest state of the evidence for all
five gaps, from the code alone, is: **insufficient to classify** — no
tick-query result, no session data, and no timestamp-set comparison have
been produced by an actual live run of the corrected script yet, since
this sandbox still has no MT5 access. The round-4 `CALENDAR_RULE_MISMATCH`
verdict for all five gaps is **retracted** — it was never properly earned.

### Files changed, round 5

* `scripts/audit_xauusd_m30_gaps.py` — rewritten per above. The only
  production file touched, as instructed.
* `tests/test_audit_xauusd_m30_gaps.py` — rewritten, 22 tests.
* This report.

Not touched: `scripts/validate_m30_candles.py`, `classify_gaps()` or any
other part of `analysis/features/timeframe_alignment.py`,
`data/historical/XAUUSD_M30.json`, `fetch_training_candles.py`, any model,
training, or feature-spec code.

### Next step

Run the corrected script on the Windows machine, MT5 open and logged in:

```bash
git pull
python scripts/audit_xauusd_m30_gaps.py --symbol XAUUSD
```

Then — **only** after reading that output — decide whether
`scripts/validate_m30_candles.py` needs a narrow, evidence-backed rule.
Per the brief: do not change the validator or calendar rules based on
assumptions. Share the output back either way, including if every gap
still lands as `UNKNOWN` — that is a valid, honest result, not a failure
of the tool.
