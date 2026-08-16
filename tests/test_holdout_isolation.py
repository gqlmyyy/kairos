"""The holdout region must be provably unreachable, not just sliced away.

Two real bugs were found and fixed while building this check, both worth
pinning permanently:

1. The first corruption boundary used a research row's OWN decision time,
   ignoring that its barrier LABEL legitimately reads forward up to `horizon`
   bars past it — that is normal labelling, not leakage, and the check must
   not corrupt those bars.
2. Deriving the boundary from which decisions happened to RESOLVE into a row
   (rather than from every decision the cutoff admits) missed unresolved
   near-boundary decisions: corrupting their forward window can flip them
   from unresolved to resolved, manufacturing a new row inside the research
   region that a naive per-resolved-row boundary never accounts for.
3. A subtler structural issue: splitting research/holdout by a COUNT of
   resolved rows (`int(len(y_all) * frac)`) makes the split point itself a
   function of what happens to be labellable in the holdout region — corrupt
   holdout candles, change how many resolve, shift the total count, and a
   few borderline rows change sides. Fixed by splitting on a TIMESTAMP fixed
   from raw candle span before any labelling happens, which is invariant to
   anything that happens to holdout candle values.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from analysis.features import timeframe_alignment as ta  # noqa: E402

trainer = pytest.importorskip("train_entry_model")
idisc = pytest.importorskip("information_discovery")


def candles(tf, n, seed, price, vol, spread_base):
    span = ta.duration(tf)
    rng = random.Random(seed)
    out, c = [], price
    for i in range(n):
        o = c * (1 + rng.gauss(0, vol * 0.3))
        c = o * (1 + rng.gauss(0, vol))
        h = max(o, c) + abs(rng.gauss(0, vol * c))
        l = min(o, c) - abs(rng.gauss(0, vol * c))
        out.append({"t": float(i * span), "open": o, "high": h, "low": l, "close": c,
                    "volume": 100.0 + rng.random() * 50,
                    "spread": spread_base + rng.random() * spread_base * 0.5,
                    "real_volume": 0.0})
    return out


@pytest.fixture(scope="module")
def dataset():
    return {
        "EURUSD": {"H4": candles("H4", 700, 1, 1.10, 0.0012, 1.2),
                  "H1": candles("H1", 2800, 2, 1.10, 0.0006, 1.2)},
        "GBPUSD": {"H4": candles("H4", 700, 3, 1.27, 0.0013, 1.5),
                  "H1": candles("H1", 2800, 4, 1.27, 0.0007, 1.5)},
    }


class TestResearchCutoffIsTimestampBased:
    def test_cutoff_is_computed_from_raw_candle_span(self, dataset):
        cutoff = idisc.research_cutoff_time(dataset, 0.7)
        all_t = [c["t"] for tf in dataset.values() for c in tf["H4"]]
        assert min(all_t) <= cutoff <= max(all_t)

    def test_cutoff_does_not_depend_on_labelling_at_all(self, dataset):
        """The cutoff must be computable before build_dataset ever runs."""
        cutoff_before = idisc.research_cutoff_time(dataset, 0.7)
        trainer.build_dataset(dataset, horizon=24)  # labelling happens here
        cutoff_after = idisc.research_cutoff_time(dataset, 0.7)
        assert cutoff_before == cutoff_after


class TestHoldoutCorruptionCannotChangeResearchOutput:
    def test_verify_holdout_isolation_passes_on_clean_data(self, dataset):
        assert idisc.verify_holdout_isolation(dataset, horizon=24, research_frac=0.7) is True

    def test_the_check_can_actually_fire(self, dataset):
        """Non-vacuity: corrupting a RESEARCH-region candle must be caught."""
        X, y, meta, holdout_n = idisc.build_research_rows(dataset, 24, 0.7)
        cutoff = idisc.research_cutoff_time(dataset, 0.7)

        damaged = {s: {tf: [dict(c) for c in series] for tf, series in tfs.items()}
                  for s, tfs in dataset.items()}
        # Corrupt a candle well inside the research region (past WARMUP_BARS=100
        # so it actually feeds a decision, and well before the ~70% cutoff).
        damaged["EURUSD"]["H4"][300]["close"] *= 50.0
        damaged["EURUSD"]["H4"][300]["high"] *= 50.0

        X2, y2, meta2, _ = idisc.build_research_rows(damaged, 24, 0.7)
        assert X != X2 or meta != meta2, (
            "corrupting a research-region candle had no effect — the "
            "isolation check would not be able to catch a real leak either")

    def test_holdout_row_count_is_reported_and_nonzero(self, dataset):
        X, y, meta, holdout_n = idisc.build_research_rows(dataset, 24, 0.7)
        assert holdout_n > 0
        assert len(X) > 0

    def test_every_research_row_precedes_the_cutoff(self, dataset):
        _, _, meta, _ = idisc.build_research_rows(dataset, 24, 0.7)
        cutoff = idisc.research_cutoff_time(dataset, 0.7)
        assert all(m["t"] <= cutoff for m in meta)
