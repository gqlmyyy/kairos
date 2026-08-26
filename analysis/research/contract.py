"""The canonical feature contract for research entry models.

One contract, four consumers
----------------------------
Training (in the xgbooost research repo), offline inference, replay and any
future live path all read THIS module. There is no second contract, no
"live variant", no implicit fallback set. That plurality is what produced the
65-vs-10 mismatch documented in ``analysis/models/entry_feature_contract.py``,
and it is the specific failure this module exists to make impossible.

What a feature declares
-----------------------
Every feature carries the full contract, not just a name:

=================  ==========================================================
``name``           unique column name; order is fixed by the model manifest
``source``         ohlc | timestamp | spread | cross_timeframe | meta
``formula``        the exact arithmetic, as a string
``timeframe``      the timeframe whose candles produce the value
``lookback``       trailing bars the formula reads
``minimum_history``bars needed before a defined value exists (>= lookback)
``dtype``          declared dtype, asserted against the produced column
``unit``           dimensionless | price | points | hours | days | minutes | bars
``normalization``  what the raw quantity was divided by, or "none"
``availability``   the instant the value is knowable: always ``close_time``
``missing_policy`` what a not-yet-defined value is, and what it is NOT
``stationarity``   SCALE_FREE | LEVEL | PRICE_UNIT | BROKER_UNIT
``requires``       the raw candle columns the formula reads
=================  ==========================================================

Why ``stationarity`` is part of the contract
--------------------------------------------
The research repo measured the thing KAIROS's legacy contract only suspected:
a raw price-unit column has no shared meaning across instruments or across
time. XAUUSD traded 1619-2789 in the research TRAIN window and 2845-3784 in
VALIDATION — zero overlap — so every tree split on ``atr``, ``macd_line`` or
``ema_20`` is a constant in validation. That is a domain shift, not
overfitting, and no amount of retraining fixes it.

The research models therefore select from the SCALE_FREE subset only
(``feature_set`` = ``scale_free_dedup`` / ``scale_free_dedup_nodir``). This
module classifies every feature the same rule-based way, so a LEVEL or
PRICE_UNIT column can never reach a research model by accident:
:func:`assert_scale_free` is called by the loader on every model's feature
list.

Legacy features (``atr``, ``macd``, ``session``, the bucketed
``trend_score``/``momentum_score``/``volatility_score``) are NOT in this
contract. They are a different, incompatible vocabulary, and they stay in
``analysis/models/entry_feature_spec.py`` where the legacy model still uses
them. Same word, different arithmetic, is the drift that is hardest to see:
this contract's ``trend_score`` is a normalised regression slope in [-inf,
inf], the legacy one is a bucket from {40, 65, 70, 75, 85}. They are not
compatible and must never be mapped onto one another.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

# --- contract vocabulary ----------------------------------------------------

FEATURE_SCHEMA_VERSION = "research-1.2.0"

SOURCE_OHLC = "ohlc"
SOURCE_TIMESTAMP = "timestamp"
SOURCE_SPREAD = "spread"
SOURCE_CROSS_TIMEFRAME = "cross_timeframe"
SOURCE_META = "meta"

# Availability is uniform and non-negotiable: a value is knowable only once
# the candle that produced it has CLOSED. The MTF aligner enforces it for
# context timeframes; for the entry timeframe it is true by construction.
AVAILABILITY_CLOSE_TIME = "close_time"

NULL_NAN_UNTIL_WARMUP = "nan_until_warmup"
NULL_DEFINED_FROM_FIRST_BAR = "defined_from_first_bar"
NULL_MIDPOINT_ON_DEGENERATE_DAY = "midpoint_0.5_when_day_range_is_zero"
NULL_UNIT_WHEN_ALSO_ZERO = "unit_when_spread_and_median_both_zero"

UNIT_DIMENSIONLESS = "dimensionless"
UNIT_PRICE = "price"
UNIT_POINTS = "points"
UNIT_HOURS = "hours"
UNIT_DAYS = "days"
UNIT_MINUTES = "minutes"

# --- stationarity classes ---------------------------------------------------
SCALE_FREE = "SCALE_FREE"
LEVEL = "LEVEL"
PRICE_UNIT = "PRICE_UNIT"
BROKER_UNIT = "BROKER_UNIT"
NON_STATIONARY = (LEVEL, PRICE_UNIT, BROKER_UNIT)

CONTEXT_PREFIXES = ("M15_", "M30_", "H1_", "H4_")


class ContractError(Exception):
    """A feature reached the contract that no rule covers, or violates it."""


@dataclass(frozen=True)
class FeatureSpec:
    """The full declared contract of one feature column."""

    name: str
    source: str
    formula: str
    timeframe: str
    lookback: int
    minimum_history: int
    dtype: str
    unit: str
    normalization: str
    availability: str
    missing_policy: str
    stationarity: str
    requires: Tuple[str, ...]

    def renamed(self, new_name: str, timeframe: str) -> "FeatureSpec":
        """Same contract, carried onto a context-prefixed column.

        The timeframe is re-pointed to the timeframe whose candles actually
        produced the value — an ``H4_rsi`` column on an H1 row is still an H4
        measurement, and reporting it as H1 would make the causality claim
        unverifiable.
        """
        return FeatureSpec(
            name=new_name, source=self.source, formula=self.formula,
            timeframe=timeframe, lookback=self.lookback,
            minimum_history=self.minimum_history, dtype=self.dtype,
            unit=self.unit, normalization=self.normalization,
            availability=self.availability, missing_policy=self.missing_policy,
            stationarity=self.stationarity, requires=self.requires,
        )

    def as_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# The feature library.
#
# Keyed by BASE name (the name before any context prefix). Parameters are the
# research defaults from the frozen research config; they are written into the
# formula strings rather than left implicit, so a contract dump is readable
# without holding the config open next to it.
# ---------------------------------------------------------------------------

RSI_PERIOD = 14
ATR_PERIOD = 14
EMA_FAST = 20
EMA_SLOW = 50
SMA_PERIOD = 20
TREND_SCORE_LOOKBACK = 20
MOMENTUM_LOOKBACK = 10
VOLATILITY_LOOKBACK = 50
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
DIRECTION_EMA_PERIOD = 20
RECENT_LOOKBACK = 20
ROLLING_VOL_LOOKBACK = 20
BB_PERIOD = 20
BB_STD = 2.0
STOCH_K = 14
STOCH_D = 3
STOCH_SMOOTH = 3
CCI_PERIOD = 20
RETURN_PERIODS = (1, 3, 5)
SPREAD_RELATIVE_LOOKBACK = 200

# Sessions, in session-timezone hours. Windows are half-open [start, end).
SESSION_WINDOWS = {
    "asian": ("00:00", "08:00"),
    "london": ("08:00", "16:00"),
    "newyork": ("13:00", "21:00"),
}

# Per-symbol trading-day open, used only by `minutes_from_session_open`. These
# are the dominant empirical opens measured by the research repo, and they are
# a fixed-UTC approximation: on DST-split days the true open shifts by an hour.
# That limitation is inherited deliberately and stated rather than hidden — a
# second DST engine for one feature would be a new source of divergence.
TRADING_DAY_OPEN_UTC = {"EURUSD": "00:00", "GBPUSD": "00:00", "XAUUSD": "01:00"}
DEFAULT_TRADING_DAY_OPEN_UTC = "00:00"

# The timezone day boundaries and session windows are evaluated in. The
# research datasets were built with session_timezone = UTC.
SESSION_TIMEZONE = "UTC"


def _spec(name, source, formula, lookback, dtype, unit, normalization,
          missing_policy, stationarity, requires, minimum_history=None):
    return FeatureSpec(
        name=name, source=source, formula=formula, timeframe="",
        lookback=int(lookback),
        minimum_history=int(lookback if minimum_history is None else minimum_history),
        dtype=dtype, unit=unit, normalization=normalization,
        availability=AVAILABILITY_CLOSE_TIME, missing_policy=missing_policy,
        stationarity=stationarity, requires=tuple(requires),
    )


_D = UNIT_DIMENSIONLESS
_OH = SOURCE_OHLC
_F = "float64"
_W = NULL_NAN_UNTIL_WARMUP
_FB = NULL_DEFINED_FROM_FIRST_BAR

#: Every per-timeframe feature this contract knows how to produce.
FEATURE_LIBRARY: Dict[str, FeatureSpec] = {f.name: f for f in [
    # ---- momentum / oscillators -------------------------------------------
    _spec("rsi", _OH, f"Wilder RSI({RSI_PERIOD}): 100 - 100/(1+RS), "
          f"RS = EWM(gain, alpha=1/{RSI_PERIOD}) / EWM(loss, alpha=1/{RSI_PERIOD})",
          RSI_PERIOD, _F, _D, "bounded 0..100 by construction", _W, SCALE_FREE, ["close"]),
    _spec("momentum_score", _OH, f"close.pct_change({MOMENTUM_LOOKBACK})",
          MOMENTUM_LOOKBACK, _F, _D, "fractional change of close", _W, SCALE_FREE, ["close"]),
    _spec("stochastic_k", _OH,
          f"SMA(100*(close-LL({STOCH_K}))/(HH({STOCH_K})-LL({STOCH_K})), {STOCH_SMOOTH})",
          STOCH_K, _F, _D, "bounded 0..100 by construction", _W, SCALE_FREE,
          ["high", "low", "close"], minimum_history=STOCH_K + STOCH_SMOOTH),
    _spec("stochastic_d", _OH, f"SMA(stochastic_k, {STOCH_D})",
          STOCH_K, _F, _D, "bounded 0..100 by construction", _W, SCALE_FREE,
          ["high", "low", "close"], minimum_history=STOCH_K + STOCH_SMOOTH + STOCH_D),
    _spec(f"cci_{CCI_PERIOD}", _OH,
          f"(TP - SMA(TP,{CCI_PERIOD})) / (0.015 * MAD(TP,{CCI_PERIOD})), TP=(h+l+c)/3",
          CCI_PERIOD, _F, _D, "Lambert 0.015 * mean absolute deviation", _W, SCALE_FREE,
          ["high", "low", "close"]),
    # ---- returns -----------------------------------------------------------
    *[_spec(f"return_{p}", _OH, f"close.pct_change({p})", p, _F, _D,
            "fractional change of close", _W, SCALE_FREE, ["close"])
      for p in RETURN_PERIODS],
    *[_spec(f"return_acceleration_{a}_{b}", _OH, f"return_{a} - return_{b}", b, _F, _D,
            "difference of two fractional returns", _W, SCALE_FREE, ["close"])
      for a, b in zip(RETURN_PERIODS, RETURN_PERIODS[1:])],
    # ---- trend -------------------------------------------------------------
    _spec("trend_strength", _OH, f"(EMA({EMA_FAST}) - EMA({EMA_SLOW})) / close",
          EMA_SLOW, _F, _D, "divided by close", _W, SCALE_FREE, ["close"]),
    _spec("trend_score", _OH,
          f"OLS slope of close over the trailing {TREND_SCORE_LOOKBACK} bars, divided by close",
          TREND_SCORE_LOOKBACK, _F, _D, "divided by close", _W, SCALE_FREE, ["close"]),
    _spec("direction", _OH, f"sign(close - EMA({DIRECTION_EMA_PERIOD})) -> -1 | 0 | +1",
          DIRECTION_EMA_PERIOD, _F, _D, "sign only", _W, SCALE_FREE, ["close"]),
    _spec("adx", _OH, f"Wilder ADX({ADX_PERIOD})", ADX_PERIOD, _F, _D,
          "bounded 0..100 by construction", _W, SCALE_FREE, ["high", "low", "close"]),
    _spec("plus_di", _OH, f"Wilder +DI({ADX_PERIOD})", ADX_PERIOD, _F, _D,
          "100 * EWM(+DM) / ATR", _W, SCALE_FREE, ["high", "low", "close"]),
    _spec("minus_di", _OH, f"Wilder -DI({ADX_PERIOD})", ADX_PERIOD, _F, _D,
          "100 * EWM(-DM) / ATR", _W, SCALE_FREE, ["high", "low", "close"]),
    _spec("market_regime", _OH,
          f"1.0 if ADX({ADX_PERIOD}) >= {ADX_TREND_THRESHOLD} (TRENDING) else 0.0 (RANGING)",
          ADX_PERIOD, _F, _D, "binary indicator", _W, SCALE_FREE, ["high", "low", "close"]),
    _spec("distance_close_ema20_atr", _OH,
          f"(close - EMA({EMA_FAST})) / ATR({ATR_PERIOD})", max(EMA_FAST, ATR_PERIOD), _F, _D,
          f"divided by ATR({ATR_PERIOD})", _W, SCALE_FREE, ["high", "low", "close"]),
    _spec("distance_close_ema50_atr", _OH,
          f"(close - EMA({EMA_SLOW})) / ATR({ATR_PERIOD})", max(EMA_SLOW, ATR_PERIOD), _F, _D,
          f"divided by ATR({ATR_PERIOD})", _W, SCALE_FREE, ["high", "low", "close"]),
    _spec("ema20_ema50_distance_atr", _OH,
          f"(EMA({EMA_FAST}) - EMA({EMA_SLOW})) / ATR({ATR_PERIOD})",
          max(EMA_SLOW, ATR_PERIOD), _F, _D, f"divided by ATR({ATR_PERIOD})", _W, SCALE_FREE,
          ["high", "low", "close"]),
    _spec("close_to_sma20_atr", _OH,
          f"(close - SMA({SMA_PERIOD})) / ATR({ATR_PERIOD})", SMA_PERIOD, _F, _D,
          f"divided by ATR({ATR_PERIOD})", _W, SCALE_FREE, ["high", "low", "close"]),
    # ---- volatility --------------------------------------------------------
    _spec("volatility_score", _OH,
          f"ATR({ATR_PERIOD}) / rolling_mean(ATR, {VOLATILITY_LOOKBACK})",
          VOLATILITY_LOOKBACK, _F, _D, "divided by its own trailing mean", _W, SCALE_FREE,
          ["high", "low", "close"]),
    _spec("atr_pct", _OH, f"ATR({ATR_PERIOD}) / close", ATR_PERIOD, _F, _D,
          "divided by close", _W, SCALE_FREE, ["high", "low", "close"]),
    _spec("rolling_volatility", _OH,
          f"rolling_std(close.pct_change(), {ROLLING_VOL_LOOKBACK})",
          ROLLING_VOL_LOOKBACK, _F, _D, "std of fractional returns", _W, SCALE_FREE, ["close"]),
    _spec("bb_width", _OH,
          f"(BB_upper - BB_lower) / SMA({BB_PERIOD}), BB = SMA +/- {BB_STD}*std(ddof=0)",
          BB_PERIOD, _F, _D, "divided by the middle band", _W, SCALE_FREE, ["close"]),
    # ---- structure ---------------------------------------------------------
    _spec("bb_percent_b", _OH, "(close - BB_lower) / (BB_upper - BB_lower); NOT clipped",
          BB_PERIOD, _F, _D, "divided by the band span", _W, SCALE_FREE, ["close"]),
    _spec("distance_from_recent_high", _OH,
          f"(close - rolling_max(high, {RECENT_LOOKBACK})) / close", RECENT_LOOKBACK, _F, _D,
          "divided by close", _W, SCALE_FREE, ["close", "high"]),
    _spec("distance_from_recent_low", _OH,
          f"(close - rolling_min(low, {RECENT_LOOKBACK})) / close", RECENT_LOOKBACK, _F, _D,
          "divided by close", _W, SCALE_FREE, ["close", "low"]),
    _spec("distance_from_day_high", _OH,
          "(close - running day high) / close; day boundary in session timezone",
          0, _F, _D, "divided by close", _FB, SCALE_FREE, ["high", "close", "timestamp"]),
    _spec("distance_from_day_low", _OH,
          "(close - running day low) / close; day boundary in session timezone",
          0, _F, _D, "divided by close", _FB, SCALE_FREE, ["low", "close", "timestamp"]),
    _spec("day_range_atr", _OH,
          f"(running day high - running day low) / ATR({ATR_PERIOD})", ATR_PERIOD, _F, _D,
          f"divided by ATR({ATR_PERIOD})", _W, SCALE_FREE, ["high", "low", "timestamp"]),
    _spec("position_in_day_range", _OH,
          "(close - running day low) / (running day high - running day low)",
          0, _F, _D, "divided by the running day range", NULL_MIDPOINT_ON_DEGENERATE_DAY,
          SCALE_FREE, ["high", "low", "close", "timestamp"]),
    # ---- ATR-normalised candle geometry ------------------------------------
    _spec("range_atr", _OH, f"(high - low) / ATR({ATR_PERIOD})", ATR_PERIOD, _F, _D,
          f"divided by ATR({ATR_PERIOD})", _W, SCALE_FREE, ["high", "low", "close"]),
    _spec("body_atr", _OH, f"abs(close - open) / ATR({ATR_PERIOD})", ATR_PERIOD, _F, _D,
          f"divided by ATR({ATR_PERIOD})", _W, SCALE_FREE, ["open", "high", "low", "close"]),
    _spec("upper_wick_atr", _OH,
          f"(high - max(open, close)) / ATR({ATR_PERIOD})", ATR_PERIOD, _F, _D,
          f"divided by ATR({ATR_PERIOD})", _W, SCALE_FREE, ["open", "high", "low", "close"]),
    _spec("lower_wick_atr", _OH,
          f"(min(open, close) - low) / ATR({ATR_PERIOD})", ATR_PERIOD, _F, _D,
          f"divided by ATR({ATR_PERIOD})", _W, SCALE_FREE, ["open", "high", "low", "close"]),
    # ---- session / clock ---------------------------------------------------
    *[_spec(f"session_{n}", SOURCE_TIMESTAMP,
            f"1.0 if the row's own timestamp (session tz) is in [{w[0]}, {w[1]}) else 0.0",
            0, _F, _D, "binary indicator", _FB, SCALE_FREE, ["timestamp"])
      for n, w in SESSION_WINDOWS.items()],
    _spec("hour_of_day", SOURCE_TIMESTAMP, "hour of the row's own timestamp in session tz",
          0, _F, UNIT_HOURS, "none", _FB, SCALE_FREE, ["timestamp"]),
    _spec("day_of_week", SOURCE_TIMESTAMP, "Mon=0 .. Sun=6 in session tz",
          0, _F, UNIT_DAYS, "none", _FB, SCALE_FREE, ["timestamp"]),
    _spec("minute_of_day", SOURCE_TIMESTAMP, "minutes since local midnight in session tz",
          0, _F, UNIT_MINUTES, "none", _FB, SCALE_FREE, ["timestamp"]),
    _spec("minutes_from_session_open", SOURCE_TIMESTAMP,
          "(minute_of_day - symbol trading-day open) mod 1440; fixed-UTC approximation "
          "across DST-split days",
          0, _F, UNIT_MINUTES, "modulo 1440", _FB, SCALE_FREE, ["timestamp"]),
    # ---- spread ------------------------------------------------------------
    _spec("spread_relative", SOURCE_SPREAD,
          f"spread / rolling_median(spread, {SPREAD_RELATIVE_LOOKBACK})",
          SPREAD_RELATIVE_LOOKBACK, _F, _D, "divided by its own trailing median",
          NULL_UNIT_WHEN_ALSO_ZERO, SCALE_FREE, ["spread"]),
]}

#: Names the contract knows but which are NOT scale-free. Listed explicitly so
#: that a request for one fails with "excluded by evidence", not "unknown".
EXCLUDED_NON_STATIONARY: Dict[str, str] = {
    "atr": PRICE_UNIT, "macd_line": PRICE_UNIT, "macd_signal": PRICE_UNIT,
    "macd_histogram": PRICE_UNIT, "range": PRICE_UNIT, "body_size": PRICE_UNIT,
    "upper_wick": PRICE_UNIT, "lower_wick": PRICE_UNIT, "day_range": PRICE_UNIT,
    "normalized_returns": SCALE_FREE,  # scale-free but deduped away as a duplicate
    "ema_20": LEVEL, "ema_50": LEVEL, "sma_20": LEVEL,
    "bb_upper": LEVEL, "bb_lower": LEVEL,
    "resistance_level": LEVEL, "support_level": LEVEL,
    "spread_points": BROKER_UNIT, "spread_ma": BROKER_UNIT,
}

# --- meta and cross-timeframe features --------------------------------------

#: The one meta column that is also a legitimate model input: it tells the
#: model which side of price it is scoring, which changes what target=1 means.
ENTRY_DIRECTION = FeatureSpec(
    name="entry_direction", source=SOURCE_META,
    formula="+1.0 for a long candidate, -1.0 for a short candidate",
    timeframe="n/a", lookback=0, minimum_history=0, dtype="float64",
    unit=UNIT_DIMENSIONLESS, normalization="sign only",
    availability=AVAILABILITY_CLOSE_TIME,
    missing_policy="required — a candidate with no side cannot be scored",
    stationarity=SCALE_FREE, requires=(),
)

MTF_TREND_SOURCE = "direction"

_MTF_SUFFIXES = ("_trend_state", "_trend_agreement", "_full_alignment")


def _mtf_spec(name: str, entry_timeframe: str) -> FeatureSpec:
    if name.endswith("_trend_state"):
        tf = name[: -len("_trend_state")]
        formula = (f"the {tf} candle's own `direction`; context timeframes arrive "
                   f"through merge_asof on close_time, so only a CLOSED candle is used")
        timeframe = tf
    elif name == "trend_alignment_score":
        formula = "mean of every available <tf>_trend_state, in [-1, +1]"
        timeframe = entry_timeframe
    elif name.endswith("_trend_agreement"):
        a, b = name[: -len("_trend_agreement")].split("_")[0:2]
        formula = (f"1.0 iff {a}_trend_state and {b}_trend_state share a NON-ZERO sign "
                   f"(a flat timeframe is not agreement), else 0.0")
        timeframe = entry_timeframe
    else:  # _full_alignment
        formula = "1.0 only when every available <tf>_trend_state shares one non-zero sign"
        timeframe = entry_timeframe
    return FeatureSpec(
        name=name, source=SOURCE_CROSS_TIMEFRAME, formula=formula,
        timeframe=timeframe, lookback=DIRECTION_EMA_PERIOD,
        minimum_history=DIRECTION_EMA_PERIOD, dtype="float64",
        unit=UNIT_DIMENSIONLESS, normalization="sign / mean of signs",
        availability=AVAILABILITY_CLOSE_TIME, missing_policy=NULL_NAN_UNTIL_WARMUP,
        stationarity=SCALE_FREE, requires=("close",),
    )


def alignment_feature_names(entry_timeframe: str,
                            context_timeframes: Sequence[str]) -> List[str]:
    """The cross-timeframe names, in the order the engine appends them."""
    tfs = [entry_timeframe] + list(context_timeframes)
    names = [f"{tf}_trend_state" for tf in tfs]
    names.append("trend_alignment_score")
    for a, b in zip(tfs, tfs[1:]):
        names.append(f"{a}_{b}_trend_agreement")
    if len(tfs) >= 2:
        names.append("_".join(tfs) + "_full_alignment")
    return names


# --- resolution -------------------------------------------------------------

def strip_context(name: str) -> Tuple[str, Optional[str]]:
    """``H4_rsi`` -> ``('rsi', 'H4')``; ``rsi`` -> ``('rsi', None)``.

    Only strips when the remainder is a real base name, so a genuinely
    prefix-named column such as ``M15_trend_state`` is left intact.
    """
    for p in CONTEXT_PREFIXES:
        if name.startswith(p) and name[len(p):] in FEATURE_LIBRARY:
            return name[len(p):], p[:-1]
    return name, None


def resolve(name: str, entry_timeframe: str) -> FeatureSpec:
    """The full contract of one column of a model's feature list.

    Raises ContractError for anything the contract does not cover. An unknown
    column is never given a default spec: guessing a feature's timeframe or
    lookback is how a look-ahead claim becomes unverifiable.
    """
    if name == ENTRY_DIRECTION.name:
        return ENTRY_DIRECTION
    if any(name.endswith(s) for s in _MTF_SUFFIXES) or name == "trend_alignment_score":
        return _mtf_spec(name, entry_timeframe)
    base, ctx = strip_context(name)
    if base in FEATURE_LIBRARY:
        spec = FEATURE_LIBRARY[base]
        tf = ctx or entry_timeframe
        return spec.renamed(name, tf) if ctx else spec.renamed(name, tf)
    if base in EXCLUDED_NON_STATIONARY:
        raise ContractError(
            f"{name!r} is a {EXCLUDED_NON_STATIONARY[base]} feature and is excluded "
            f"from the research contract by evidence: its support moves with the "
            f"instrument's price level, so a split learned on it does not transfer. "
            f"See the module docstring.")
    raise ContractError(
        f"{name!r} is not covered by the research feature contract. Add an explicit "
        f"entry to FEATURE_LIBRARY — a column must never be contracted by guesswork.")


@dataclass(frozen=True)
class CanonicalContract:
    """The ordered, fully-described feature contract of ONE model."""

    symbol: str
    entry_timeframe: str
    context_timeframes: Tuple[str, ...]
    feature_names: Tuple[str, ...]
    specs: Tuple[FeatureSpec, ...]
    schema_version: str = FEATURE_SCHEMA_VERSION

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def required_columns(self) -> Tuple[str, ...]:
        """Raw candle columns every feature in this contract reads."""
        cols = set()
        for s in self.specs:
            cols.update(s.requires)
        return tuple(sorted(cols))

    def specs_requiring(self, column: str) -> Tuple[FeatureSpec, ...]:
        return tuple(s for s in self.specs if column in s.requires)

    def minimum_history(self, timeframe: str) -> int:
        """Bars of `timeframe` needed before every feature on it is defined."""
        relevant = [s.minimum_history for s in self.specs if s.timeframe == timeframe]
        return max(relevant) if relevant else 0

    def describe(self) -> str:
        return (f"{self.symbol}/{self.entry_timeframe} ({self.schema_version}): "
                f"{self.feature_count} features, context={list(self.context_timeframes)}")


def build_contract(symbol: str, entry_timeframe: str,
                   feature_names: Sequence[str]) -> CanonicalContract:
    """Describe a model's feature list in full, in the model's own order.

    ``feature_names`` comes from the model manifest and is AUTHORITATIVE: the
    contract adapts to the model, never the other way round. Every name is
    resolved or the call fails.
    """
    names = tuple(feature_names)
    if not names:
        raise ContractError("a model contract with no features is not a contract")
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ContractError(f"feature list contains duplicates: {dupes}")

    specs = tuple(resolve(n, entry_timeframe) for n in names)
    contexts = tuple(sorted(
        {ctx for n in names if (ctx := strip_context(n)[1]) is not None},
        key=lambda t: {"M15": 0, "M30": 1, "H1": 2, "H4": 3}.get(t, 99),
    ))
    return CanonicalContract(
        symbol=symbol, entry_timeframe=entry_timeframe,
        context_timeframes=contexts, feature_names=names, specs=specs,
    )


def assert_scale_free(contract: CanonicalContract) -> None:
    """Refuse a contract that smuggles a price-scale feature into a model.

    Called by the loader. This is the mechanical enforcement of the research
    finding, not a style preference: a PRICE_UNIT or LEVEL column in the
    feature list means the model cannot transfer across price regimes, and
    KAIROS must not serve it as if it could.
    """
    offenders = [(s.name, s.stationarity) for s in contract.specs
                 if s.stationarity in NON_STATIONARY]
    if offenders:
        raise ContractError(
            f"{contract.describe()} includes non-scale-free features {offenders}. "
            f"The research contract is scale-free only.")


def contract_fingerprint(contract: CanonicalContract) -> str:
    """A stable hash over the ordered contract, INCLUDING each formula.

    Order and arithmetic are both part of the contract, so the fingerprint
    must move when either does. Renaming nothing but changing a lookback has
    to be visible, or the "same schema" claim means nothing.
    """
    import hashlib

    payload = "\n".join(
        f"{s.name}|{s.source}|{s.formula}|{s.timeframe}|{s.lookback}|"
        f"{s.minimum_history}|{s.dtype}|{s.unit}|{s.normalization}|"
        f"{s.availability}|{s.missing_policy}|{s.stationarity}"
        for s in contract.specs
    )
    head = f"{contract.schema_version}|{contract.symbol}|{contract.entry_timeframe}\n"
    return hashlib.sha256((head + payload).encode("utf-8")).hexdigest()
