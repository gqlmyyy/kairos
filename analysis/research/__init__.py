"""Canonical integration of the xgbooost research entry models.

This package is the ONE contract KAIROS speaks to a research model with. It is
deliberately separate from ``analysis/models/`` (the legacy 10-feature entry
path) and ``analysis/entry_v2/`` (an abandoned 65-feature experiment): those
remain readable and runnable for comparison, but nothing in this package reads
them, and nothing in them writes here.

Scope. Entry models only. Nothing here trains, executes orders, touches an
account, or imports MetaTrader5 — the whole package runs offline on Linux from
stored OHLC candles.

Layout::

    contract.py       the canonical feature contract (formula/unit/availability/…)
    indicators.py     causal indicator formulas
    price_action.py   causal price-action formulas
    engine.py         per-timeframe features + MTF alignment
    candles.py        candle sources and the column-availability declaration
    availability.py   VALID / MISSING / INVALID / UNAVAILABLE
    model_registry.py RESEARCH / CANDIDATE / VALIDATED / RETIRED
    model_loader.py   load -> validate -> serve, or MODEL_NOT_COMPATIBLE
    inference.py      contract-driven prediction, p_win semantics
    replay.py         offline historical replay
"""
