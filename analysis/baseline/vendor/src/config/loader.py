"""Configuration loading. All experiment-affecting parameters live in config/*.yaml
so that experiment metadata can record exactly what was used (RULE 18)."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

_FILES = {
    "symbols": "symbols.yaml",
    "timeframes": "timeframes.yaml",
    "features": "features.yaml",
    "target": "target.yaml",
    "training": "training.yaml",
    "session_rules": "session_rules.yaml",
    "sources": "sources.yaml",
}


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Config:
    """Immutable-by-convention bundle of all config sections. Use `load_config()`."""

    symbols: dict = field(default_factory=dict)
    timeframes: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    target: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    session_rules: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    config_dir: str = str(CONFIG_DIR)

    def as_dict(self) -> dict:
        return {
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "features": self.features,
            "target": self.target,
            "training": self.training,
            "session_rules": self.session_rules,
            "sources": self.sources,
        }

    def deep_copy(self) -> "Config":
        return Config(**{k: copy.deepcopy(v) for k, v in self.as_dict().items()})

    # --- convenience accessors -------------------------------------------------
    def symbol_cfg(self, symbol: str) -> dict:
        try:
            return self.symbols["symbols"][symbol]
        except KeyError as e:
            raise KeyError(
                f"Symbol '{symbol}' not found in config/symbols.yaml. "
                f"Known symbols: {list(self.symbols.get('symbols', {}).keys())}"
            ) from e

    def timeframe_cfg(self, tf: str) -> dict:
        try:
            return self.timeframes["timeframes"][tf]
        except KeyError as e:
            raise KeyError(
                f"Timeframe '{tf}' not found in config/timeframes.yaml. "
                f"Known timeframes: {list(self.timeframes.get('timeframes', {}).keys())}"
            ) from e

    def column_map(self, symbol: str, file_format: str = "csv") -> dict:
        sym_cfg = self.symbol_cfg(symbol)
        if file_format == "json":
            override = sym_cfg.get("column_map_json")
            if override:
                return override
            return self.symbols["default_json_column_map"]
        override = sym_cfg.get("column_map")
        if override:
            return override
        return self.symbols["default_column_map"]

    def default_source(self) -> str:
        return self.sources.get("default_source", "mt5")

    def source_cfg(self, source: str) -> dict:
        try:
            return self.sources["sources"][source]
        except KeyError as e:
            raise KeyError(
                f"Unknown data source '{source}'. Known sources: "
                f"{list(self.sources.get('sources', {}).keys())} (config/sources.yaml)"
            ) from e

    def known_sources(self) -> list[str]:
        return list(self.sources.get("sources", {}).keys())

    def broker_symbol_override(self, symbol: str) -> str | None:
        """The exact broker instrument name pinned for this canonical symbol,
        or None to resolve it at the MT5 acquisition boundary. Used ONLY by
        src/data_acquisition/mt5_fetcher.py -- every other layer of the
        project uses the canonical symbol name."""
        value = self.symbol_cfg(symbol).get("broker_symbol")
        return value or None

    def data_timezone(self) -> str:
        return self.symbols.get("data_timezone", "UTC")

    def session_timezone(self) -> str:
        return self.symbols.get("session_timezone", "UTC")

    def context_timeframes_for(self, entry_timeframe: str) -> list[str]:
        try:
            return self.timeframes["mtf_stacks"][entry_timeframe]["context_timeframes"]
        except KeyError as e:
            raise KeyError(
                f"No mtf_stacks entry for entry_timeframe '{entry_timeframe}' in "
                f"config/timeframes.yaml. Known: {list(self.timeframes.get('mtf_stacks', {}).keys())}"
            ) from e

    def session_rules_for(self, symbol: str) -> list[dict]:
        """VERIFIED/HIGH-confidence session rules for `symbol` only. Empty
        list unless config/session_rules.yaml has real evidence-backed
        entries for it -- see that file's header for why it starts empty."""
        return self.session_rules.get("rules", {}).get(symbol, [])

    def low_confidence_session_candidates(self) -> dict:
        """Context-only candidates (e.g. the generic FX weekend convention).
        Never eligible to grant EXPECTED_SESSION_GAP -- see
        config/session_rules.yaml."""
        return self.session_rules.get("low_confidence_candidates", {})

    def session_calendars_for(self, symbol: str) -> list:
        """Session Calendar Engine calendars for one symbol, newest-effective
        first. Loaded from config/session_calendars/<symbol>.yaml; a file
        with `enabled: false` (the shipped default) yields nothing, so an
        unverified calendar can never gate anything."""
        from src.validation.session_calendar import load_session_calendars
        if not hasattr(self, "_session_calendar_cache"):
            self._session_calendar_cache = load_session_calendars(
                Path(self.config_dir) / "session_calendars")
        return self._session_calendar_cache.get(symbol, [])

    def data_gap_multiple(self) -> float:
        return self.session_rules.get("data_gap_multiple", 6)


def load_config(config_dir: str | os.PathLike | None = None) -> Config:
    """Load all config/*.yaml files into a single Config object."""
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    sections: dict[str, Any] = {}
    for key, filename in _FILES.items():
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(f"Required config file missing: {path}")
        sections[key] = _read_yaml(path)
    return Config(config_dir=str(directory), **sections)
