"""A small, explicit registry of research entry models.

Five states, each meaning something a human can check:

``RESEARCH``
    Imported and contract-checked. The research repository's own verdict on it
    was NOT a demonstrated edge.
``CANDIDATE``
    Statistically established in research, but calibration or economics did
    not pass.
``VALIDATED``
    Everything through the cost-aware validation backtest passed in research,
    but the one-shot out-of-sample look has not confirmed it.
``PRODUCTION_ELIGIBLE``
    The research gates AND the final out-of-sample look passed, the artifact
    hash verifies, KAIROS reproduces the research vectors bit-for-bit, and a
    human recorded an explicit approval. **Eligible is not enabled**: this
    status says the model MAY be considered for live use, never that it is
    live. Nothing in KAIROS reads this status to turn trading on.
``RETIRED``
    Superseded or withdrawn. Kept on disk for comparison, never served.

Status is not something a model can award itself. ``PRODUCTION_ELIGIBLE`` in
particular is reachable only through
:mod:`analysis.research.production_gate`, which requires every piece of
evidence to be present and a countersigned approval record — importing an
artifact, or loading one successfully, can never produce it.

Legacy artifacts under ``models/entry/`` are not in this registry and are not
reachable through it. Two vocabularies, two stores, no overlap — so a legacy
model can never be served against the research contract, or vice versa.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

RESEARCH = "RESEARCH"
CANDIDATE = "CANDIDATE"
VALIDATED = "VALIDATED"
PRODUCTION_ELIGIBLE = "PRODUCTION_ELIGIBLE"
RETIRED = "RETIRED"

STATUSES = (RESEARCH, CANDIDATE, VALIDATED, PRODUCTION_ELIGIBLE, RETIRED)

#: The ladder, weakest first. A status may only be set if every level below it
#: is also supported by evidence.
LADDER = (RESEARCH, CANDIDATE, VALIDATED, PRODUCTION_ELIGIBLE)

#: Statuses whose models the loader will serve for OFFLINE inference. RETIRED
#: is excluded on purpose: it stays on disk for comparison but must never
#: answer a query.
SERVABLE_STATUSES = frozenset({RESEARCH, CANDIDATE, VALIDATED, PRODUCTION_ELIGIBLE})

#: Statuses that may only be written by the production gate, never by an
#: import. Kept as data so the import script can assert it rather than
#: remember it.
GATED_STATUSES = frozenset({PRODUCTION_ELIGIBLE})

DEFAULT_REGISTRY_PATH = Path("models/research/registry.json")


class RegistryError(Exception):
    """The registry cannot answer, or is being asked for something incoherent."""


@dataclass(frozen=True)
class RegistryEntry:
    model_id: str
    symbol: str
    timeframe: str
    version: str
    status: str
    path: str
    model_hash: str
    feature_schema_version: str
    feature_count: int
    target: str
    research_verdict: str

    def describe(self) -> str:
        return (f"{self.model_id} [{self.status}] {self.symbol}/{self.timeframe} "
                f"v{self.version} ({self.feature_count} features, "
                f"verdict={self.research_verdict})")


@dataclass(frozen=True)
class ModelRegistry:
    entries: Dict[str, RegistryEntry]
    path: Path

    def get(self, model_id: str) -> RegistryEntry:
        try:
            return self.entries[model_id]
        except KeyError:
            raise RegistryError(
                f"no model {model_id!r} in {self.path}. Known: "
                f"{sorted(self.entries)}") from None

    def find(self, symbol: str, timeframe: str,
             statuses: Sequence[str] = tuple(SERVABLE_STATUSES),
             version: Optional[str] = None) -> List[RegistryEntry]:
        """Every registered model for one symbol/timeframe, newest version last.

        Symbol and timeframe are matched exactly. A near-miss is not a match:
        an EURUSD M15 model is not an approximation of a GBPUSD H1 one, and
        returning it would be worse than returning nothing.

        ``version`` narrows to one research generation (``research_v2`` /
        ``research_v3``). Generations coexist on purpose so they can be run
        side by side.
        """
        allowed = set(statuses)
        found = [e for e in self.entries.values()
                 if e.symbol == symbol and e.timeframe == timeframe
                 and e.status in allowed
                 and (version is None or e.version == version)]
        return sorted(found, key=lambda e: (e.version, e.model_id))

    def versions(self, symbol: str, timeframe: str) -> List[str]:
        return sorted({e.version for e in self.entries.values()
                       if e.symbol == symbol and e.timeframe == timeframe})

    def resolve(self, symbol: str, timeframe: str,
                statuses: Sequence[str] = tuple(SERVABLE_STATUSES),
                version: Optional[str] = None) -> RegistryEntry:
        """The single model for a symbol/timeframe, or an error explaining why not.

        When several generations are registered and no ``version`` is given,
        this raises rather than picking. Choosing between two models on the
        caller's behalf — by recency, by status, by anything — would be exactly
        the implicit second source of truth this registry exists to prevent.
        """
        found = self.find(symbol, timeframe, statuses, version)
        if not found:
            registered = sorted({(e.symbol, e.timeframe, e.version)
                                 for e in self.entries.values()})
            raise RegistryError(
                f"no model registered for {symbol}/{timeframe}"
                + (f" version={version!r}" if version else "")
                + f" with status in {sorted(statuses)}. Registered: {registered}")
        if len(found) > 1:
            raise RegistryError(
                f"{len(found)} models registered for {symbol}/{timeframe}: "
                f"{[(e.model_id, e.version) for e in found]}. Pass version= to "
                f"choose one, or retire the rest — letting the registry pick would "
                f"make the choice implicit, which is a second source of truth.")
        return found[0]

    def by_status(self, status: str) -> List[RegistryEntry]:
        return [e for e in self.entries.values() if e.status == status]

    def describe(self) -> str:
        counts = {s: len(self.by_status(s)) for s in STATUSES}
        return f"{len(self.entries)} models in {self.path}: {counts}"


def _parse_entry(raw: Dict) -> RegistryEntry:
    required = ("model_id", "symbol", "timeframe", "version", "status", "path",
                "model_hash", "feature_schema_version", "feature_count", "target",
                "research_verdict")
    missing = [k for k in required if k not in raw]
    if missing:
        raise RegistryError(f"registry entry {raw.get('model_id')!r} is missing {missing}")
    if raw["status"] not in STATUSES:
        raise RegistryError(
            f"registry entry {raw['model_id']!r} has status {raw['status']!r}; "
            f"valid: {list(STATUSES)}")
    return RegistryEntry(
        model_id=raw["model_id"], symbol=raw["symbol"], timeframe=raw["timeframe"],
        version=raw["version"], status=raw["status"], path=raw["path"],
        model_hash=raw["model_hash"],
        feature_schema_version=raw["feature_schema_version"],
        feature_count=int(raw["feature_count"]), target=raw["target"],
        research_verdict=raw["research_verdict"],
    )


def load_registry(path=DEFAULT_REGISTRY_PATH) -> ModelRegistry:
    p = Path(path)
    if not p.exists():
        raise RegistryError(
            f"no registry at {p}. Import a research model first: "
            f"python scripts/import_research_model.py --help")
    try:
        with p.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"{p} is not valid JSON: {exc}") from exc

    models = raw.get("models")
    if not isinstance(models, list):
        raise RegistryError(f"{p}: expected a 'models' array")
    entries: Dict[str, RegistryEntry] = {}
    for item in models:
        entry = _parse_entry(item)
        if entry.model_id in entries:
            raise RegistryError(f"{p}: duplicate model_id {entry.model_id!r}")
        entries[entry.model_id] = entry
    return ModelRegistry(entries=entries, path=p)


def merge_entries(new: Sequence[RegistryEntry],
                  path=DEFAULT_REGISTRY_PATH) -> List[RegistryEntry]:
    """New entries on top of whatever is already registered, keyed by model_id.

    An import of one research generation must not silently unregister another:
    the generations are kept side by side so they can be compared, which is the
    whole reason the artifacts are preserved rather than replaced.
    """
    existing: Dict[str, RegistryEntry] = {}
    try:
        existing = dict(load_registry(path).entries)
    except RegistryError:
        pass
    existing.update({e.model_id: e for e in new})
    return list(existing.values())


def write_registry(entries: Sequence[RegistryEntry], path=DEFAULT_REGISTRY_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "kairos-research-registry-1",
        "note": ("Offline research integration only. No PRODUCTION status exists here "
                 "and nothing in this registry is wired to live trading."),
        "models": [
            {
                "model_id": e.model_id, "symbol": e.symbol, "timeframe": e.timeframe,
                "version": e.version, "status": e.status, "path": e.path,
                "model_hash": e.model_hash,
                "feature_schema_version": e.feature_schema_version,
                "feature_count": e.feature_count, "target": e.target,
                "research_verdict": e.research_verdict,
            }
            for e in sorted(entries, key=lambda x: x.model_id)
        ],
    }
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return p
