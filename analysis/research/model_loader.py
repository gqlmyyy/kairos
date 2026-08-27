"""Load a research model, or refuse it with MODEL_NOT_COMPATIBLE.

The chain, in order, every time
-------------------------------
1. registry lookup      — is this model registered, and is its status servable?
2. artifact present     — model.joblib and model_card.json both on disk
3. card valid           — every required field, no defaults invented
4. model hash           — the bytes on disk are the bytes the card describes
5. symbol               — the card's symbol equals the requested symbol
6. timeframe            — likewise, exactly
7. feature schema       — the card's schema version equals the one this build implements
8. contract resolvable  — every feature name resolves in the canonical contract
9. scale-free           — no LEVEL / PRICE_UNIT / BROKER_UNIT feature smuggled in
10. artifact agreement  — the estimator's own feature count and names agree
11. feature order       — the card's order equals the artifact's, element by element

Any failure raises :class:`ModelNotCompatible`. There is no fallback, no
truncation, no zero-padding, no positional guessing, and no "load it anyway
and see". A model whose contract cannot be satisfied must block, not be
coaxed into producing a plausible-looking number — that is precisely how the
legacy path served a 65-feature artifact ten values for months.

Order matters as much as membership: the same names permuted is a different
model input, and both XGBoost and a scikit-learn pipeline would accept it
silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from analysis.research import contract as C
from analysis.research import model_card as mc
from analysis.research import model_registry as reg

STATUS_NOT_COMPATIBLE = "MODEL_NOT_COMPATIBLE"


class ModelNotCompatible(Exception):
    """This model may not be served for this request. Never recovered from."""

    status = STATUS_NOT_COMPATIBLE

    def __init__(self, reason: str, *, stage: str = ""):
        self.reason = reason
        self.stage = stage
        super().__init__(f"{STATUS_NOT_COMPATIBLE}"
                         + (f" [{stage}]" if stage else "") + f": {reason}")


@dataclass(frozen=True)
class LoadedModel:
    """A model that passed every check, plus the contract it speaks."""

    entry: reg.RegistryEntry
    card: mc.ModelCard
    contract: C.CanonicalContract
    estimator: Any
    calibrator: Any
    directory: Path

    @property
    def symbol(self) -> str:
        return self.card.symbol

    @property
    def timeframe(self) -> str:
        return self.card.timeframe

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return self.card.feature_list

    def describe(self) -> str:
        return (f"{self.card.describe()} | {self.contract.describe()} | "
                f"status={self.entry.status}")


def _artifact_feature_names(estimator: Any) -> Optional[Tuple[str, ...]]:
    """The feature names the fitted estimator itself carries, if any.

    Both shapes the research repo ships are handled: a bare XGBClassifier and
    a sklearn Pipeline whose final step is the estimator. When the artifact
    records names they are AUTHORITATIVE over the card, because they are what
    the library will match on at predict time.
    """
    for obj in (estimator, getattr(estimator, "_final_estimator", None)):
        if obj is None:
            continue
        names = getattr(obj, "feature_names_in_", None)
        if names is not None and len(names):
            return tuple(str(n) for n in names)
        booster = getattr(obj, "get_booster", None)
        if callable(booster):
            try:
                bn = booster().feature_names
            except Exception:  # noqa: BLE001 - an unreadable booster is checked elsewhere
                bn = None
            if bn:
                return tuple(str(n) for n in bn)
    return None


def _artifact_feature_count(estimator: Any) -> Optional[int]:
    for obj in (estimator, getattr(estimator, "_final_estimator", None)):
        if obj is None:
            continue
        n = getattr(obj, "n_features_in_", None)
        if isinstance(n, (int,)) and n > 0:
            return int(n)
    return None


def load_model(
    symbol: str,
    timeframe: str,
    *,
    registry_path=reg.DEFAULT_REGISTRY_PATH,
    statuses: Sequence[str] = tuple(reg.SERVABLE_STATUSES),
    version: Optional[str] = None,
    verify_hash: bool = True,
) -> LoadedModel:
    """Resolve, validate and load the model for one symbol/timeframe.

    ``version`` selects a research generation when more than one is registered.
    With several registered and no version given the registry raises rather
    than choosing — see :meth:`ModelRegistry.resolve`.
    """
    # 1. registry
    try:
        registry = reg.load_registry(registry_path)
        entry = registry.resolve(symbol, timeframe, statuses, version)
    except reg.RegistryError as exc:
        raise ModelNotCompatible(str(exc), stage="registry") from exc

    directory = Path(entry.path)
    model_path = directory / mc.MODEL_FILENAME

    # 2. artifact present
    if not model_path.exists():
        raise ModelNotCompatible(
            f"registry points at {model_path}, which does not exist", stage="artifact")

    # 3. card
    try:
        card = mc.load(directory)
    except mc.ModelCardError as exc:
        raise ModelNotCompatible(str(exc), stage="model_card") from exc

    # 4. model hash — the bytes on disk are the bytes the card describes
    if verify_hash:
        actual = mc.sha256_file(model_path)
        if actual != card.model_hash:
            raise ModelNotCompatible(
                f"model hash mismatch for {model_path}: card declares "
                f"{card.model_hash[:16]}..., file is {actual[:16]}.... The artifact "
                f"changed after it was imported; re-import it rather than editing "
                f"the card.", stage="model_hash")
        if actual != entry.model_hash:
            raise ModelNotCompatible(
                f"registry hash {entry.model_hash[:16]}... disagrees with the artifact "
                f"{actual[:16]}...", stage="model_hash")

    # 5/6. symbol and timeframe — exact, never approximate
    if card.symbol != symbol:
        raise ModelNotCompatible(
            f"symbol mismatch: model was trained for {card.symbol}, requested for "
            f"{symbol}. Instruments differ in price scale, tick size and session; a "
            f"model is not transferable between them.", stage="symbol")
    if card.timeframe != timeframe:
        raise ModelNotCompatible(
            f"timeframe mismatch: model was trained for {card.timeframe}, requested "
            f"for {timeframe}. The horizon, the ATR that sizes TP/SL and every "
            f"lookback are all measured in bars of the training timeframe.",
            stage="timeframe")

    # 7. feature schema version
    if card.feature_schema_version != mc.CURRENT_FEATURE_SCHEMA_VERSION:
        raise ModelNotCompatible(
            f"feature schema mismatch: model was built for "
            f"{card.feature_schema_version!r}, this build implements "
            f"{mc.CURRENT_FEATURE_SCHEMA_VERSION!r}. Same names with different "
            f"arithmetic is still a different model input.", stage="feature_schema")

    # 8/9. the canonical contract, and the scale-free rule
    try:
        contract = C.build_contract(card.symbol, card.timeframe, card.feature_list)
        C.assert_scale_free(contract)
    except C.ContractError as exc:
        raise ModelNotCompatible(str(exc), stage="feature_contract") from exc

    if tuple(contract.context_timeframes) != tuple(card.context_timeframes):
        raise ModelNotCompatible(
            f"context timeframes implied by the feature list "
            f"{list(contract.context_timeframes)} disagree with the card's "
            f"{list(card.context_timeframes)}", stage="feature_contract")

    # load the artifact only once everything declared about it checks out
    try:
        import joblib
        payload = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001 - any load failure blocks
        raise ModelNotCompatible(
            f"{model_path} could not be loaded: {type(exc).__name__}: {exc}",
            stage="artifact") from exc

    if not isinstance(payload, dict) or "model" not in payload:
        raise ModelNotCompatible(
            f"{model_path} does not hold the expected "
            f"{{'model', 'calibrator', 'features'}} bundle", stage="artifact")
    estimator = payload["model"]
    calibrator = payload.get("calibrator")

    # 10. the artifact's own account of itself
    artifact_count = _artifact_feature_count(estimator)
    if artifact_count is not None and artifact_count != card.feature_count:
        raise ModelNotCompatible(
            f"the card declares {card.feature_count} features but the fitted "
            f"estimator expects {artifact_count}", stage="feature_count")

    bundled = payload.get("features")
    if bundled is not None and tuple(bundled) != card.feature_list:
        raise ModelNotCompatible(
            _order_diff(tuple(bundled), card.feature_list,
                        "the artifact's bundled feature list", "the card"),
            stage="feature_order")

    # 11. order, element by element
    artifact_names = _artifact_feature_names(estimator)
    if artifact_names is not None and artifact_names != card.feature_list:
        raise ModelNotCompatible(
            _order_diff(artifact_names, card.feature_list,
                        "the fitted estimator", "the card"),
            stage="feature_order")

    return LoadedModel(entry=entry, card=card, contract=contract,
                       estimator=estimator, calibrator=calibrator,
                       directory=directory)


def _order_diff(a: Tuple[str, ...], b: Tuple[str, ...], a_label: str, b_label: str) -> str:
    """A diff that names the first real difference rather than dumping two lists."""
    if set(a) == set(b):
        first = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        return (f"feature ORDER mismatch at index {first}: {a_label} has {a[first]!r}, "
                f"{b_label} has {b[first]!r} (same names, different order — which is a "
                f"different model input, accepted silently by both XGBoost and sklearn)")
    only_a = [n for n in a if n not in set(b)][:5]
    only_b = [n for n in b if n not in set(a)][:5]
    return (f"feature NAME mismatch: only in {a_label}={only_a}, "
            f"only in {b_label}={only_b}")
