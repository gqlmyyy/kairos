"""Entry v2 — QUARANTINED / INVALIDATED. Do not train from this package.

The forensic audit in ENTRY_PIPELINE_AUDIT.md established, against this
package's own output, that the dataset it builds is not usable:

  * Look-ahead. `dataset_builder` grids on H1 timestamps and attaches the
    "latest H4 candle at or before t". Candle stamps are OPEN times, so that is
    the candle which opens at t and closes three hours later. Measured:
    h4_close equals the H1 close at t+3h for 96.6% / 96.4% / 80.5% of rows
    (EURUSD / GBPUSD / XAUUSD).
  * Entry price. A guard in `feature_engineering` that is always false lets
    `entry_price` fall through to `h4_ema_50` — in 100.0000% of 24,851 rows.
    The displacement from the real close averages 0.92 ATR against barriers of
    1.0 and 1.5 ATR.
  * Timeframe. "H4" indicators are computed over the forward-filled H1 grid.
    The shipped h4_rsi_14 matches an H1-grid RSI exactly (err 0.0000) and a
    true H4 RSI not at all (err ~12.5 points).
  * Direction. No direction column exists, so every row is labelled BUY, and
    the 65-feature schema has no direction feature.

Any metric computed on that dataset — including the deployed model's own test
score — is meaningless, so the artifact it produced must never be promoted.

What is still allowed here: importing these modules to read, audit or test
them. `analysis/entry_v2/inference.py` remains importable and still fails
closed through the feature contract.

What is blocked: running the dataset, feature, label and training entrypoints,
which would regenerate invalid data or a model trained on it. Set
KAIROS_ALLOW_INVALIDATED_ENTRY_V2=1 to override for forensic work — it will not
let anything reach production, since promotion is separately gated by
analysis.models.production_model_guard.
"""

import os

QUARANTINE_ENV_VAR = "KAIROS_ALLOW_INVALIDATED_ENTRY_V2"

QUARANTINE_REASON = (
    "analysis/entry_v2 is quarantined: proven look-ahead (H4 candle attached "
    "before it closes), entry_price resolving to h4_ema_50 in 100% of rows, "
    "H4 indicators computed on a repeated H1 grid, and no direction column. "
    "See ENTRY_PIPELINE_AUDIT.md."
)


class InvalidatedPipelineError(RuntimeError):
    """Raised when quarantined entry_v2 code is executed rather than inspected."""


def quarantine_override_enabled() -> bool:
    return os.environ.get(QUARANTINE_ENV_VAR, "").strip() in {"1", "true", "TRUE", "yes"}


def refuse_invalidated_pipeline(component: str) -> None:
    """Block a quarantined entrypoint unless explicitly overridden."""
    if quarantine_override_enabled():
        return
    raise InvalidatedPipelineError(
        f"{component} is disabled. {QUARANTINE_REASON} "
        f"Set {QUARANTINE_ENV_VAR}=1 only for forensic work."
    )
