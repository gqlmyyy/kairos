"""Trade-management orchestrator.

Runs the layers for one open trade, in exactly this order:

     1. Trade Profile Management   (entry only, resolved once)
     2. Initial Protection         (entry only)
     3. Risk Governor              (entry gate; never touches open trades)
     4. Intrabar Management
     5. Signal Flip Exit           (hard override, highest priority)
     6. Unified Exit Score
     7. Trade Age Management + Time Stop
     8. Break-even
     9. Partial Take Profit
    10. Adaptive Trailing Stop
    11. Minimum Modify Distance    (final filter before any write)

Steps 1-3 belong to the entry path and are exposed as separate helpers; this
module's ``manage_open_trade`` covers steps 4-11, which is what the post-entry
loop calls.

The orchestrator is the only component here that produces side effects. Every
layer is pure and returns a LayerResult; composing them and deciding what
actually reaches the broker happens in one place, on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

from . import tm_config as C
from . import layer1_breakeven, layer1_min_modify_distance
from . import layer2_exit_score, layer2_signal_flip
from . import layer3_adaptive_trailing, layer4_trade_age, layer5_partial_tp
from . import layer6_trade_profile
from .layer1_intrabar import IntrabarState
from .layer2_exit_probability import ProbabilityHistory, compute_exit_probability
from .types import LayerResult, ModifyRequest, TradeContext

logger = get_logger("tm.orchestrator")


@dataclass
class ManagementOutcome:
    """What the orchestrator concluded for one trade, one pass."""

    order_id: str
    close_full: bool = False
    close_fraction: float = 0.0
    modify: Optional[ModifyRequest] = None
    reasons: List[str] = field(default_factory=list)
    layer_results: List[LayerResult] = field(default_factory=list)
    rejected_modify_reason: str = ""
    partial_level_index: Optional[int] = None

    @property
    def has_action(self) -> bool:
        return bool(self.close_full or self.close_fraction > 0 or self.modify)


class TradeManagementOrchestrator:
    """Stateful across loop passes: probability history and intrabar tracking."""

    def __init__(self) -> None:
        self._probability_history: Dict[str, ProbabilityHistory] = {}
        self.intrabar_state = IntrabarState()

    # ------------------------------------------------------------------ entry
    @staticmethod
    def resolve_profile(
        stored_profile: Optional[str] = None,
        regime: Optional[str] = None,
        mtf_aligned: Optional[bool] = None,
        trend_strength: Optional[float] = None,
    ) -> tuple:
        """Step 1 - Trade Profile Management."""
        return layer6_trade_profile.profile_for_trade(
            stored_profile=stored_profile,
            regime=regime,
            mtf_aligned=mtf_aligned,
            trend_strength=trend_strength,
        )

    @staticmethod
    def compute_entry_protection(
        symbol: str,
        atr: float,
        regime: str,
        account_equity: Optional[float] = None,
        settings: Optional[dict] = None,
    ):
        """Step 2 - Initial Protection."""
        from .layer1_initial_protection import compute_initial_protection

        return compute_initial_protection(symbol, atr, regime, account_equity, settings)

    @staticmethod
    def check_entry_gate(open_position_count: Optional[int] = None):
        """Step 3 - Risk Governor (entry gate only)."""
        from .layer1_risk_governor_gate import check_entry_allowed

        return check_entry_allowed(open_position_count)

    # ------------------------------------------------------------- management
    def manage_open_trade(
        self,
        ctx: TradeContext,
        settings: Optional[Dict[str, Any]] = None,
        signal: Optional[Dict[str, Any]] = None,
        readings: Optional[Dict[str, Any]] = None,
        exit_features: Optional[Dict[str, Any]] = None,
        is_new_candle: bool = True,
        mfe_samples: Optional[List[float]] = None,
    ) -> ManagementOutcome:
        """Steps 4-11 for a single open trade."""
        settings = settings or layer6_trade_profile.resolve_settings(ctx.profile)
        outcome = ManagementOutcome(order_id=ctx.order_id)

        def record(result: LayerResult) -> LayerResult:
            outcome.layer_results.append(result)
            if result.reasons:
                outcome.reasons.extend(result.reasons)
            return result

        # --- Step 5: Signal Flip Exit (hard override) -----------------------
        flip = record(layer2_signal_flip.check_signal_flip(ctx, signal, settings))
        if flip.terminal and flip.close_full:
            outcome.close_full = True
            return outcome

        # --- Step 6: Unified Exit Score -------------------------------------
        # Probability is recomputed once per closed candle, not per loop pass.
        probability = None
        if exit_features is not None and is_new_candle:
            history = self._probability_history.setdefault(ctx.order_id, ProbabilityHistory())
            probability = compute_exit_probability(exit_features, history, settings)

        exit_score = record(layer2_exit_score.evaluate(ctx, readings, probability, settings))
        if exit_score.close_full:
            outcome.close_full = True
            return outcome

        # --- Step 7: Trade Age + Time Stop ----------------------------------
        age = record(layer4_trade_age.evaluate(ctx, settings))
        if age.close_full:
            outcome.close_full = True
            return outcome

        age_scale = float(age.meta.get("age_scale", 1.0))
        allow_large_changes = bool(age.meta.get("allow_large_changes", True))

        # --- Step 8: Break-even ---------------------------------------------
        breakeven = record(layer1_breakeven.apply_breakeven(ctx, settings))

        # --- Step 9: Partial Take Profit ------------------------------------
        partial = record(layer5_partial_tp.evaluate(ctx, settings))
        if partial.close_fraction > 0:
            outcome.close_fraction = partial.close_fraction
            outcome.partial_level_index = partial.meta.get("level_index")

        # --- Step 10: Adaptive Trailing -------------------------------------
        # During the settle phase the trail is held back; break-even may still
        # act, since moving the stop to entry is protective, not a "big change".
        trailing = LayerResult.noop("adaptive_trailing", "suppressed_during_settle_phase")
        if allow_large_changes:
            pullback_tolerance = layer5_partial_tp.compute_pullback_tolerance(mfe_samples, settings)
            trailing = layer3_adaptive_trailing.evaluate(
                ctx, age_scale=age_scale,
                pullback_tolerance=pullback_tolerance,
                settings=settings,
            )
        record(trailing)

        # --- Compose the SL/TP proposal -------------------------------------
        candidate_sl = self._pick_stop(ctx, [breakeven.new_sl, trailing.new_sl])
        candidate_tp = trailing.new_tp

        if candidate_sl is None and candidate_tp is None:
            return outcome

        request = ModifyRequest(
            order_id=ctx.order_id,
            symbol=ctx.symbol,
            direction=ctx.direction,
            new_sl=candidate_sl,
            new_tp=candidate_tp,
            reasons=tuple(
                r for layer in (breakeven, trailing) for r in layer.reasons
            ),
        )

        # --- Step 11: Minimum Modify Distance (final filter) ----------------
        verdict = layer1_min_modify_distance.filter_modification(request, ctx, settings)
        if verdict.approved:
            outcome.modify = verdict.request
        else:
            outcome.rejected_modify_reason = verdict.reason
            logger.debug(
                "[TM] order=%s modify rejected: %s", ctx.order_id, verdict.reason
            )

        return outcome

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _pick_stop(ctx: TradeContext, candidates: List[Optional[float]]) -> Optional[float]:
        """Choose the most protective stop among the layers' proposals.

        Only stops that improve protection are eligible; among those the one
        closest to price wins, because a tighter protective stop is strictly
        better than a looser one once both are valid.
        """
        valid = [c for c in candidates if c is not None and c > 0 and ctx.sl_is_improvement(c)]
        if not valid:
            return None
        return max(valid) if ctx.is_buy else min(valid)

    def forget_trade(self, order_id: str) -> None:
        """Drop per-trade state once a position is gone."""
        self._probability_history.pop(str(order_id), None)
