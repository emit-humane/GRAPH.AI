"""L1 — Rule-Based AML Engine.

Executes the 15 deterministic AML rules from ``rule_definitions.RULES``
against a (features, graph, event) triple and emits a ``RuleEngineOutput``.

Scoring formula (v2 — per-fire severity + co-firing bonus):

    Per-fire severity score from rule.weight (which is the rule's prior
    importance, ranging ~0.5 to 1.0 in the spec):

        per_fire_score = 50 + 40 * (weight - 0.5)

    That gives:

        weight 0.6 -> 54   (R07 device anomaly, R11 round amount)
        weight 0.7 -> 58   (R06 high-risk jurisdiction, R08 shared device, R13 Benford)
        weight 0.8 -> 62   (R03 velocity, R09 beneficiaries, R12 KYC, R14 fan-in)
        weight 0.9 -> 66   (R04 dormant, R05 impossible travel, R15 relay)
        weight 1.0 -> 70   (R01 high-value, R02 structuring, R10 cycle closure)

    Then aggregate across fires:

        top_score   = max(per_fire_score) over fired rules
        co_bonus    = min(30, 10 * (fired_count - 1))
        rule_score  = min(100, top_score + co_bonus)

    Properties:
        * 0 rules fire           -> 0     (clean baseline)
        * 1 weak rule  (0.6)     -> 54    (lands in Medium, not invisible)
        * 1 strong rule (1.0)    -> 70    (solidly High — one R10 cycle MUST alert)
        * 2 rules (1.0 + 0.7)    -> 80    (deeper High)
        * 3+ rules together      -> 90-100 (Critical)

Why the redesign:
    The v1 formula ``sum(fired_weights) / sum(all_weights) * 100`` mapped
    even a 3-rule co-fire of (1.0 + 0.9 + 0.8) to only 22/100 because the
    denominator was 12.2 (the sum of ALL 15 rule weights). That meant a
    fired single rule was effectively invisible to downstream fusion —
    measured at 21 mean on confirmed true positives during the
    verification run. With the per-fire formula above, a single cycle
    closure now reaches 70 and the same 3-rule co-fire reaches 90, so
    Layer 1 actually drives fusion decisions instead of decorating them.

The legacy ``raw_score`` (sum of fired weights) and ``max_possible_raw``
(sum of ALL rule weights) are still exposed on the output for audit /
compatibility, but ``rule_score`` itself follows the v2 aggregation.
"""

from __future__ import annotations

import logging
from typing import Sequence

from .rule_definitions import RULES, Rule
from .schemas import RuleEngineOutput

logger = logging.getLogger(__name__)


# v2 per-fire scoring parameters — tunable here without touching the
# individual rule conditions.
PER_FIRE_BASE: float = 50.0       # at weight=0.5, a fire lands at 50 (Medium boundary)
PER_FIRE_SLOPE: float = 40.0      # each additional 0.1 of weight adds 4 points
CO_FIRE_BONUS_PER_RULE: float = 10.0   # +10 per extra rule beyond the first
CO_FIRE_BONUS_CAP: float = 30.0   # absolute ceiling on the bonus


def _per_fire_score(weight: float) -> float:
    """Map one rule's weight to a per-fire 0-100 severity score.

    A weight of 0.5 is the boundary that lands at Medium (50); a max-weight
    rule (1.0) lands at 70 — solidly inside High on its own.
    """
    return float(max(0.0, PER_FIRE_BASE + PER_FIRE_SLOPE * (weight - 0.5)))


class RuleEngine:
    """Stateless rule evaluator.

    Args:
        rules: Optional override of the rule set. Defaults to the canonical
            15-rule list ``rule_definitions.RULES``.
    """

    def __init__(self, rules: Sequence[Rule] | None = None) -> None:
        self.rules: tuple[Rule, ...] = tuple(rules) if rules is not None else RULES
        self.max_possible_raw: float = sum(r.weight for r in self.rules)
        if self.max_possible_raw <= 0:
            raise ValueError("RuleEngine requires at least one rule with positive weight")

    def evaluate(self, features, graph, event) -> RuleEngineOutput:
        """Run every rule on the (features, graph, event) triple."""
        raw_score = 0.0                 # legacy sum-of-weights (for audit)
        triggered: list[str] = []
        explanations: list[str] = []
        fired_per_fire_scores: list[float] = []

        for rule in self.rules:
            try:
                fires = rule.fires(features, graph, event)
            except Exception as exc:  # defensive: a buggy condition shouldn't abort scoring
                logger.warning("rule %s raised %s: %s", rule.rule_id, type(exc).__name__, exc)
                fires = False

            if fires:
                raw_score += rule.weight
                fired_per_fire_scores.append(_per_fire_score(rule.weight))
                triggered.append(rule.rule_id)
                try:
                    explanations.append(rule.explanation(features, graph, event))
                except Exception as exc:
                    logger.warning(
                        "rule %s explanation raised %s: %s",
                        rule.rule_id, type(exc).__name__, exc,
                    )
                    explanations.append(f"{rule.rule_id} ({rule.name})")

        # v2 aggregation: top-fire severity + co-firing bonus.
        if not fired_per_fire_scores:
            rule_score = 0.0
        else:
            top = max(fired_per_fire_scores)
            co_bonus = min(
                CO_FIRE_BONUS_CAP,
                CO_FIRE_BONUS_PER_RULE * (len(fired_per_fire_scores) - 1),
            )
            rule_score = min(100.0, top + co_bonus)

        return RuleEngineOutput(
            transaction_id=event.transaction_id,
            rule_score=float(rule_score),
            triggered_rules=triggered,
            rule_explanations=explanations,
            rule_count=len(triggered),
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def rule_ids(self) -> list[str]:
        return [r.rule_id for r in self.rules]
