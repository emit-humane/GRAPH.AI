"""L1 — Rule-Based AML Engine.

Executes the 15 deterministic AML rules from ``rule_definitions.RULES``
against a (features, graph, event) triple and emits a ``RuleEngineOutput``.

Scoring formula (architecture):
    raw_score        = sum(fire_i * weight_i)
    max_possible_raw = sum(weight_i for r in RULES)   # COMPUTED, never hardcoded
    rule_score       = min(100, raw_score / max_possible_raw * 100)
"""

from __future__ import annotations

import logging
from typing import Sequence

from .rule_definitions import RULES, Rule
from .schemas import RuleEngineOutput

logger = logging.getLogger(__name__)


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
        raw_score = 0.0
        triggered: list[str] = []
        explanations: list[str] = []

        for rule in self.rules:
            try:
                fires = rule.fires(features, graph, event)
            except Exception as exc:  # defensive: a buggy condition shouldn't abort scoring
                logger.warning("rule %s raised %s: %s", rule.rule_id, type(exc).__name__, exc)
                fires = False

            if fires:
                raw_score += rule.weight
                triggered.append(rule.rule_id)
                try:
                    explanations.append(rule.explanation(features, graph, event))
                except Exception as exc:
                    logger.warning(
                        "rule %s explanation raised %s: %s",
                        rule.rule_id, type(exc).__name__, exc,
                    )
                    explanations.append(f"{rule.rule_id} ({rule.name})")

        rule_score = min(100.0, (raw_score / self.max_possible_raw) * 100.0)
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
