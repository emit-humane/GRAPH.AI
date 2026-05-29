"""Layer 1 rule-engine output schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RuleEngineOutput(BaseModel):
    """Output of one RuleEngine.evaluate() call.

    Matches the architecture's L1 RuleEngineOutput definition. ``rule_score``
    is normalised to [0, 100] via the spec's
    ``min(100, raw / max_possible_raw * 100)`` formula.
    """

    model_config = ConfigDict(populate_by_name=True)

    transaction_id: str
    rule_score: float
    triggered_rules: list[str]
    rule_explanations: list[str]
    rule_count: int
