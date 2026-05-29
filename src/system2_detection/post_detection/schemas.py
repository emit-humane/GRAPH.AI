"""Post-detection schemas (D8 Risk Fusion output)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FusedRiskOutput(BaseModel):
    """Output of the D8 Risk Fusion + Explanation Assembler.

    Field names follow the v2 build prompt. ``score_breakdown`` is a
    structured dict that carries every layer's score AND its fusion weight
    so downstream consumers (the dashboard, audit logs, evaluators) can
    reconstruct exactly how the final number was reached.
    """

    model_config = ConfigDict(populate_by_name=True)

    transaction_id: str
    sender_account: str
    transaction_risk_score: float                # 0-100
    group_risk_score: float                      # 0-100, community-level
    risk_level: str                              # Low | Medium | High | Critical
    risk_level_group: str                        # same enum, on group_risk_score
    score_breakdown: dict[str, Any] = Field(default_factory=dict)
    triggered_patterns: list[str] = Field(default_factory=list)
    top_shap_features: list[str] = Field(default_factory=list)
    explanation: str = ""
    fusion_mode: str = "static_weights"          # "static_weights" | "meta_learner"
