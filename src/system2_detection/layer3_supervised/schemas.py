"""Layer 3 (D5) supervised-detector output schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SupervisedOutput(BaseModel):
    """Output of the supervised inferencer for one transaction.

    Args:
        transaction_id: Event identifier.
        supervised_score: Calibrated risk score in [0, 100]
            (= fraud_probability * 100).
        fraud_probability: Calibrated probability of suspiciousness from
            the primary XGBoost model.
        shap_values: ``{feature_name: shap_contribution}`` for the top
            contributing features of THIS prediction (positive values push
            toward suspicious; negative values pull toward normal).
        top_features: Ordered list of the feature names with the largest
            |shap| for this prediction. First entry has the largest
            absolute contribution.
        model_agreement: 1 - (max(p) - min(p)) across xgb/lgbm/rf — close
            to 1 when all three classifiers concur, close to 0 when they
            disagree. Useful as a confidence multiplier in P1 fusion.
    """

    model_config = ConfigDict(populate_by_name=True)

    transaction_id: str
    supervised_score: float
    fraud_probability: float
    shap_values: dict[str, float] = Field(default_factory=dict)
    top_features: list[str] = Field(default_factory=list)
    model_agreement: float = 1.0
