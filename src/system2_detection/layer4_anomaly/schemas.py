"""Layer 4 (D6) behavioural-anomaly inference output schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BehavioralAnomalyOutput(BaseModel):
    """Output of the D6 behavioural-anomaly inferencer.

    Field name conventions follow the v2 build prompt — ``anomaly_score`` is
    the ensemble (the architecture's older spec called it
    ``ensemble_anomaly_score``). All three sub-scores and the ensemble are
    in [0, 100]; higher = more anomalous.

    ``anomaly_drivers`` is the top-3 list of feature names that contributed
    most to the autoencoder's per-feature reconstruction loss for THIS event
    — useful for the dashboard's "why is this flagged" explanation.
    """

    model_config = ConfigDict(populate_by_name=True)

    transaction_id: str
    anomaly_score: float       # ensemble, 0-100 (mean of the three below)
    iso_score: float           # 0-100, derived from IsolationForest score_samples
    lof_score: float           # 0-100, derived from LOF score_samples
    autoencoder_score: float   # 0-100, derived from autoencoder reconstruction MSE
    anomaly_drivers: list[str] = Field(default_factory=list)
