"""Layer 5 (D7) inferencer output schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GNNInferenceOutput(BaseModel):
    """Output of the D7 TGN / Node2Vec inferencer.

    Field names follow the v2 build prompt (``tgn_score`` for the aggregated
    structural anomaly, ``link_prediction_score`` = 1 − link probability,
    ``embedding_drift`` = max of sender/receiver drift). ``tgn_mode``
    surfaces whether the full TGN was used or the lighter node2vec
    fallback — useful for downstream fusion to weight the score accordingly.
    """

    model_config = ConfigDict(populate_by_name=True)

    transaction_id: str
    tgn_score: float                                  # 0-100, aggregated structural anomaly
    link_prediction_score: float                      # 1 - link probability, in [0, 1]
    embedding_drift: float                            # max(sender_drift, receiver_drift), in [0, 1]
    temporal_graph_explanations: list[str] = Field(default_factory=list)
    subgraph_evidence: dict[str, Any] = Field(default_factory=dict)
    tgn_mode: str = "tgn"                             # "tgn" or "node2vec_fallback"
