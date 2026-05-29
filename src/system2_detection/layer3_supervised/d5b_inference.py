"""D5b — Supervised Detector Inference (online).

Wraps the four ``supervised_*`` artifacts produced by D5a into a single
``SupervisedInferencer`` that consumes the three live vectors and emits a
``SupervisedOutput`` per transaction:

    .score(features, graph, rule_out) -> SupervisedOutput

The feature vector is rebuilt in the same canonical order D5a saved alongside
the calibrated XGBoost. The primary fraud_probability comes from XGBoost
(calibrated), SHAP values explain THIS prediction on the raw XGBoost model,
and model_agreement compares the three classifiers' probabilities.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..layer1_rules.rule_definitions import RULES
from ..shared.s3_artifact_store import ArtifactNotFoundError, ArtifactStore
from ..shared.schemas import (
    FEATURE_COLUMNS,
    LiveFeatureVector,
    LiveGraphFeatureVector,
)
from .d5a_training import (
    GRAPH_TRAINING_COLS,
    RULE_INDICATOR_COLS,
    TRAINING_FEATURE_COLUMNS,
)
from .schemas import SupervisedOutput

logger = logging.getLogger(__name__)

# --- LiveGraphFeatureVector field <- training-matrix column mapping --- #
# Live vector fields are sender-prefixed; the training matrix uses L2A names
# with a ``g_`` prefix. Map each training column to the corresponding field on
# the live vector (with a sensible default if the field isn't exposed live).
_LIVE_GRAPH_FIELD: dict[str, tuple[str, Any]] = {
    "g_in_degree": ("sender_in_degree", 0),
    "g_out_degree": ("sender_out_degree", 0),
    "g_in_degree_unique": ("sender_in_degree_unique", 0),
    "g_out_degree_unique": ("sender_out_degree_unique", 0),
    "g_pagerank_score": ("sender_pagerank", 0.0),
    "g_betweenness_centrality": ("sender_betweenness", 0.0),
    "g_clustering_coefficient": (None, 0.0),  # not exposed live
    "g_two_hop_cycle_count": ("sender_2hop_cycle_count", 0),
    "g_three_hop_cycle_count": ("sender_3hop_cycle_count", 0),
    "g_fan_in_score": ("sender_fan_in_score", 0.0),
    "g_fan_out_score": ("sender_fan_out_score", 0.0),
    "g_scatter_gather_score": (None, 0.0),  # not exposed live
    "g_community_id": ("sender_community_id", -1),
    "g_community_size": ("sender_community_size", 0),
    "g_community_density": ("sender_community_density", 0.0),
    "g_ego_in_volume_7d": (None, 0.0),       # not exposed live
    "g_ego_out_volume_7d": (None, 0.0),      # not exposed live
    "g_volume_asymmetry": (None, 0.0),       # not exposed live
    "g_benford_chi2_community": ("sender_benford_chi2_community", 0.0),
}


def _value_for_training_column(
    column: str,
    features: LiveFeatureVector,
    graph: LiveGraphFeatureVector,
    rule_score: float,
    triggered: set[str],
) -> float:
    """Lookup the right value for one training-matrix column."""
    if column in FEATURE_COLUMNS:
        return float(getattr(features, column))
    if column in GRAPH_TRAINING_COLS:
        live_field, default = _LIVE_GRAPH_FIELD[column]
        if live_field is None:
            return float(default)
        return float(getattr(graph, live_field, default))
    if column == "rule_score":
        return float(rule_score)
    if column.startswith("rule_R"):
        return float(column.split("rule_")[1] in triggered)
    return 0.0


def build_feature_vector(
    features: LiveFeatureVector,
    graph: LiveGraphFeatureVector,
    rule_out,
    feature_names: list[str],
) -> np.ndarray:
    """Return a 1-D float vector aligned to ``feature_names``."""
    triggered = set(getattr(rule_out, "triggered_rules", []) or [])
    rule_score = float(getattr(rule_out, "rule_score", 0.0))
    vec = np.array(
        [
            _value_for_training_column(col, features, graph, rule_score, triggered)
            for col in feature_names
        ],
        dtype=float,
    )
    return vec


# --------------------------------------------------------------------------- #
# Inferencer
# --------------------------------------------------------------------------- #


class SupervisedInferencer:
    """Loads the four D5a artifacts and scores live events."""

    REQUIRED_ARTIFACTS = (
        "supervised_xgb.pkl",
        "supervised_lgbm.pkl",
        "supervised_rf.pkl",
        "supervised_shap_explainer.pkl",
    )

    def __init__(
        self,
        store: ArtifactStore | None = None,
        top_n_features: int = 15,
    ) -> None:
        self.store = store or ArtifactStore()
        self.top_n_features = top_n_features

        # Each artifact is a (model_or_explainer, feature_names) tuple.
        self.xgb_cal, feat_xgb = self.store.load("supervised_xgb.pkl")
        self.lgb_cal, feat_lgb = self.store.load("supervised_lgbm.pkl")
        self.rf_cal, feat_rf = self.store.load("supervised_rf.pkl")
        self.explainer, feat_shap = self.store.load("supervised_shap_explainer.pkl")

        # All four artifacts must agree on the feature ordering.
        if not (feat_xgb == feat_lgb == feat_rf == feat_shap):
            raise RuntimeError("supervised artifacts disagree on feature_names ordering")
        self.feature_names: list[str] = list(feat_xgb)

    # ------------------------------ scoring -------------------------------- #

    def score(
        self,
        features: LiveFeatureVector,
        graph: LiveGraphFeatureVector,
        rule_out,
    ) -> SupervisedOutput:
        vec = build_feature_vector(features, graph, rule_out, self.feature_names)
        # Wrap as a single-row DataFrame so models fit with feature names
        # don't emit "X does not have valid feature names" warnings every call.
        vec_df = pd.DataFrame([vec], columns=self.feature_names)

        # Primary probability — calibrated XGBoost
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            p_xgb = float(self.xgb_cal.predict_proba(vec_df)[0, 1])
            p_lgb = float(self.lgb_cal.predict_proba(vec_df)[0, 1])
            p_rf = float(self.rf_cal.predict_proba(vec_df)[0, 1])
        vec_2d = vec.reshape(1, -1)

        # SHAP values on the raw XGBoost (binary outputs a 1-D array per row)
        try:
            sv = self.explainer.shap_values(vec_2d)
            # shap returns either ndarray (n_samples, n_features) or list[ndarray]
            if isinstance(sv, list):
                # binary classifier returns list of length 2 with class shap arrays
                shap_row = np.asarray(sv[-1])[0]
            else:
                shap_row = np.asarray(sv)[0]
        except Exception as exc:
            logger.warning("shap explainer failed: %s", exc)
            shap_row = np.zeros(len(self.feature_names), dtype=float)

        order = np.argsort(-np.abs(shap_row))
        top_idx = order[: self.top_n_features]
        shap_values = {self.feature_names[int(i)]: float(shap_row[int(i)]) for i in top_idx}
        top_features = [self.feature_names[int(i)] for i in top_idx]

        # Agreement: 1 - (max - min) of the three calibrated probabilities
        probs = np.array([p_xgb, p_lgb, p_rf])
        agreement = float(np.clip(1.0 - (probs.max() - probs.min()), 0.0, 1.0))

        return SupervisedOutput(
            transaction_id=features.transaction_id,
            supervised_score=float(np.clip(p_xgb * 100.0, 0.0, 100.0)),
            fraud_probability=p_xgb,
            shap_values=shap_values,
            top_features=top_features,
            model_agreement=agreement,
        )

    # ------------------------------ helpers -------------------------------- #

    @classmethod
    def all_artifacts_present(cls, store: ArtifactStore | None = None) -> bool:
        store = store or ArtifactStore()
        return all(store.exists(name) for name in cls.REQUIRED_ARTIFACTS)
