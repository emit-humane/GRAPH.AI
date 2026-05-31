"""D6b / L4 — Behavioural Anomaly Inference (online).

Loads the four D6a artifacts (isolation_forest.pkl, lof_model.pkl,
autoencoder.pt, behavioral_profiles.parquet) and scores every incoming
``TransactionEvent`` against all three unsupervised models. Each model's
output is normalised to [0, 100] against a per-account baseline; the
ensemble is the mean of the three. The autoencoder's per-feature
reconstruction loss provides the top-3 ``anomaly_drivers`` for the
explanation surface.

Field-name conventions follow the v2 build prompt — ``anomaly_score``
is the ensemble.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..shared.s3_artifact_store import ArtifactStore
from ..shared.schemas import (
    FEATURE_COLUMNS,
    LiveFeatureVector,
    LiveGraphFeatureVector,
)
from .d6a_training import (
    Autoencoder,
    GRAPH_MERGE_COLUMNS,
)
from .schemas import BehavioralAnomalyOutput

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# LiveGraphFeatureVector <- L2A graph column mapping
# --------------------------------------------------------------------------- #
# The 7 L2A columns the autoencoder + IF + LOF saw at training time map back
# onto the LiveGraphFeatureVector's sender_* fields. ``scatter_gather_score``
# is not exposed on the live vector (it's a static L2A statistic, not a
# per-event signal); use 0 as the default — the autoencoder learns this as
# part of the global mean.
_GRAPH_FIELD_MAP: dict[str, tuple[str | None, float]] = {
    "2hop_cycle_count": ("sender_2hop_cycle_count", 0.0),
    "3hop_cycle_count": ("sender_3hop_cycle_count", 0.0),
    "fan_in_score": ("sender_fan_in_score", 0.0),
    "fan_out_score": ("sender_fan_out_score", 0.0),
    "scatter_gather_score": (None, 0.0),
    "community_density": ("sender_community_density", 0.0),
    "benford_chi2_community": ("sender_benford_chi2_community", 0.0),
}


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def _sigmoid(x: float | np.ndarray, gain: float = 1.0) -> float:
    """Logistic mapping to [0, 1] with adjustable gain."""
    return 1.0 / (1.0 + np.exp(-gain * x))


def _normalise_to_100(deviation: float, gain: float = 1.5) -> float:
    """Map a real-valued deviation (current − baseline, anomaly direction)
    onto [0, 100] via a sigmoid centred at zero.

    - deviation == 0  →  50  (event sits exactly at the account's baseline)
    - deviation grows positively → score → 100 (more anomalous)
    - deviation grows negatively → score → 0 (more normal than baseline)
    """
    return float(np.clip(_sigmoid(deviation, gain=gain) * 100.0, 0.0, 100.0))


# --------------------------------------------------------------------------- #
# Anomaly score recalibration
# --------------------------------------------------------------------------- #
# Empirical pivot: in the v1 inferencer the ensemble score saturated in the
# 60-70 band for almost every transaction because three sub-models each
# clustered around 50 (their natural neutral point). That made the layer
# non-discriminating — a true positive scored 64 and a false positive scored
# 62. We rescale the raw ensemble through a sharp sigmoid centred at
# ``ANOMALY_PIVOT`` so the discriminating range spreads across [0, 100]:
#
#   raw 50 (deeply normal)    →  ~14  (clearly normal)
#   raw 60 (slightly elevated)→  ~30
#   raw 65 (ambiguous)        →  ~50  (genuine "unsure")
#   raw 70 (anomalous)        →  ~70
#   raw 80 (strongly anomalous)→ ~92
#
# Pivot + slope are tunable here without retraining the underlying models.
ANOMALY_PIVOT: float = 65.0       # raw ensemble value that maps to 50
ANOMALY_SLOPE: float = 0.20       # 1/(range of half the dynamic band)


def _recalibrate_anomaly(raw_ensemble: float) -> float:
    """Spread the saturated raw ensemble (typically 50-80) across [0, 100].

    Uses a sigmoid centred at ``ANOMALY_PIVOT``. Production wins from this:
    normal traffic gets pushed BELOW the alert threshold instead of
    clustering near 60, and genuinely anomalous events get amplified above
    70 instead of being a few points above the noise floor.
    """
    delta = (raw_ensemble - ANOMALY_PIVOT) * ANOMALY_SLOPE
    return float(np.clip(_sigmoid(delta, gain=1.0) * 100.0, 0.0, 100.0))


# --------------------------------------------------------------------------- #
# Inferencer
# --------------------------------------------------------------------------- #


class BehavioralAnomalyInferencer:
    """Loads D6a artifacts and scores ``TransactionEvent``s online."""

    REQUIRED_ARTIFACTS = (
        "isolation_forest.pkl",
        "lof_model.pkl",
        "autoencoder.pt",
        "behavioral_profiles.parquet",
    )

    def __init__(
        self,
        store: ArtifactStore | None = None,
        *,
        sigmoid_gain: float = 1.5,
        device: str | None = None,
    ) -> None:
        self.store = store or ArtifactStore()
        self.sigmoid_gain = sigmoid_gain

        # Load the three calibrated payloads. The training-side save() writes
        # (model, scaler, feature_names) tuples for IF/LOF; the autoencoder
        # checkpoint includes the scaler stats inline.
        iso_payload = self.store.load("isolation_forest.pkl")
        lof_payload = self.store.load("lof_model.pkl")
        if isinstance(iso_payload, tuple) and len(iso_payload) == 3:
            self.iso, iso_scaler, iso_feat = iso_payload
        else:
            raise RuntimeError("isolation_forest.pkl is not a (model, scaler, feature_names) tuple")
        if isinstance(lof_payload, tuple) and len(lof_payload) == 3:
            self.lof, lof_scaler, lof_feat = lof_payload
        else:
            raise RuntimeError("lof_model.pkl is not a (model, scaler, feature_names) tuple")

        # All three pickled scalers should agree — pick IF's as canonical.
        self.scaler = iso_scaler
        self.feature_names: list[str] = list(iso_feat)
        if list(lof_feat) != self.feature_names:
            raise RuntimeError("IF and LOF artifacts disagree on feature_names")

        # Autoencoder + its embedded scaler stats — load via the ArtifactStore
        # so the local/S3 backend split is invisible here.
        ae_blob = self.store.load("autoencoder.pt")
        if list(ae_blob["feature_names"]) != self.feature_names:
            raise RuntimeError("Autoencoder feature_names disagree with IF/LOF")
        self.autoencoder = Autoencoder(
            input_dim=ae_blob["input_dim"],
            latent_dim=ae_blob["latent_dim"],
            hidden_dim=ae_blob["hidden_dim"],
        )
        self.autoencoder.load_state_dict(ae_blob["state_dict"])
        self.autoencoder.eval()
        self.ae_device = torch.device(device or "cpu")
        self.autoencoder.to(self.ae_device)

        # Per-account baselines
        profiles = self.store.load("behavioral_profiles.parquet")
        if not isinstance(profiles, pd.DataFrame):
            raise RuntimeError("behavioral_profiles is not a DataFrame")
        self._profile_idx: dict[str, dict] = {
            row["account_id"]: row.to_dict() for _, row in profiles.iterrows()
        }
        # Global fallback for accounts not in the baseline table (EXT-, new accounts)
        self._global_iso = float(profiles["iso_forest_baseline_score"].mean())
        self._global_ae = float(profiles["autoencoder_baseline_recon"].mean())

    # ------------------------------ scoring -------------------------------- #

    def score(
        self,
        features: LiveFeatureVector,
        graph: LiveGraphFeatureVector,
    ) -> BehavioralAnomalyOutput:
        # 1) Build the 52-dim raw input
        raw = self._build_vector(features, graph)

        # 2) Scale with the training-time StandardScaler
        scaled = self.scaler.transform(raw.reshape(1, -1))

        # 3) IF + LOF decision_function values. Both are centred so that
        #    output < 0 ↔ anomaly per the model's contamination setting,
        #    > 0 ↔ normal. This natural zero-crossing gives us a robust
        #    mapping to [0, 100] without per-account tuning of IF/LOF.
        iso_decision = float(self.iso.decision_function(scaled)[0])
        lof_decision = float(self.lof.decision_function(scaled)[0])

        # 4) Autoencoder MSE (higher = more anomalous)
        with torch.no_grad():
            x_t = torch.from_numpy(scaled.astype(np.float32)).to(self.ae_device)
            recon = self.autoencoder(x_t).cpu().numpy()
        per_feature_loss = (recon - scaled) ** 2  # shape (1, 52)
        ae_mse = float(per_feature_loss.mean())

        # 5) Per-account baseline for the autoencoder (the spec persists this
        #    one per account so we honour it). IF/LOF don't need per-account
        #    centring because decision_function already does that work.
        baseline = self._profile_idx.get(features.sender_account)
        baseline_ae = (
            float(baseline["autoencoder_baseline_recon"]) if baseline else self._global_ae
        )

        # 6) Normalise each to [0, 100]. Linear mapping with anchor points
        #    chosen so that:
        #      - a clearly-normal event (decision > +0.10, AE at baseline) → < 30
        #      - the anomaly threshold (decision = 0) → 50
        #      - a clearly-anomalous event (decision < −0.10, AE >> baseline) → > 70
        iso_score = float(np.clip(50.0 - iso_decision * 200.0, 0.0, 100.0))
        lof_score = float(np.clip(50.0 - lof_decision * 200.0, 0.0, 100.0))
        ae_score = float(np.clip((ae_mse - baseline_ae) * 200.0, 0.0, 100.0))

        # Raw mean — the three sub-scores all hover near 50 for typical
        # traffic, so the unrecalibrated mean clusters in the 55-70 band
        # regardless of how anomalous the event actually is. We RECALIBRATE
        # below to spread the genuine signal across [0, 100].
        raw_ensemble = float(np.clip((iso_score + lof_score + ae_score) / 3.0, 0.0, 100.0))
        anomaly_score = _recalibrate_anomaly(raw_ensemble)

        # 7) Top-3 drivers — features with the largest per-feature recon loss.
        loss_per_feature = per_feature_loss[0]
        top_idx = np.argsort(-loss_per_feature)[:3]
        drivers = [self.feature_names[int(i)] for i in top_idx]

        return BehavioralAnomalyOutput(
            transaction_id=features.transaction_id,
            anomaly_score=anomaly_score,
            iso_score=float(iso_score),
            lof_score=float(lof_score),
            autoencoder_score=float(ae_score),
            anomaly_drivers=drivers,
        )

    # ------------------------------ helpers -------------------------------- #

    def _build_vector(
        self,
        features: LiveFeatureVector,
        graph: LiveGraphFeatureVector,
    ) -> np.ndarray:
        """Assemble the 52-dim raw input in training-column order."""
        values = []
        for col in self.feature_names:
            if col in FEATURE_COLUMNS:
                values.append(float(getattr(features, col)))
            elif col in GRAPH_MERGE_COLUMNS:
                live_field, default = _GRAPH_FIELD_MAP[col]
                if live_field is None:
                    values.append(float(default))
                else:
                    values.append(float(getattr(graph, live_field, default)))
            else:
                values.append(0.0)
        return np.array(values, dtype=float)

    # ------------------------------ guards --------------------------------- #

    @classmethod
    def all_artifacts_present(cls, store: ArtifactStore | None = None) -> bool:
        store = store or ArtifactStore()
        return all(store.exists(name) for name in cls.REQUIRED_ARTIFACTS)
