"""D6a / L4 — Behavioural Anomaly Model Training (offline).

Trains three complementary UNSUPERVISED detectors on the historical feature
matrix. Layer 4's purpose is to learn what *normal* looks like so it can flag
deviation — strictly no labels touched. This is the structural complement to
Layer 3 (D5), which learns known suspicious behaviour from labels. Keep
separate.

Spec reference: docs/architecture.md "L3A Behavioral Anomaly Model Training"
(the architecture's "LAYER 3" is now LAYER 4 / D6 in the v2 build playbook).

Input:
    artifacts/behavioral_features.parquet   (S1 — one row per transaction,
                                             45 feature columns + 2 keys)
    artifacts/graph_features.parquet        (L2A — one row per account,
                                             19 graph features)

Feature matrix construction:
    For every transaction row in behavioural_features, look up the sender's
    7 L2A graph columns (2hop_cycle_count, 3hop_cycle_count, fan_in_score,
    fan_out_score, scatter_gather_score, community_density,
    benford_chi2_community) and append → 52 features per row.
    Scale with a fresh StandardScaler fit on this 52-dim matrix (the spec
    mentions "feature_scaler.pkl from S1" but that scaler is 45-dim; fitting
    a new 52-dim scaler is what the spec actually requires for the
    autoencoder to train on well-conditioned inputs).

Models — exactly per the architecture:
    1. IsolationForest(n_estimators=200, contamination=0.01,
                       max_features=1.0, random_state=42)
    2. LocalOutlierFactor(n_neighbors=30, novelty=True,
                          contamination=0.01, metric='euclidean')
    3. Autoencoder: 52 → 32 → 16 → 32 → 52, MSE loss, 50 epochs, LR 1e-3,
                    batch 256

Outputs (under artifacts/):
    isolation_forest.pkl          trained IsolationForest
    lof_model.pkl                 trained LOF (novelty=True)
    autoencoder.pt                {state_dict, input_dim, latent_dim,
                                   feature_names, scaler_mean, scaler_scale}
    behavioral_profiles.parquet   per-account baseline stats
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Schema bits
# --------------------------------------------------------------------------- #

# The 7 L2A columns the spec calls out for the merge.
GRAPH_MERGE_COLUMNS: tuple[str, ...] = (
    "2hop_cycle_count",
    "3hop_cycle_count",
    "fan_in_score",
    "fan_out_score",
    "scatter_gather_score",
    "community_density",
    "benford_chi2_community",
)

# behavioral_features.parquet column convention
KEY_COLUMNS: tuple[str, ...] = ("transaction_id", "sender_account")

# behavioral_profiles output schema (per spec)
PROFILE_COLUMNS: tuple[str, ...] = (
    "account_id",
    "mean_amount",
    "std_amount",
    "p95_amount",
    "typical_tx_velocity_24h",
    "typical_beneficiary_count",
    "iso_forest_baseline_score",
    "autoencoder_baseline_recon",
)


# --------------------------------------------------------------------------- #
# Autoencoder
# --------------------------------------------------------------------------- #


class Autoencoder(nn.Module):
    """52 → 32 → 16 → 32 → 52 with ReLU between encoder + decoder layers."""

    def __init__(self, input_dim: int = 52, latent_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class TrainingConfig:
    """Hyperparameters with the architecture's spec values as defaults."""

    # Isolation Forest
    iso_n_estimators: int = 200
    iso_contamination: float = 0.01
    iso_max_features: float = 1.0
    seed: int = 42

    # LOF
    lof_n_neighbors: int = 30
    lof_contamination: float = 0.01
    lof_metric: str = "euclidean"

    # Autoencoder
    ae_input_dim: int = 52
    ae_latent_dim: int = 16
    ae_hidden_dim: int = 32
    ae_epochs: int = 50
    ae_learning_rate: float = 1e-3
    ae_batch_size: int = 256

    # LOF on very large datasets is O(N^2); allow subsampling for tractability.
    lof_max_samples: int | None = 30_000


# --------------------------------------------------------------------------- #
# Feature matrix construction
# --------------------------------------------------------------------------- #


def build_feature_matrix(
    behavioral_path: Path,
    graph_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, list[str], StandardScaler]:
    """Returns (joined_df, scaled_X, feature_names, fitted_scaler).

    joined_df keeps the keys so we can aggregate per account when building
    behavioral_profiles. scaled_X is the (N, 52) matrix the three models
    train on.
    """
    if not behavioral_path.exists():
        raise FileNotFoundError(
            f"behavioral_features.parquet missing at {behavioral_path} — run S1 first"
        )
    if not graph_path.exists():
        raise FileNotFoundError(
            f"graph_features.parquet missing at {graph_path} — run L2A first"
        )

    logger.info("[D6a] loading behavioural + graph features ...")
    bh = pd.read_parquet(behavioral_path)
    gf = pd.read_parquet(graph_path)

    # The behavioural matrix's keys are transaction_id + sender_account; every
    # other column is one of the 45 features.
    bh_keys = [c for c in KEY_COLUMNS if c in bh.columns]
    bh_features = [c for c in bh.columns if c not in bh_keys]
    if len(bh_features) != 45:
        logger.warning(
            "[D6a] behavioral_features has %d feature columns (expected 45)",
            len(bh_features),
        )

    # Pull just the 7 graph columns we need + the join key.
    missing = [c for c in GRAPH_MERGE_COLUMNS if c not in gf.columns]
    if missing:
        raise RuntimeError(f"graph_features.parquet missing required columns: {missing}")
    gf_subset = gf[["account_id", *GRAPH_MERGE_COLUMNS]].rename(
        columns={"account_id": "sender_account"}
    )

    joined = bh.merge(gf_subset, on="sender_account", how="left")
    # External / unseen senders may not be in L2A (rare but possible) — fill with 0.
    for col in GRAPH_MERGE_COLUMNS:
        joined[col] = joined[col].fillna(0.0)

    feature_names = bh_features + list(GRAPH_MERGE_COLUMNS)
    matrix = joined[feature_names].astype(float).to_numpy()
    logger.info("[D6a] feature matrix shape: %s", matrix.shape)

    # Fit a fresh StandardScaler over the 52-dim matrix. The spec mentions the
    # S1 scaler (45-dim); a 52-dim fit here matches the actual matrix shape
    # and gives the autoencoder well-conditioned inputs.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    logger.info("[D6a] fitted StandardScaler over %d features", scaler.n_features_in_)

    return joined, scaled, feature_names, scaler


# --------------------------------------------------------------------------- #
# Model fits
# --------------------------------------------------------------------------- #


def train_isolation_forest(X: np.ndarray, cfg: TrainingConfig) -> IsolationForest:
    logger.info("[D6a] training IsolationForest (n_est=%d, contamination=%.3f) ...",
                cfg.iso_n_estimators, cfg.iso_contamination)
    iso = IsolationForest(
        n_estimators=cfg.iso_n_estimators,
        contamination=cfg.iso_contamination,
        max_features=cfg.iso_max_features,
        random_state=cfg.seed,
        n_jobs=-1,
    )
    iso.fit(X)
    return iso


def train_lof(X: np.ndarray, cfg: TrainingConfig) -> tuple[LocalOutlierFactor, np.ndarray | None]:
    """Returns (fitted_LOF, train_subsample_indices_or_None).

    LOF is O(N^2) in the worst case; large historical matrices get subsampled
    to keep training tractable. The subsample is stratified-uniform and the
    indices are returned so callers know which rows the fit saw.
    """
    n = X.shape[0]
    if cfg.lof_max_samples and n > cfg.lof_max_samples:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(n, size=cfg.lof_max_samples, replace=False)
        idx.sort()
        X_fit = X[idx]
        logger.info("[D6a] training LOF on subsample (%d / %d, n_neighbors=%d) ...",
                    cfg.lof_max_samples, n, cfg.lof_n_neighbors)
    else:
        idx = None
        X_fit = X
        logger.info("[D6a] training LOF (n=%d, n_neighbors=%d) ...", n, cfg.lof_n_neighbors)
    lof = LocalOutlierFactor(
        n_neighbors=cfg.lof_n_neighbors,
        novelty=True,
        contamination=cfg.lof_contamination,
        metric=cfg.lof_metric,
        n_jobs=-1,
    )
    lof.fit(X_fit)
    return lof, idx


def train_autoencoder(
    X: np.ndarray,
    cfg: TrainingConfig,
    *,
    progress_every: int = 5,
) -> tuple[Autoencoder, list[float]]:
    """Trains the autoencoder and returns (model, per-epoch loss history)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("[D6a] training Autoencoder (%dD->%dD, epochs=%d, lr=%.4f, bs=%d, dev=%s) ...",
                cfg.ae_input_dim, cfg.ae_latent_dim, cfg.ae_epochs,
                cfg.ae_learning_rate, cfg.ae_batch_size, device)

    model = Autoencoder(cfg.ae_input_dim, cfg.ae_latent_dim, cfg.ae_hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.ae_learning_rate)
    loss_fn = nn.MSELoss()

    tensor = torch.from_numpy(X.astype(np.float32))
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=cfg.ae_batch_size,
        shuffle=True,
        drop_last=False,
    )

    history: list[float] = []
    model.train()
    for epoch in range(1, cfg.ae_epochs + 1):
        total = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optim.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optim.step()
            total += float(loss.detach().cpu())
            n_batches += 1
        epoch_loss = total / max(1, n_batches)
        history.append(epoch_loss)
        if epoch == 1 or epoch == cfg.ae_epochs or epoch % progress_every == 0:
            logger.info("[D6a]     epoch %02d/%d  loss=%.6f", epoch, cfg.ae_epochs, epoch_loss)

    model.eval()
    # Return on CPU so saving + downstream inference don't drag a GPU tensor along.
    model.to("cpu")
    return model, history


# --------------------------------------------------------------------------- #
# behavioral_profiles
# --------------------------------------------------------------------------- #


def build_profiles(
    joined: pd.DataFrame,
    scaled_X: np.ndarray,
    iso: IsolationForest,
    autoencoder: Autoencoder,
    feature_names: list[str],
) -> pd.DataFrame:
    """Aggregate per-account baselines + per-account mean of model scores."""
    logger.info("[D6a] computing per-row IF + AE scores ...")
    iso_scores = iso.score_samples(scaled_X)  # higher = more normal (negative-anomaly)
    with torch.no_grad():
        tensor = torch.from_numpy(scaled_X.astype(np.float32))
        recon = autoencoder(tensor).numpy()
    ae_recon = np.mean((recon - scaled_X) ** 2, axis=1)

    df = joined.copy()
    df["_iso_score"] = iso_scores
    df["_ae_recon"] = ae_recon

    logger.info("[D6a] aggregating per-account profiles ...")
    # S1's canonical column is "amount_raw"; older builds wrote "amount".
    amt_col = "amount_raw" if "amount_raw" in df.columns else ("amount" if "amount" in df.columns else None)
    if amt_col is None:
        raise RuntimeError("behavioral_features has no amount column to aggregate")
    grouped = df.groupby("sender_account", sort=False)
    profile = grouped.agg(
        mean_amount=(amt_col, "mean"),
        std_amount=(amt_col, "std"),
        p95_amount=(amt_col, lambda s: float(np.percentile(s, 95))),
        typical_tx_velocity_24h=("tx_velocity_24h", "mean"),
        typical_beneficiary_count=("beneficiary_count_7d", "mean"),
        iso_forest_baseline_score=("_iso_score", "mean"),
        autoencoder_baseline_recon=("_ae_recon", "mean"),
    ).reset_index().rename(columns={"sender_account": "account_id"})

    # Cast to spec types
    profile["std_amount"] = profile["std_amount"].fillna(0.0)
    profile = profile[list(PROFILE_COLUMNS)]
    profile["typical_tx_velocity_24h"] = profile["typical_tx_velocity_24h"].astype(np.float32)
    profile["typical_beneficiary_count"] = profile["typical_beneficiary_count"].astype(np.float32)
    profile["iso_forest_baseline_score"] = profile["iso_forest_baseline_score"].astype(np.float32)
    profile["autoencoder_baseline_recon"] = profile["autoencoder_baseline_recon"].astype(np.float32)
    return profile


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_artifacts(
    iso: IsolationForest,
    lof: LocalOutlierFactor,
    autoencoder: Autoencoder,
    scaler: StandardScaler,
    feature_names: list[str],
    profiles: pd.DataFrame,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    out_dir = out_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    iso_path = out_dir / "isolation_forest.pkl"
    joblib.dump((iso, scaler, list(feature_names)), iso_path)
    paths["isolation_forest.pkl"] = iso_path

    lof_path = out_dir / "lof_model.pkl"
    joblib.dump((lof, scaler, list(feature_names)), lof_path)
    paths["lof_model.pkl"] = lof_path

    ae_path = out_dir / "autoencoder.pt"
    torch.save(
        {
            "state_dict": autoencoder.state_dict(),
            "input_dim": autoencoder.input_dim,
            "latent_dim": autoencoder.latent_dim,
            "hidden_dim": autoencoder.hidden_dim,
            "feature_names": list(feature_names),
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
        },
        ae_path,
    )
    paths["autoencoder.pt"] = ae_path

    profiles_path = out_dir / "behavioral_profiles.parquet"
    profiles.to_parquet(profiles_path, index=False)
    paths["behavioral_profiles.parquet"] = profiles_path

    return paths


def load_autoencoder(checkpoint_path: Path) -> tuple[Autoencoder, dict]:
    """Round-trip helper for tests / inference. Returns (model, metadata)."""
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Autoencoder(
        input_dim=blob["input_dim"],
        latent_dim=blob["latent_dim"],
        hidden_dim=blob["hidden_dim"],
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #


def run(
    behavioral_path: Path | None = None,
    graph_path: Path | None = None,
    out_dir: Path | None = None,
    cfg: TrainingConfig | None = None,
) -> dict[str, Path]:
    cfg = cfg or TrainingConfig()
    behavioral_path = behavioral_path or (ARTIFACTS_DIR / "behavioral_features.parquet")
    graph_path = graph_path or (ARTIFACTS_DIR / "graph_features.parquet")

    joined, scaled_X, feat_names, scaler = build_feature_matrix(behavioral_path, graph_path)

    iso = train_isolation_forest(scaled_X, cfg)
    lof, _lof_idx = train_lof(scaled_X, cfg)
    ae, _history = train_autoencoder(scaled_X, cfg)
    profiles = build_profiles(joined, scaled_X, iso, ae, feat_names)

    paths = save_artifacts(iso, lof, ae, scaler, feat_names, profiles, out_dir=out_dir)
    return paths


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.system2_detection.layer4_anomaly.d6a_training",
        description="Train the Layer-4 unsupervised behavioural anomaly models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--behavioral-features", type=Path, default=None,
                   help="Default: artifacts/behavioral_features.parquet")
    p.add_argument("--graph-features", type=Path, default=None,
                   help="Default: artifacts/graph_features.parquet")
    p.add_argument("--artifacts-dir", type=Path, default=None,
                   help="Default: artifacts/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ae-epochs", type=int, default=50)
    p.add_argument("--ae-batch-size", type=int, default=256)
    p.add_argument("--ae-learning-rate", type=float, default=1e-3)
    p.add_argument("--lof-max-samples", type=int, default=30_000,
                   help="LOF subsample cap for very large matrices (set 0 for no cap).")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    cfg = TrainingConfig(
        seed=args.seed,
        ae_epochs=args.ae_epochs,
        ae_batch_size=args.ae_batch_size,
        ae_learning_rate=args.ae_learning_rate,
        lof_max_samples=(args.lof_max_samples or None),
    )
    t0 = time.time()
    paths = run(
        behavioral_path=args.behavioral_features,
        graph_path=args.graph_features,
        out_dir=args.artifacts_dir,
        cfg=cfg,
    )
    print("\n[D6a] wrote artifacts:")
    for name, p in paths.items():
        print(f"  {name:30s} {p}")
    print(f"\n[D6a] total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
