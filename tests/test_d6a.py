"""Tests for the Layer 4 (D6a) behavioural anomaly training.

Build a small generator dataset, run S1 + L2A on it, then run D6a end-to-end
and assert:
    * all four artifacts persisted
    * autoencoder checkpoint round-trips (state + metadata preserved)
    * behavioral_profiles row count == number of distinct sender accounts
    * IF / LOF / AE models can score new vectors after load
    * no labels are read during training (unsupervised contract)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from src.system1_generator import g1_account_builder, g2_normal_generator, g3_scenario_injector, g4_splitter
from src.system1_generator.common import CONFIG_PATH, GeneratorConfig
from src.system2_detection.layer2_graph.d4a_graph_preprocessor import build_graph_features
from src.system2_detection.layer4_anomaly import d6a_training
from src.system2_detection.shared import s1_feature_engineering, s2_multigraph_builder


@pytest.fixture(scope="session")
def d6_fixture(tmp_path_factory):
    work = tmp_path_factory.mktemp("d6-fixture")
    cfg = GeneratorConfig.from_file(CONFIG_PATH).override(
        seed=GeneratorConfig.from_file(CONFIG_PATH).seed + 3000,
        num_accounts=300,
        num_historical_transactions=6_000,
        num_stream_transactions=0,
        fraud_ratio=0.02,
    )
    # Generate data
    ctx = g1_account_builder.build_accounts(cfg)
    normal_df = g2_normal_generator.generate_normal(cfg, ctx)
    combined, _, gt = g3_scenario_injector.inject_scenarios(cfg, ctx, normal_df)
    hist_df, _ = g4_splitter.split_and_export(combined, gt, out_dir=work)
    historical_csv = work / "historical_transactions.csv"

    # S1 + L2A artifacts in this tempdir
    artifacts_dir = work / "artifacts"
    artifacts_dir.mkdir()
    features, scaler = s1_feature_engineering.build_features(historical_csv=historical_csv)
    s1_feature_engineering.save_features(features, scaler, out_dir=artifacts_dir)

    G, _, _ = s2_multigraph_builder.build_multigraph(
        historical_csv=historical_csv, accounts_df=ctx.accounts
    )
    gf = build_graph_features(G, betweenness_sample_k=None)
    gf.to_parquet(artifacts_dir / "graph_features.parquet", index=False)

    # D6a training — keep AE epochs low for test speed
    test_cfg = d6a_training.TrainingConfig(ae_epochs=5, ae_batch_size=128, lof_max_samples=None)
    paths = d6a_training.run(
        behavioral_path=artifacts_dir / "behavioral_features.parquet",
        graph_path=artifacts_dir / "graph_features.parquet",
        out_dir=artifacts_dir,
        cfg=test_cfg,
    )

    return {
        "work": work,
        "artifacts_dir": artifacts_dir,
        "hist_df": hist_df,
        "behavioral": features,
        "graph_features": gf,
        "paths": paths,
        "cfg": test_cfg,
    }


# --------------------------------------------------------------------------- #
# Artifact presence
# --------------------------------------------------------------------------- #


def test_all_four_artifacts_persisted(d6_fixture):
    d = d6_fixture["artifacts_dir"]
    for name in (
        "isolation_forest.pkl",
        "lof_model.pkl",
        "autoencoder.pt",
        "behavioral_profiles.parquet",
    ):
        assert (d / name).exists(), f"missing {name}"


def test_iso_pkl_loadable_and_predicts(d6_fixture):
    iso, scaler, feat_names = joblib.load(d6_fixture["artifacts_dir"] / "isolation_forest.pkl")
    assert iso.n_estimators == 200
    assert scaler.n_features_in_ == len(feat_names) == 52
    # Predict on a random row to exercise the loaded pair
    X = np.zeros((1, len(feat_names)), dtype=float)
    scaled = scaler.transform(X)
    s = iso.score_samples(scaled)
    assert s.shape == (1,)


def test_lof_pkl_loadable(d6_fixture):
    lof, scaler, feat_names = joblib.load(d6_fixture["artifacts_dir"] / "lof_model.pkl")
    assert lof.n_neighbors == 30
    assert getattr(lof, "novelty", False) is True
    X = np.zeros((2, len(feat_names)), dtype=float)
    scores = lof.score_samples(scaler.transform(X))
    assert scores.shape == (2,)


# --------------------------------------------------------------------------- #
# Autoencoder checkpoint round-trip
# --------------------------------------------------------------------------- #


def test_autoencoder_checkpoint_roundtrip(d6_fixture):
    ckpt_path = d6_fixture["artifacts_dir"] / "autoencoder.pt"
    model, blob = d6a_training.load_autoencoder(ckpt_path)
    # Metadata
    assert blob["input_dim"] == 52
    assert blob["latent_dim"] == 16
    assert blob["hidden_dim"] == 32
    assert len(blob["feature_names"]) == 52
    assert blob["scaler_mean"].shape == (52,)
    assert blob["scaler_scale"].shape == (52,)
    # Architecture
    assert isinstance(model, d6a_training.Autoencoder)
    assert model.input_dim == 52
    # Run a forward pass through the loaded model
    x = torch.zeros((3, 52), dtype=torch.float32)
    y = model(x)
    assert y.shape == (3, 52)
    # State-dict equality vs a fresh load
    blob2 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k, v in model.state_dict().items():
        assert torch.allclose(v, blob2["state_dict"][k])


# --------------------------------------------------------------------------- #
# behavioral_profiles row count + schema
# --------------------------------------------------------------------------- #


def test_profiles_row_count_matches_distinct_senders(d6_fixture):
    profiles = pd.read_parquet(d6_fixture["artifacts_dir"] / "behavioral_profiles.parquet")
    expected = d6_fixture["behavioral"]["sender_account"].nunique()
    assert len(profiles) == expected, f"profiles has {len(profiles)} rows, distinct senders = {expected}"


def test_profiles_schema_exact(d6_fixture):
    profiles = pd.read_parquet(d6_fixture["artifacts_dir"] / "behavioral_profiles.parquet")
    assert list(profiles.columns) == list(d6a_training.PROFILE_COLUMNS)
    # All accounts must be unique
    assert profiles["account_id"].is_unique
    # Aggregates must be finite
    for col in (
        "mean_amount", "std_amount", "p95_amount",
        "typical_tx_velocity_24h", "typical_beneficiary_count",
        "iso_forest_baseline_score", "autoencoder_baseline_recon",
    ):
        assert np.isfinite(profiles[col].to_numpy()).all(), f"non-finite values in {col}"


# --------------------------------------------------------------------------- #
# Layer-3 / Layer-4 boundary — no labels touched
# --------------------------------------------------------------------------- #


def test_training_uses_no_labels(monkeypatch, tmp_path):
    """D6a is unsupervised; it must not read any labeled file
    (warmup_labeled_transactions.csv or hidden_ground_truth.csv).
    """
    # Build tiny upstream artifacts in tmp_path
    base = GeneratorConfig.from_file(CONFIG_PATH)
    cfg = base.override(seed=base.seed + 4000, num_accounts=80,
                        num_historical_transactions=400, num_stream_transactions=0,
                        fraud_ratio=0.05)
    ctx = g1_account_builder.build_accounts(cfg)
    normal_df = g2_normal_generator.generate_normal(cfg, ctx)
    combined, _, gt = g3_scenario_injector.inject_scenarios(cfg, ctx, normal_df)
    g4_splitter.split_and_export(combined, gt, out_dir=tmp_path)

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    features, scaler = s1_feature_engineering.build_features(historical_csv=tmp_path / "historical_transactions.csv")
    s1_feature_engineering.save_features(features, scaler, out_dir=artifacts_dir)
    G, _, _ = s2_multigraph_builder.build_multigraph(
        historical_csv=tmp_path / "historical_transactions.csv", accounts_df=ctx.accounts,
    )
    gf = build_graph_features(G, betweenness_sample_k=None)
    gf.to_parquet(artifacts_dir / "graph_features.parquet", index=False)

    # Guard read_csv + read_parquet against label leakage
    real_read_csv = pd.read_csv
    real_read_parquet = pd.read_parquet
    banned_substrings = ("warmup_labeled", "hidden_ground_truth")

    def _guard(label: str, real):
        def inner(*args, **kwargs):
            for a in list(args) + list(kwargs.values()):
                if isinstance(a, (str, Path)) and any(b in str(a).lower() for b in banned_substrings):
                    raise AssertionError(
                        f"D6a {label} attempted to read a labeled file: {a}"
                    )
            return real(*args, **kwargs)
        return inner

    monkeypatch.setattr(pd, "read_csv", _guard("read_csv", real_read_csv))
    monkeypatch.setattr(pd, "read_parquet", _guard("read_parquet", real_read_parquet))

    test_cfg = d6a_training.TrainingConfig(ae_epochs=3, ae_batch_size=64, lof_max_samples=None)
    paths = d6a_training.run(
        behavioral_path=artifacts_dir / "behavioral_features.parquet",
        graph_path=artifacts_dir / "graph_features.parquet",
        out_dir=artifacts_dir,
        cfg=test_cfg,
    )
    assert (artifacts_dir / "isolation_forest.pkl").exists()
    assert (artifacts_dir / "behavioral_profiles.parquet").exists()


# --------------------------------------------------------------------------- #
# Layer-5 boundary — no Node2Vec / TGN imports from this module
# --------------------------------------------------------------------------- #


def test_d6_module_does_not_load_node2vec_or_tgn_temporal():
    """L4 uses torch (for the autoencoder) but must not pull node2vec or
    torch_geometric / torch_geometric_temporal — those belong to Layer 5.
    """
    import importlib
    import sys

    mod = "src.system2_detection.layer4_anomaly.d6a_training"
    if mod in sys.modules:
        del sys.modules[mod]
    before = set(sys.modules.keys())
    importlib.import_module(mod)
    newly = set(sys.modules.keys()) - before
    banned = {"node2vec", "torch_geometric", "torch_geometric_temporal"}
    for name in newly:
        top = name.split(".")[0]
        assert top not in banned, f"L4 unexpectedly loaded {name}"
