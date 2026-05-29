"""Tests for the Layer 4 (D6b) behavioural anomaly inferencer.

Trains a small D6a fixture once per session, then asserts:
    * A near-baseline vector scores < 30
    * A vector with an extreme z-score on one feature scores > 60 AND lists
      that feature in anomaly_drivers
    * Output is a valid BehavioralAnomalyOutput
    * All four sub-scores in [0, 100]
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.system1_generator import g1_account_builder, g2_normal_generator, g3_scenario_injector, g4_splitter
from src.system1_generator.common import CONFIG_PATH, GeneratorConfig
from src.system2_detection.layer2_graph.d4a_graph_preprocessor import build_graph_features
from src.system2_detection.layer4_anomaly import d6a_training, d6b_inference
from src.system2_detection.layer4_anomaly.schemas import BehavioralAnomalyOutput
from src.system2_detection.shared import s1_feature_engineering, s2_multigraph_builder
from src.system2_detection.shared.s3_artifact_store import ArtifactStore
from src.system2_detection.shared.schemas import LiveFeatureVector, LiveGraphFeatureVector


@pytest.fixture(scope="session")
def trained_d6(tmp_path_factory):
    work = tmp_path_factory.mktemp("d6b-fixture")
    base = GeneratorConfig.from_file(CONFIG_PATH)
    cfg = base.override(
        seed=base.seed + 7000,
        num_accounts=300,
        num_historical_transactions=6_000,
        num_stream_transactions=0,
        fraud_ratio=0.02,
    )
    # Generate + build upstream artifacts
    ctx = g1_account_builder.build_accounts(cfg)
    normal_df = g2_normal_generator.generate_normal(cfg, ctx)
    combined, _, gt = g3_scenario_injector.inject_scenarios(cfg, ctx, normal_df)
    g4_splitter.split_and_export(combined, gt, out_dir=work)
    historical_csv = work / "historical_transactions.csv"

    artifacts_dir = work / "artifacts"
    artifacts_dir.mkdir()
    features, scaler = s1_feature_engineering.build_features(historical_csv=historical_csv)
    s1_feature_engineering.save_features(features, scaler, out_dir=artifacts_dir)
    G, _, _ = s2_multigraph_builder.build_multigraph(
        historical_csv=historical_csv, accounts_df=ctx.accounts,
    )
    gf = build_graph_features(G, betweenness_sample_k=None)
    gf.to_parquet(artifacts_dir / "graph_features.parquet", index=False)

    # D6a training (small + fast)
    test_cfg = d6a_training.TrainingConfig(ae_epochs=10, ae_batch_size=128, lof_max_samples=None)
    d6a_training.run(
        behavioral_path=artifacts_dir / "behavioral_features.parquet",
        graph_path=artifacts_dir / "graph_features.parquet",
        out_dir=artifacts_dir,
        cfg=test_cfg,
    )

    store = ArtifactStore(base_dir=artifacts_dir)
    inferencer = d6b_inference.BehavioralAnomalyInferencer(store=store)
    behavioral_df = pd.read_parquet(artifacts_dir / "behavioral_features.parquet")
    graph_df = pd.read_parquet(artifacts_dir / "graph_features.parquet")

    return {
        "work": work,
        "artifacts_dir": artifacts_dir,
        "store": store,
        "inferencer": inferencer,
        "behavioral": behavioral_df,
        "graph_features": graph_df.set_index("account_id"),
    }


@pytest.fixture
def inferencer(trained_d6):
    return trained_d6["inferencer"]


# --------------------------------------------------------------------------- #
# Live*Vector builders for tests
# --------------------------------------------------------------------------- #


def _live_feature_vector_from_row(row: pd.Series, **overrides) -> LiveFeatureVector:
    """Build a normal-ish LiveFeatureVector from a behavioural row."""
    from src.system2_detection.shared.schemas import FEATURE_COLUMNS
    kwargs = {col: row[col] for col in FEATURE_COLUMNS}
    kwargs.update(overrides)
    return LiveFeatureVector(
        transaction_id=str(row["transaction_id"]),
        sender_account=str(row["sender_account"]),
        scaled_feature_vector=[],
        **kwargs,
    )


def _zero_graph_vec(transaction_id: str, sender: str, receiver: str = "B-test", **overrides):
    defaults = dict(
        sender_in_degree=0,
        sender_out_degree=0,
        sender_in_degree_unique=0,
        sender_out_degree_unique=0,
        sender_2hop_cycle_count=0,
        sender_3hop_cycle_count=0,
        sender_fan_in_score=0.0,
        sender_fan_out_score=0.0,
        sender_pagerank=0.0,
        sender_betweenness=0.0,
        sender_community_id=-1,
        sender_community_size=0,
        sender_community_density=0.0,
        sender_community_risk_score=0.0,
        sender_benford_chi2_community=0.0,
        receiver_in_degree=0,
        receiver_out_degree=0,
        receiver_community_id=-1,
        receiver_community_risk_score=0.0,
        edge_creates_cycle=False,
        cycle_length=0,
        shared_community=False,
    )
    defaults.update(overrides)
    return LiveGraphFeatureVector(
        transaction_id=transaction_id,
        sender_account=sender,
        receiver_account=receiver,
        **defaults,
    )


def _realistic_graph_vec(
    transaction_id: str, sender: str, graph_features: pd.DataFrame, receiver: str = "B-test", **overrides
):
    """Build a LiveGraphFeatureVector populated with the sender's L2A row.

    Uses defaults of 0 for senders not in L2A (e.g., EXT- placeholders).
    """
    if sender in graph_features.index:
        row = graph_features.loc[sender]
        defaults = dict(
            sender_in_degree=int(row["in_degree"]),
            sender_out_degree=int(row["out_degree"]),
            sender_in_degree_unique=int(row["in_degree_unique"]),
            sender_out_degree_unique=int(row["out_degree_unique"]),
            sender_2hop_cycle_count=int(row["2hop_cycle_count"]),
            sender_3hop_cycle_count=int(row["3hop_cycle_count"]),
            sender_fan_in_score=float(row["fan_in_score"]),
            sender_fan_out_score=float(row["fan_out_score"]),
            sender_pagerank=float(row["pagerank_score"]),
            sender_betweenness=float(row["betweenness_centrality"]),
            sender_community_id=int(row["community_id"]),
            sender_community_size=int(row["community_size"]),
            sender_community_density=float(row["community_density"]),
            sender_community_risk_score=0.0,
            sender_benford_chi2_community=float(row["benford_chi2_community"]),
        )
    else:
        defaults = dict(
            sender_in_degree=0, sender_out_degree=0,
            sender_in_degree_unique=0, sender_out_degree_unique=0,
            sender_2hop_cycle_count=0, sender_3hop_cycle_count=0,
            sender_fan_in_score=0.0, sender_fan_out_score=0.0,
            sender_pagerank=0.0, sender_betweenness=0.0,
            sender_community_id=-1, sender_community_size=0,
            sender_community_density=0.0, sender_community_risk_score=0.0,
            sender_benford_chi2_community=0.0,
        )
    defaults.update(
        receiver_in_degree=0, receiver_out_degree=0,
        receiver_community_id=-1, receiver_community_risk_score=0.0,
        edge_creates_cycle=False, cycle_length=0, shared_community=False,
    )
    defaults.update(overrides)
    return LiveGraphFeatureVector(
        transaction_id=transaction_id,
        sender_account=sender,
        receiver_account=receiver,
        **defaults,
    )


# --------------------------------------------------------------------------- #
# Score behaviour
# --------------------------------------------------------------------------- #


def test_score_returns_valid_output(trained_d6, inferencer):
    row = trained_d6["behavioral"].iloc[0]
    features = _live_feature_vector_from_row(row)
    graph = _zero_graph_vec(features.transaction_id, features.sender_account)
    out = inferencer.score(features, graph)
    assert isinstance(out, BehavioralAnomalyOutput)
    assert 0.0 <= out.anomaly_score <= 100.0
    for sub in (out.iso_score, out.lof_score, out.autoencoder_score):
        assert 0.0 <= sub <= 100.0


def test_anomaly_drivers_subset_of_features(trained_d6, inferencer):
    row = trained_d6["behavioral"].iloc[0]
    features = _live_feature_vector_from_row(row)
    graph = _zero_graph_vec(features.transaction_id, features.sender_account)
    out = inferencer.score(features, graph)
    assert len(out.anomaly_drivers) == 3
    feat_set = set(inferencer.feature_names)
    assert all(d in feat_set for d in out.anomaly_drivers)


def test_normal_vector_scores_low(trained_d6, inferencer):
    """A near-baseline transaction should score below 30 on the ensemble.

    Uses the median row from the training set with the sender's actual L2A
    graph features (zero-graph defaults would themselves look anomalous to
    the autoencoder, defeating the test).
    """
    behavioral = trained_d6["behavioral"]
    graph_features = trained_d6["graph_features"]
    # Median amount → a sample that lives at the centre of the training mass
    behavioral_sorted = behavioral.sort_values("amount_raw").reset_index(drop=True)
    row = behavioral_sorted.iloc[len(behavioral_sorted) // 2]
    features = _live_feature_vector_from_row(row)
    graph = _realistic_graph_vec(features.transaction_id, features.sender_account, graph_features)
    out = inferencer.score(features, graph)
    assert out.anomaly_score < 30, (
        f"baseline-like sample scored {out.anomaly_score:.1f}; "
        f"sub-scores iso={out.iso_score:.1f}, lof={out.lof_score:.1f}, "
        f"ae={out.autoencoder_score:.1f}"
    )


def test_extreme_zscore_scores_high_and_drives(trained_d6, inferencer):
    """A vector with one feature pushed many standard deviations from the
    training mean must score > 60 AND list that feature in the drivers."""
    behavioral = trained_d6["behavioral"]
    row = behavioral.iloc[0]
    extreme_feature = "amount_raw"
    # Push amount_raw to a value ≈100x the baseline maximum
    extreme_value = float(behavioral["amount_raw"].max()) * 100.0
    features = _live_feature_vector_from_row(row, **{extreme_feature: extreme_value})
    # Pair with very high graph-anomaly values too — pushes IF/LOF further out
    graph = _zero_graph_vec(
        features.transaction_id,
        features.sender_account,
        sender_2hop_cycle_count=100,
        sender_3hop_cycle_count=100,
        sender_fan_in_score=0.95,
        sender_fan_out_score=0.95,
    )
    out = inferencer.score(features, graph)
    assert out.anomaly_score > 60, (
        f"extreme outlier scored {out.anomaly_score:.1f}; "
        f"iso={out.iso_score:.1f}, lof={out.lof_score:.1f}, ae={out.autoencoder_score:.1f}"
    )
    assert extreme_feature in out.anomaly_drivers, (
        f"top drivers were {out.anomaly_drivers}, expected to include {extreme_feature}"
    )


# --------------------------------------------------------------------------- #
# Artifact contract
# --------------------------------------------------------------------------- #


def test_all_artifacts_present_helper(trained_d6):
    assert d6b_inference.BehavioralAnomalyInferencer.all_artifacts_present(trained_d6["store"])


def test_feature_names_consistent_across_artifacts(inferencer):
    assert len(inferencer.feature_names) == 52


def test_module_does_not_import_node2vec_or_torch_geometric():
    """L4 inference uses torch but must not pull node2vec / torch_geometric."""
    import importlib
    import sys

    mod = "src.system2_detection.layer4_anomaly.d6b_inference"
    if mod in sys.modules:
        del sys.modules[mod]
    before = set(sys.modules.keys())
    importlib.import_module(mod)
    newly = set(sys.modules.keys()) - before
    banned = {"node2vec", "torch_geometric", "torch_geometric_temporal"}
    for name in newly:
        top = name.split(".")[0]
        assert top not in banned, f"L4 inference unexpectedly loaded {name}"
