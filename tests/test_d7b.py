"""Tests for the Layer 5 (D7b) TGN inferencer.

Train a tiny TGN once per session, then verify:
    * a single score() call returns a finite GNNInferenceOutput with all
      fields in spec range
    * the model's live memory tensor for the sender + receiver MUTATES
      across two successive score() calls (the GRU update is in-place)
    * a cycle-closing event surfaces a structural explanation
    * the node2vec fallback path activates when the TGN artifacts are absent
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.system1_generator import g1_account_builder, g2_normal_generator, g3_scenario_injector, g4_splitter
from src.system1_generator.common import CONFIG_PATH, GeneratorConfig
from src.system2_detection.layer2_graph.d4a_graph_preprocessor import build_graph_features
from src.system2_detection.layer5_gnn import d7_node2vec, d7a_tgn_training, d7b_inference
from src.system2_detection.layer5_gnn.schemas import GNNInferenceOutput
from src.system2_detection.shared import s2_multigraph_builder
from src.system2_detection.shared.s3_artifact_store import ArtifactNotFoundError, ArtifactStore
from src.system2_detection.shared.schemas import (
    FEATURE_COLUMNS,
    LiveFeatureVector,
    LiveGraphFeatureVector,
    TransactionEvent,
)


# --------------------------------------------------------------------------- #
# Session fixture — train a small TGN once into a tempdir
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def trained_tgn(tmp_path_factory):
    work = tmp_path_factory.mktemp("d7b-fixture")
    base = GeneratorConfig.from_file(CONFIG_PATH)
    cfg = base.override(
        seed=base.seed + 9000,
        num_accounts=150,
        num_historical_transactions=2_000,
        num_stream_transactions=0,
        fraud_ratio=0.02,
    )
    ctx = g1_account_builder.build_accounts(cfg)
    normal_df = g2_normal_generator.generate_normal(cfg, ctx)
    combined, _, gt = g3_scenario_injector.inject_scenarios(cfg, ctx, normal_df)
    g4_splitter.split_and_export(combined, gt, out_dir=work)
    historical_csv = work / "historical_transactions.csv"

    G, _, _ = s2_multigraph_builder.build_multigraph(
        historical_csv=historical_csv, accounts_df=ctx.accounts
    )
    gf = build_graph_features(G, betweenness_sample_k=None)
    artifacts_dir = work / "artifacts"
    artifacts_dir.mkdir()
    gf.to_parquet(artifacts_dir / "graph_features.parquet", index=False)
    joblib.dump(G, artifacts_dir / "transaction_multigraph.pkl")

    # Tiny TGN via smoke mode
    tcfg = d7a_tgn_training.TGNConfig(
        smoke=True, smoke_edge_cap=400, smoke_epochs=2,
        batch_size=128,
    )
    d7a_tgn_training.run(
        graph_path=artifacts_dir / "transaction_multigraph.pkl",
        graph_features_path=artifacts_dir / "graph_features.parquet",
        out_dir=artifacts_dir,
        cfg=tcfg,
    )

    return {
        "work": work,
        "artifacts_dir": artifacts_dir,
        "graph": G,
        "graph_features": gf,
        "accounts": ctx.accounts,
    }


@pytest.fixture
def inferencer(trained_tgn):
    store = ArtifactStore(base_dir=trained_tgn["artifacts_dir"])
    return d7b_inference.TGNInferencer(store=store)


# --------------------------------------------------------------------------- #
# Live*Vector builders
# --------------------------------------------------------------------------- #


def _make_features(txn_id: str, sender: str, **overrides) -> LiveFeatureVector:
    defaults = {col: 0.0 for col in FEATURE_COLUMNS}
    defaults.update({
        "hour_of_day": 12, "day_of_week": 2,
        "amount_leading_digit": 1,
        "tx_velocity_1h": 1.0, "tx_velocity_6h": 1.0, "tx_velocity_24h": 1.0, "tx_velocity_7d": 1.0,
    })
    defaults.update(overrides)
    return LiveFeatureVector(
        transaction_id=txn_id, sender_account=sender,
        scaled_feature_vector=[], **defaults,
    )


def _make_graph_vec(
    txn_id: str, sender: str, receiver: str, *,
    edge_creates_cycle: bool = False, cycle_length: int = 0,
    two_hop_neighborhood: list[str] | None = None,
    two_hop_edge_list: list[tuple] | None = None,
    **overrides,
) -> LiveGraphFeatureVector:
    defaults = dict(
        sender_in_degree=2, sender_out_degree=2,
        sender_in_degree_unique=2, sender_out_degree_unique=2,
        sender_2hop_cycle_count=0, sender_3hop_cycle_count=0,
        sender_fan_in_score=0.1, sender_fan_out_score=0.1,
        sender_pagerank=0.001, sender_betweenness=0.001,
        sender_community_id=0, sender_community_size=10,
        sender_community_density=0.05, sender_community_risk_score=0.0,
        sender_benford_chi2_community=1.0,
        receiver_in_degree=1, receiver_out_degree=1,
        receiver_community_id=0, receiver_community_risk_score=0.0,
        shared_community=True,
    )
    defaults.update(overrides)
    return LiveGraphFeatureVector(
        transaction_id=txn_id, sender_account=sender, receiver_account=receiver,
        edge_creates_cycle=edge_creates_cycle, cycle_length=cycle_length,
        two_hop_neighborhood=two_hop_neighborhood or [],
        two_hop_edge_list=two_hop_edge_list or [],
        **defaults,
    )


def _make_event(txn_id: str, sender: str, receiver: str, **overrides) -> TransactionEvent:
    defaults = dict(
        timestamp=datetime(2025, 4, 1, 12, 0, 0, tzinfo=timezone.utc),
        sender_bank="HDFC", receiver_bank="HDFC",
        sender_country="IN", receiver_country="IN",
        amount=10_000.0, currency="INR",
        transaction_type="UPI", payment_channel="Mobile",
        device_id="dev-test", ip_address="203.0.113.10",
        geo_latitude=19.07, geo_longitude=72.87,
        merchant_category="Grocery", transaction_status="Success",
        is_international=False, amount_leading_digit=1,
    )
    defaults.update(overrides)
    return TransactionEvent(
        transaction_id=txn_id, sender_account=sender, receiver_account=receiver, **defaults,
    )


# --------------------------------------------------------------------------- #
# Sanity / shape
# --------------------------------------------------------------------------- #


def test_inferencer_loads_in_tgn_mode(inferencer):
    assert inferencer.tgn_mode == "tgn"
    assert inferencer.embedding_dim == 64
    assert len(inferencer.node_index) > 0


def test_single_score_call_finite_and_in_range(inferencer, trained_tgn):
    accounts = trained_tgn["accounts"]
    sender = accounts.iloc[0]["account_id"]
    receiver = accounts.iloc[1]["account_id"]
    out = inferencer.score(
        _make_features("tx-001", sender),
        _make_graph_vec("tx-001", sender, receiver),
        _make_event("tx-001", sender, receiver),
    )
    assert isinstance(out, GNNInferenceOutput)
    assert np.isfinite(out.tgn_score)
    assert 0.0 <= out.tgn_score <= 100.0
    assert 0.0 <= out.link_prediction_score <= 1.0
    assert 0.0 <= out.embedding_drift <= 1.0
    assert out.tgn_mode == "tgn"
    assert out.temporal_graph_explanations  # at least one string
    assert "two_hop_edge_count" in out.subgraph_evidence


# --------------------------------------------------------------------------- #
# Memory mutates across calls
# --------------------------------------------------------------------------- #


def test_memory_mutates_across_score_calls(inferencer, trained_tgn):
    accounts = trained_tgn["accounts"]
    sender = accounts.iloc[0]["account_id"]
    receiver = accounts.iloc[2]["account_id"]

    pre_send = inferencer.memory_for(sender)
    pre_recv = inferencer.memory_for(receiver)
    assert pre_send is not None and pre_recv is not None

    inferencer.score(
        _make_features("tx-100", sender),
        _make_graph_vec("tx-100", sender, receiver),
        _make_event("tx-100", sender, receiver, amount=500_000.0),
    )

    post_send = inferencer.memory_for(sender)
    post_recv = inferencer.memory_for(receiver)

    # Both endpoints' memory should have moved
    assert not np.allclose(pre_send, post_send), "sender memory did not mutate"
    assert not np.allclose(pre_recv, post_recv), "receiver memory did not mutate"


# --------------------------------------------------------------------------- #
# Structural explanations
# --------------------------------------------------------------------------- #


def test_cycle_closing_event_yields_structural_explanation(inferencer, trained_tgn):
    accounts = trained_tgn["accounts"]
    sender = accounts.iloc[3]["account_id"]
    receiver = accounts.iloc[4]["account_id"]

    out = inferencer.score(
        _make_features("tx-cyc", sender),
        _make_graph_vec(
            "tx-cyc", sender, receiver,
            edge_creates_cycle=True, cycle_length=3,
            two_hop_neighborhood=[sender, receiver, "A", "B"],
            two_hop_edge_list=[(sender, receiver, "ts1", 100.0)],
        ),
        _make_event("tx-cyc", sender, receiver),
    )
    joined = " ".join(out.temporal_graph_explanations).lower()
    assert "cycle" in joined, f"expected a cycle-closure explanation, got: {out.temporal_graph_explanations}"
    assert out.subgraph_evidence["edge_creates_cycle"] is True
    assert out.subgraph_evidence["cycle_length"] == 3


def test_fan_in_pattern_surfaces_in_explanations(inferencer, trained_tgn):
    accounts = trained_tgn["accounts"]
    sender = accounts.iloc[5]["account_id"]
    receiver = accounts.iloc[6]["account_id"]

    out = inferencer.score(
        _make_features("tx-fanin", sender),
        _make_graph_vec(
            "tx-fanin", sender, receiver,
            receiver_in_degree_unique_24h=8, receiver_inflow_amount_cv=0.20,
        ),
        _make_event("tx-fanin", sender, receiver),
    )
    joined = " ".join(out.temporal_graph_explanations).lower()
    assert "fan-in" in joined or "collection" in joined


# --------------------------------------------------------------------------- #
# Node2Vec fallback
# --------------------------------------------------------------------------- #


def test_node2vec_fallback_activates_when_tgn_artifacts_absent(trained_tgn, tmp_path):
    """Drop TGN artifacts into a fresh tempdir but only keep a node2vec
    embedding + index — the inferencer must pick up the fallback path and
    set tgn_mode = "node2vec_fallback"."""
    src_dir = trained_tgn["artifacts_dir"]
    G = trained_tgn["graph"]

    fallback_dir = tmp_path / "fallback_only"
    fallback_dir.mkdir()
    # Train a tiny node2vec into the fallback dir
    embeddings, index = d7_node2vec.train_node2vec(
        G, dimensions=64, walk_length=5, num_walks=5, seed=42,
    )
    d7_node2vec.save_artifacts(embeddings, index, out_dir=fallback_dir)

    store = ArtifactStore(base_dir=fallback_dir)
    inf = d7b_inference.TGNInferencer(store=store)
    assert inf.tgn_mode == "node2vec_fallback"

    accounts = trained_tgn["accounts"]
    sender = accounts.iloc[0]["account_id"]
    receiver = accounts.iloc[1]["account_id"]
    out = inf.score(
        _make_features("tx-fb", sender),
        _make_graph_vec("tx-fb", sender, receiver),
        _make_event("tx-fb", sender, receiver),
    )
    assert out.tgn_mode == "node2vec_fallback"
    assert 0.0 <= out.tgn_score <= 100.0
    # Explanation should mention the fallback
    joined = " ".join(out.temporal_graph_explanations).lower()
    assert "fallback" in joined or "node2vec" in joined


def test_missing_all_artifacts_raises(tmp_path):
    """No TGN, no node2vec → constructor refuses to start."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    store = ArtifactStore(base_dir=empty_dir)
    with pytest.raises(ArtifactNotFoundError):
        d7b_inference.TGNInferencer(store=store)


# --------------------------------------------------------------------------- #
# Layer boundary — the only L5 module allowed to import torch + torch_geometric.
# (No banned-import test here because L5 IS where torch lives — that's the point.)
# --------------------------------------------------------------------------- #


def test_any_artifacts_present_helper(trained_tgn):
    store = ArtifactStore(base_dir=trained_tgn["artifacts_dir"])
    assert d7b_inference.TGNInferencer.any_artifacts_present(store)
