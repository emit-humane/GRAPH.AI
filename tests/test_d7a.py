"""Tests for Layer 5 (D7) — TGN primary + Node2Vec fallback + GraphSAGE mid-tier.

The TGN model is exercised in ``--smoke`` mode (1000 edges × 2 epochs) on a
small generator fixture so each test session runs in under a minute.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from src.system1_generator import g1_account_builder, g2_normal_generator, g3_scenario_injector, g4_splitter
from src.system1_generator.common import CONFIG_PATH, GeneratorConfig
from src.system2_detection.layer2_graph.d4a_graph_preprocessor import build_graph_features
from src.system2_detection.layer5_gnn import d7_node2vec, d7a_tgn_training
from src.system2_detection.shared import s2_multigraph_builder


@pytest.fixture(scope="session")
def d7_fixture(tmp_path_factory):
    work = tmp_path_factory.mktemp("d7-fixture")
    base = GeneratorConfig.from_file(CONFIG_PATH)
    cfg = base.override(
        seed=base.seed + 8000,
        num_accounts=200,
        num_historical_transactions=4_000,
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

    return {
        "work": work,
        "artifacts_dir": artifacts_dir,
        "graph": G,
        "graph_features": gf,
    }


# --------------------------------------------------------------------------- #
# Node2Vec — fallback path
# --------------------------------------------------------------------------- #


def test_node2vec_embeddings_have_correct_shape(d7_fixture):
    G = d7_fixture["graph"]
    embeddings, index = d7_node2vec.train_node2vec(
        G, dimensions=64, walk_length=10, num_walks=10, seed=42,
    )
    assert embeddings.shape == (G.number_of_nodes(), 64)
    assert embeddings.dtype == np.float32
    assert len(index) == G.number_of_nodes()
    # Indices form a contiguous 0..N-1 mapping
    assert sorted(index.values()) == list(range(G.number_of_nodes()))


def test_node2vec_save_artifacts(d7_fixture, tmp_path):
    G = d7_fixture["graph"]
    embeddings, index = d7_node2vec.train_node2vec(
        G, dimensions=64, walk_length=10, num_walks=10, seed=42,
    )
    out_dir = tmp_path / "n2v"
    paths = d7_node2vec.save_artifacts(embeddings, index, out_dir=out_dir)
    assert (out_dir / "node2vec_embeddings.npy").exists()
    assert (out_dir / "node2vec_index.json").exists()
    loaded = np.load(out_dir / "node2vec_embeddings.npy")
    np.testing.assert_array_equal(loaded, embeddings)


# --------------------------------------------------------------------------- #
# TGN — primary model in smoke mode
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def trained_tgn(d7_fixture):
    """Run TGN end-to-end on the fixture in smoke mode; cache artifacts."""
    artifacts_dir = d7_fixture["artifacts_dir"]
    cfg = d7a_tgn_training.TGNConfig(
        smoke=True, smoke_edge_cap=500, smoke_epochs=2,
        batch_size=128,
    )
    paths = d7a_tgn_training.run(
        graph_path=artifacts_dir / "transaction_multigraph.pkl",
        graph_features_path=artifacts_dir / "graph_features.parquet",
        out_dir=artifacts_dir,
        cfg=cfg,
    )
    return {"artifacts_dir": artifacts_dir, "paths": paths, "cfg": cfg}


def test_tgn_all_artifacts_persisted(trained_tgn):
    d = trained_tgn["artifacts_dir"]
    for name in (
        "tgn_model.pt",
        "node_embeddings.npy",
        "node_embedding_index.json",
        "tgn_memory_state.pkl",
    ):
        assert (d / name).exists(), f"missing artifact {name}"


def test_node_embeddings_shape(trained_tgn, d7_fixture):
    emb = np.load(trained_tgn["artifacts_dir"] / "node_embeddings.npy")
    n_nodes = d7_fixture["graph"].number_of_nodes()
    assert emb.shape == (n_nodes, 64)
    assert emb.dtype == np.float32


def test_embedding_index_round_trips(trained_tgn, d7_fixture):
    with open(trained_tgn["artifacts_dir"] / "node_embedding_index.json", "r") as fh:
        idx = json.load(fh)
    assert len(idx) == d7_fixture["graph"].number_of_nodes()
    assert sorted(idx.values()) == list(range(d7_fixture["graph"].number_of_nodes()))


def test_memory_state_pickle_round_trips(trained_tgn, d7_fixture):
    mem = joblib.load(trained_tgn["artifacts_dir"] / "tgn_memory_state.pkl")
    assert len(mem) == d7_fixture["graph"].number_of_nodes()
    # All per-node memory tensors have the expected shape (64,)
    sample = next(iter(mem.values()))
    assert isinstance(sample, np.ndarray)
    assert sample.shape == (64,)
    assert sample.dtype == np.float32


def test_tgn_checkpoint_forward_works(trained_tgn, d7_fixture):
    """Reload tgn_model.pt and run a fresh forward pass on a tiny batch."""
    ckpt_path = trained_tgn["artifacts_dir"] / "tgn_model.pt"
    model, blob, node_index = d7a_tgn_training.load_tgn(ckpt_path)
    assert model.n_nodes == d7_fixture["graph"].number_of_nodes()
    # Synthesise a tiny edge batch
    n_nodes = model.n_nodes
    batch = 4
    src = torch.zeros(batch, dtype=torch.long)
    dst = torch.arange(1, batch + 1, dtype=torch.long).clamp(max=n_nodes - 1)
    ts = torch.linspace(1.0, 2.0, batch)
    edge_feat = torch.zeros(batch, blob["config"]["edge_feat_dim"])
    node_feats = torch.zeros(n_nodes, blob["config"]["node_feat_dim"])

    emb_src, emb_dst = model(src, dst, ts, edge_feat, node_feats)
    assert emb_src.shape == (batch, blob["config"]["embedding_dim"])
    assert emb_dst.shape == (batch, blob["config"]["embedding_dim"])


def test_embedding_index_matches_memory_state_keys(trained_tgn):
    with open(trained_tgn["artifacts_dir"] / "node_embedding_index.json", "r") as fh:
        idx = json.load(fh)
    mem = joblib.load(trained_tgn["artifacts_dir"] / "tgn_memory_state.pkl")
    assert set(idx.keys()) == set(mem.keys())
