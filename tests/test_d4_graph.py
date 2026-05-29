"""Tests for L2A (D4) graph feature preprocessor.

Uses a small generated dataset built once per session. Verifies:
    * one row per node in the multigraph
    * exactly 20 columns matching the spec
    * pagerank distribution sums to ~1
    * Infomap returned dense (non-singleton-dominated) community ids
    * 2/3-hop cycle counts are positive for accounts known to be in
      cycle-creating fraud rings (read via the test-only hidden ground truth)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.system1_generator import (
    g1_account_builder,
    g2_normal_generator,
    g3_scenario_injector,
    g4_splitter,
)
from src.system1_generator.common import CONFIG_PATH, GeneratorConfig
from src.system2_detection.layer2_graph.d4a_graph_preprocessor import (
    GRAPH_FEATURE_COLUMNS,
    build_graph_features,
    save_graph_features,
)
from src.system2_detection.shared import s2_multigraph_builder


# Cycle-creating typologies — for these we expect at least some ring members
# to have non-zero 2-hop or 3-hop cycle counts in the historical graph.
CYCLE_PATTERNS = {"circular_laundering", "round_tripping", "fraud_ring"}


@pytest.fixture(scope="session")
def d4_fixture(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("d4-fixture")
    cfg = GeneratorConfig.from_file(CONFIG_PATH).override(
        num_accounts=500,
        num_historical_transactions=10_000,
        num_stream_transactions=2_500,
    )
    ctx = g1_account_builder.build_accounts(cfg)
    normal_df = g2_normal_generator.generate_normal(cfg, ctx)
    combined, suspicious, gt = g3_scenario_injector.inject_scenarios(cfg, ctx, normal_df)
    hist_df, _ = g4_splitter.split_and_export(combined, gt, out_dir=out_dir)

    G, _, _ = s2_multigraph_builder.build_multigraph(
        historical_csv=out_dir / "historical_transactions.csv", accounts_df=ctx.accounts
    )
    # Exact betweenness on small graph keeps the test deterministic.
    df = build_graph_features(G, betweenness_sample_k=None)
    save_graph_features(df, out_dir=out_dir / "artifacts")

    return {
        "graph": G,
        "features": df,
        "hist_df": hist_df,
        "ground_truth": gt,
        "suspicious": suspicious,
        "out_dir": out_dir,
    }


# --------------------------------------------------------------------------- #
# Schema / shape
# --------------------------------------------------------------------------- #


def test_row_count_matches_node_count(d4_fixture):
    G = d4_fixture["graph"]
    df = d4_fixture["features"]
    assert len(df) == G.number_of_nodes()


def test_exactly_20_columns_in_canonical_order(d4_fixture):
    df = d4_fixture["features"]
    assert df.shape[1] == 20
    assert list(df.columns) == list(GRAPH_FEATURE_COLUMNS)


def test_account_id_unique_and_covers_all_nodes(d4_fixture):
    df = d4_fixture["features"]
    G = d4_fixture["graph"]
    assert df["account_id"].is_unique
    assert set(df["account_id"]) == set(G.nodes())


def test_artifact_persisted(d4_fixture):
    p = d4_fixture["out_dir"] / "artifacts" / "graph_features.parquet"
    assert p.exists()
    loaded = pd.read_parquet(p)
    assert list(loaded.columns) == list(GRAPH_FEATURE_COLUMNS)


# --------------------------------------------------------------------------- #
# Numerical properties
# --------------------------------------------------------------------------- #


def test_pagerank_sums_to_one(d4_fixture):
    pr_sum = float(d4_fixture["features"]["pagerank_score"].sum())
    assert abs(pr_sum - 1.0) < 1e-3, f"PageRank does not sum to 1 ({pr_sum:.4f})"


def test_betweenness_in_unit_interval(d4_fixture):
    bc = d4_fixture["features"]["betweenness_centrality"]
    assert (bc >= 0.0).all()
    assert (bc <= 1.0).all()


def test_clustering_in_unit_interval(d4_fixture):
    cc = d4_fixture["features"]["clustering_coefficient"]
    assert (cc >= 0.0).all() and (cc <= 1.0).all()


def test_fan_scores_complement(d4_fixture):
    """fan_in + fan_out ≈ 1 for any node with at least one connection."""
    df = d4_fixture["features"]
    active = df[(df["in_degree_unique"] + df["out_degree_unique"]) > 0]
    s = active["fan_in_score"] + active["fan_out_score"]
    assert ((s - 1.0).abs() < 1e-3).all()


def test_volume_asymmetry_in_pm1_range(d4_fixture):
    va = d4_fixture["features"]["volume_asymmetry"]
    assert (va >= -1.001).all() and (va <= 1.001).all()


# --------------------------------------------------------------------------- #
# Community structure
# --------------------------------------------------------------------------- #


def test_community_ids_are_dense(d4_fixture):
    """Infomap should NOT degenerate to a singleton-per-node assignment.

    Healthy directed Infomap on our generator output groups most nodes into
    a moderate number of communities (a handful to a few dozen), not one
    community per node.
    """
    df = d4_fixture["features"]
    n_nodes = len(df)
    n_communities = df["community_id"].nunique()
    # Dense = many fewer communities than nodes
    assert n_communities < n_nodes * 0.5, (
        f"{n_communities} communities for {n_nodes} nodes — Infomap degenerate"
    )
    # And the largest community covers a meaningful fraction
    largest = df["community_id"].value_counts().iloc[0]
    assert largest >= 5


def test_community_size_matches_membership_count(d4_fixture):
    """The community_size column on each row must equal the number of rows
    sharing that community_id (consistency check)."""
    df = d4_fixture["features"]
    membership = df.groupby("community_id").size().to_dict()
    for cid, expected in membership.items():
        actual = int(df.loc[df["community_id"] == cid, "community_size"].iloc[0])
        assert actual == expected, f"community {cid}: column says {actual}, membership is {expected}"


def test_community_density_in_unit_interval(d4_fixture):
    cd = d4_fixture["features"]["community_density"]
    assert (cd >= 0.0).all() and (cd <= 1.0).all()


# --------------------------------------------------------------------------- #
# Cycle counts — verified against known fraud rings (ground-truth read)
# --------------------------------------------------------------------------- #


def test_cycle_counts_positive_for_known_cycle_rings(d4_fixture):
    """Members of cycle-creating typologies (circular_laundering,
    round_tripping, fraud_ring) should show non-zero 2-hop or 3-hop cycle
    counts in the L2A output.

    We intentionally read hidden_ground_truth here — this is a test of L2A
    correctness, not a production code path.
    """
    suspicious = d4_fixture["suspicious"]
    df = d4_fixture["features"].set_index("account_id")

    cycle_rings = suspicious[
        suspicious["synthetic_pattern_type"].isin(CYCLE_PATTERNS)
    ]
    if cycle_rings.empty:
        pytest.skip("fixture produced no cycle-creating rings")

    # Pool of all accounts participating in cycle-creating rings
    cycle_accounts = (
        pd.unique(
            pd.concat([cycle_rings["sender_account"], cycle_rings["receiver_account"]])
        )
        .tolist()
    )
    cycle_accounts = [a for a in cycle_accounts if a in df.index]
    assert cycle_accounts, "no cycle-ring accounts present in graph_features output"

    # At least one of them must register a cycle
    cy_counts = df.loc[cycle_accounts, ["2hop_cycle_count", "3hop_cycle_count"]].sum(axis=1)
    n_with_cycle = int((cy_counts > 0).sum())
    n_total = int(len(cy_counts))
    assert n_with_cycle >= max(2, n_total // 10), (
        f"only {n_with_cycle}/{n_total} cycle-ring accounts have any 2- or 3-hop cycle count"
    )


def test_two_hop_cycle_count_bounded_by_unique_neighbors(d4_fixture):
    """2-hop cycles A→B→A require distinct B, so count cannot exceed
    min(in_degree_unique, out_degree_unique)."""
    df = d4_fixture["features"]
    upper = df[["in_degree_unique", "out_degree_unique"]].min(axis=1)
    assert (df["2hop_cycle_count"] <= upper + 1e-9).all()


# --------------------------------------------------------------------------- #
# Bench (no ML imported)
# --------------------------------------------------------------------------- #


def test_module_does_not_import_torch_or_node2vec():
    """L2A must be pure graph heuristics — no ML libraries.

    Loading the module must not pull torch / torch_geometric / node2vec.
    """
    import importlib
    import sys

    # Make sure module is fresh
    mod_name = "src.system2_detection.layer2_graph.d4a_graph_preprocessor"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    # Pre-record what was already loaded before we import L2A
    before = set(sys.modules.keys())
    importlib.import_module(mod_name)
    after = set(sys.modules.keys())
    newly_loaded = after - before
    banned = {"torch", "torch_geometric", "node2vec"}
    for name in newly_loaded:
        top = name.split(".")[0]
        assert top not in banned, f"L2A unexpectedly loaded {name}"
