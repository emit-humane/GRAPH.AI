"""Tests for L2B (D4b) graph analytics engine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.system1_generator import g1_account_builder, g2_normal_generator, g3_scenario_injector, g4_splitter
from src.system1_generator.common import CONFIG_PATH, GeneratorConfig
from src.system2_detection.layer2_graph.d4a_graph_preprocessor import build_graph_features
from src.system2_detection.layer2_graph.d4b_graph_analytics import (
    COMMUNITY_PROFILE_COLUMNS,
    SUSPICIOUS_PATH_COLUMNS,
    VALID_PATTERNS,
    PATTERN_CIRCULAR,
    PATTERN_LAYERING_CHAIN,
    build_community_profiles_and_paths,
    save_artifacts,
)
from src.system2_detection.shared import s2_multigraph_builder


CYCLE_PATTERNS = {"circular_laundering", "round_tripping", "fraud_ring"}


@pytest.fixture(scope="session")
def d4b_fixture(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("d4b-fixture")
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
    gf = build_graph_features(G, betweenness_sample_k=None)
    profiles, paths = build_community_profiles_and_paths(G, gf)
    save_artifacts(profiles, paths, out_dir=out_dir / "artifacts")

    return {
        "graph": G,
        "graph_features": gf,
        "profiles": profiles,
        "paths": paths,
        "suspicious": suspicious,
        "hist_df": hist_df,
        "out_dir": out_dir,
    }


# --------------------------------------------------------------------------- #
# community_profiles — schema and shape
# --------------------------------------------------------------------------- #


def test_community_profiles_columns_exact(d4b_fixture):
    df = d4b_fixture["profiles"]
    assert list(df.columns) == list(COMMUNITY_PROFILE_COLUMNS)
    assert df.shape[1] == 10


def test_community_profiles_row_count_matches_n_communities(d4b_fixture):
    df = d4b_fixture["profiles"]
    gf = d4b_fixture["graph_features"]
    assert len(df) == gf["community_id"].nunique()


def test_community_ids_unique(d4b_fixture):
    df = d4b_fixture["profiles"]
    assert df["community_id"].is_unique


def test_artifacts_persisted(d4b_fixture):
    out_dir = d4b_fixture["out_dir"] / "artifacts"
    assert (out_dir / "community_profiles.parquet").exists()
    assert (out_dir / "suspicious_paths.parquet").exists()


# --------------------------------------------------------------------------- #
# community_profiles — semantic checks
# --------------------------------------------------------------------------- #


def test_dominant_pattern_values_in_valid_set(d4b_fixture):
    df = d4b_fixture["profiles"]
    bad = set(df["dominant_pattern"]) - set(VALID_PATTERNS)
    assert not bad, f"unknown dominant_pattern values: {bad}"


def test_risk_score_in_unit_range(d4b_fixture):
    df = d4b_fixture["profiles"]
    assert (df["community_risk_score"] >= 0.0).all()
    assert (df["community_risk_score"] <= 100.0).all()


def test_community_density_in_unit_range(d4b_fixture):
    df = d4b_fixture["profiles"]
    assert (df["community_density"] >= 0.0).all()
    assert (df["community_density"] <= 1.0).all()


def test_size_matches_member_list_length(d4b_fixture):
    df = d4b_fixture["profiles"]
    for _, row in df.iterrows():
        assert len(row["member_accounts"]) == row["community_size"]


def test_has_cycle_implies_positive_max_cycle_length(d4b_fixture):
    df = d4b_fixture["profiles"]
    cy = df[df["has_cycle"]]
    assert (cy["max_cycle_length"] >= 2).all(), "has_cycle but max_cycle_length < 2"


def test_no_cycle_implies_max_cycle_length_zero(d4b_fixture):
    df = d4b_fixture["profiles"]
    nc = df[~df["has_cycle"]]
    assert (nc["max_cycle_length"] == 0).all()


def test_circular_communities_have_at_least_one_known_ring_member(d4b_fixture):
    """Communities flagged circular should contain members of cycle-creating
    rings from the ground truth — directly read here for verification."""
    profiles = d4b_fixture["profiles"]
    suspicious = d4b_fixture["suspicious"]

    circular = profiles[profiles["dominant_pattern"] == PATTERN_CIRCULAR]
    if circular.empty:
        pytest.skip("fixture produced no circular communities")

    cycle_accounts = set(
        pd.unique(
            pd.concat([
                suspicious[suspicious["synthetic_pattern_type"].isin(CYCLE_PATTERNS)]["sender_account"],
                suspicious[suspicious["synthetic_pattern_type"].isin(CYCLE_PATTERNS)]["receiver_account"],
            ])
        )
    )
    if not cycle_accounts:
        pytest.skip("fixture produced no cycle-ring accounts")

    n_circular_with_cycle_member = 0
    for _, row in circular.iterrows():
        if set(row["member_accounts"]) & cycle_accounts:
            n_circular_with_cycle_member += 1
    assert n_circular_with_cycle_member >= 1


# --------------------------------------------------------------------------- #
# suspicious_paths — schema and shape
# --------------------------------------------------------------------------- #


def test_suspicious_paths_columns_exact(d4b_fixture):
    df = d4b_fixture["paths"]
    assert list(df.columns) == list(SUSPICIOUS_PATH_COLUMNS)
    assert df.shape[1] == 6


def test_path_length_in_2_to_8(d4b_fixture):
    df = d4b_fixture["paths"]
    if df.empty:
        pytest.skip("no paths produced — community classification yielded no circular/layering communities")
    assert (df["path_length"] >= 2).all()
    assert (df["path_length"] <= 8).all()


def test_path_pattern_type_valid(d4b_fixture):
    df = d4b_fixture["paths"]
    if df.empty:
        pytest.skip("no paths")
    allowed = {"layering_chain", "round_trip", "scatter_gather"}
    bad = set(df["pattern_type"]) - allowed
    assert not bad, f"unknown pattern_type values: {bad}"


def test_path_nodes_present_in_graph(d4b_fixture):
    G = d4b_fixture["graph"]
    df = d4b_fixture["paths"]
    if df.empty:
        pytest.skip("no paths")
    for nodes in df["path_nodes"].head(50):
        for n in nodes:
            assert G.has_node(n), f"path references missing node {n}"


def test_path_edges_have_matching_length(d4b_fixture):
    df = d4b_fixture["paths"]
    if df.empty:
        pytest.skip("no paths")
    for _, row in df.head(30).iterrows():
        # path_length is the number of edges, which equals len(nodes) - 1
        assert len(row["path_edges"]) == row["path_length"], (
            f"path_length={row['path_length']} but edges={len(row['path_edges'])}"
        )
        assert len(row["path_nodes"]) == row["path_length"] + 1


def test_path_ids_unique(d4b_fixture):
    df = d4b_fixture["paths"]
    if df.empty:
        pytest.skip("no paths")
    assert df["path_id"].nunique() == len(df)


def test_total_value_positive(d4b_fixture):
    df = d4b_fixture["paths"]
    if df.empty:
        pytest.skip("no paths")
    assert (df["total_value"] > 0).all()


# --------------------------------------------------------------------------- #
# Layer boundary — no ML
# --------------------------------------------------------------------------- #


def test_module_does_not_load_torch_or_node2vec():
    import importlib
    import sys

    mod_name = "src.system2_detection.layer2_graph.d4b_graph_analytics"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    before = set(sys.modules.keys())
    importlib.import_module(mod_name)
    newly = set(sys.modules.keys()) - before
    banned = {"torch", "torch_geometric", "node2vec"}
    for name in newly:
        top = name.split(".")[0]
        assert top not in banned, f"L2B unexpectedly loaded {name}"
