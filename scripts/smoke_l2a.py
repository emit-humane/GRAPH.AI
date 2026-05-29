"""Tiny smoke run of L2A on a small generated dataset."""

import joblib

from src.system1_generator import g1_account_builder, g2_normal_generator, g3_scenario_injector, g4_splitter
from src.system1_generator.common import CONFIG_PATH, GeneratorConfig, DATA_DIR
from src.system2_detection.layer2_graph.d4a_graph_preprocessor import build_graph_features, GRAPH_FEATURE_COLUMNS
from src.system2_detection.shared import s2_multigraph_builder


def main():
    cfg = GeneratorConfig.from_file(CONFIG_PATH).override(
        num_accounts=400, num_historical_transactions=8_000, num_stream_transactions=2_000
    )
    ctx = g1_account_builder.build_accounts(cfg)
    normal_df = g2_normal_generator.generate_normal(cfg, ctx)
    combined, _, gt = g3_scenario_injector.inject_scenarios(cfg, ctx, normal_df)
    out_dir = DATA_DIR / "_l2a_smoke"
    hist, _ = g4_splitter.split_and_export(combined, gt, out_dir=out_dir)
    G, _, _ = s2_multigraph_builder.build_multigraph(
        historical_csv=out_dir / "historical_transactions.csv", accounts_df=ctx.accounts
    )
    print(f"[smoke] nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    df = build_graph_features(G, betweenness_sample_k=None)
    print(f"[smoke] features rows={len(df)} cols={df.shape[1]}")
    assert list(df.columns) == list(GRAPH_FEATURE_COLUMNS)
    print(f"[smoke] pagerank_sum={df['pagerank_score'].sum():.3f}")
    print(f"[smoke] num_communities={df['community_id'].nunique()}")
    print(f"[smoke] 2hop>0 nodes={(df['2hop_cycle_count'] > 0).sum()}")
    print(f"[smoke] 3hop>0 nodes={(df['3hop_cycle_count'] > 0).sum()}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
