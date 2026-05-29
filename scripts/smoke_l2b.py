"""Tiny smoke run of L2B."""

from src.system1_generator import g1_account_builder, g2_normal_generator, g3_scenario_injector, g4_splitter
from src.system1_generator.common import CONFIG_PATH, GeneratorConfig, DATA_DIR
from src.system2_detection.layer2_graph.d4a_graph_preprocessor import build_graph_features
from src.system2_detection.layer2_graph.d4b_graph_analytics import (
    COMMUNITY_PROFILE_COLUMNS,
    SUSPICIOUS_PATH_COLUMNS,
    build_community_profiles_and_paths,
)
from src.system2_detection.shared import s2_multigraph_builder


def main():
    cfg = GeneratorConfig.from_file(CONFIG_PATH).override(
        num_accounts=400, num_historical_transactions=8_000, num_stream_transactions=2_000
    )
    ctx = g1_account_builder.build_accounts(cfg)
    normal_df = g2_normal_generator.generate_normal(cfg, ctx)
    combined, _, gt = g3_scenario_injector.inject_scenarios(cfg, ctx, normal_df)
    out_dir = DATA_DIR / "_l2b_smoke"
    hist, _ = g4_splitter.split_and_export(combined, gt, out_dir=out_dir)
    G, _, _ = s2_multigraph_builder.build_multigraph(
        historical_csv=out_dir / "historical_transactions.csv", accounts_df=ctx.accounts
    )
    gf = build_graph_features(G, betweenness_sample_k=None)
    profiles, paths = build_community_profiles_and_paths(G, gf)
    print(f"[smoke] graph nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    print(f"[smoke] communities: {len(profiles)}")
    print(f"[smoke] community profile cols={list(profiles.columns)}")
    print(f"[smoke] paths: {len(paths)}")
    print(f"[smoke] path cols={list(paths.columns)}")
    print(f"[smoke] patterns: {profiles['dominant_pattern'].value_counts().to_dict()}")
    print(f"[smoke] risk score range: [{profiles['community_risk_score'].min():.1f}, {profiles['community_risk_score'].max():.1f}]")
    if len(paths) > 0:
        print(f"[smoke] path length range: [{paths['path_length'].min()}, {paths['path_length'].max()}]")
        print(f"[smoke] path pattern_type values: {paths['pattern_type'].unique().tolist()}")


if __name__ == "__main__":
    main()
